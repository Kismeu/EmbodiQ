"""The one runtime module: INT4 main branch (+ optional AWQ scale, + optional
HSA high-precision residual).

Three properties this has to keep, all of them for FastWAM's sake, which runs
``torch.compile(fullgraph=True, mode="reduce-overhead")`` by default:

* **No graph break.** The kernel is reached through a registered custom op with
  a fake (meta) implementation, so Dynamo treats it as opaque-but-typed.
* **Static tensor addresses.** Every quantization tensor is a registered buffer
  allocated once at load. Building an index tensor inside ``forward`` is the
  single most likely way to silently break CUDA-graph capture.
* **No returned graph-owned buffer.** We always allocate a fresh output.

The AWQ scale stays *inside* the module (``y = (x / s) @ quant(W * s)``) rather
than being folded into a preceding op. Folding targets largely do not exist
here: FastWAM's norms are ``elementwise_affine=False`` (no affine to fold into),
its ``self_attn.o`` is applied outside the block, and Qwen2.5-0.5B is GQA(14/2)
so ``o_proj`` cannot fold into ``v_proj``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from embodiq.quant.pack import (
    dequantize_int4, group_quantize_int4, quantize_int8_per_channel, to_tinygemm,
)

_OP_REGISTERED = False
# Module-level reference: a torch.library.Library that goes out of scope takes
# its registrations with it, and the op then vanishes at first call.
_LIB = None


def _register_op() -> None:
    """Register the fused call as a custom op so fullgraph=True survives it."""
    global _OP_REGISTERED, _LIB
    if _OP_REGISTERED:
        return
    lib = _LIB = torch.library.Library("embodiq", "FRAGMENT")
    lib.define(
        "w4a16(Tensor x, Tensor packed, Tensor scales_zeros, Tensor? s_inv, "
        "Tensor? res_idx, Tensor? res_val, Tensor? bias, int group_size, int out_features) -> Tensor"
    )

    def _impl(x, packed, scales_zeros, s_inv, res_idx, res_val, bias, group_size, out_features):
        orig = x.shape
        xf = x.reshape(-1, orig[-1])
        if s_inv is not None:
            xf = xf * s_inv
        y = torch.ops.aten._weight_int4pack_mm(
            xf.to(torch.bfloat16), packed, group_size, scales_zeros
        )
        if res_idx is not None and res_val is not None:
            # gather + small dense GEMM. Deliberately NOT a contiguous slice:
            # pre-permuting input channels would shift g=128 group boundaries and
            # change the main branch's quantization error, so the HSA main branch
            # would stop being the same object as the AWQ row it is compared to.
            y = y + torch.nn.functional.linear(xf.index_select(1, res_idx), res_val)
        if bias is not None:
            y = y + bias
        return y.reshape(*orig[:-1], out_features).to(x.dtype)

    def _meta(x, packed, scales_zeros, s_inv, res_idx, res_val, bias, group_size, out_features):
        return x.new_empty((*x.shape[:-1], out_features))

    lib.impl("w4a16", _impl, "CUDA")
    lib.impl("w4a16", _impl, "CPU")
    # register_fake on torch >= 2.5 (FastWAM's 2.7 prefix); impl_abstract is the
    # deprecated spelling still needed on the torch 2.2 prefixes.
    if hasattr(torch.library, "register_fake"):
        torch.library.register_fake("embodiq::w4a16", _meta)
    else:
        torch.library.impl_abstract("embodiq::w4a16", _meta)
    _OP_REGISTERED = True


class QuantLinear(nn.Module):
    """Drop-in for nn.Linear. Build with :meth:`from_linear`."""

    def __init__(self, in_features: int, out_features: int, group_size: int) -> None:
        super().__init__()
        _register_op()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        *,
        input_scale: Optional[torch.Tensor] = None,   # AWQ s, per input channel
        residual_idx: Optional[torch.Tensor] = None,  # HSA support set
        residual_val: Optional[torch.Tensor] = None,  # [out, k] fp16
    ) -> "QuantLinear":
        w = linear.weight.detach()
        dev = w.device
        m = cls(linear.in_features, linear.out_features, group_size)

        w_eff = w.to(torch.float32)
        if input_scale is not None:
            w_eff = w_eff * input_scale.to(torch.float32).unsqueeze(0)

        q, s, z = group_quantize_int4(w_eff, group_size)
        packed, sz = to_tinygemm(q, s, z, group_size)
        m.register_buffer("packed", packed)
        m.register_buffer("scales_zeros", sz)
        m.register_buffer(
            "s_inv",
            (1.0 / input_scale.to(torch.bfloat16)).to(dev) if input_scale is not None else None,
        )
        m.register_buffer("res_idx", residual_idx.to(dev).to(torch.int32) if residual_idx is not None else None)
        m.register_buffer("res_val", residual_val.to(dev).to(torch.bfloat16) if residual_val is not None else None)
        m.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        m._w_hat_ref = None  # populated lazily only by verification code
        return m

    def dequantized(self) -> torch.Tensor:
        """Reconstruct the effective weight (for swap verification / fidelity)."""
        n, k, g = self.out_features, self.in_features, self.group_size
        sz = self.scales_zeros.transpose(0, 1)  # [N, K/g, 2]
        scales, zeros = sz[..., 0].float(), sz[..., 1].float()
        # unpack via the kernel itself so we never re-derive the layout by hand
        eye = torch.eye(k, device=self.packed.device, dtype=torch.bfloat16)
        w_eff = torch.ops.aten._weight_int4pack_mm(eye, self.packed, g, self.scales_zeros).float().T
        if self.res_idx is not None:
            w_eff[:, self.res_idx.long()] += self.res_val.float()
        if self.s_inv is not None:
            w_eff = w_eff * self.s_inv.float().unsqueeze(0)
        return w_eff

    def extra_repr(self) -> str:
        bits = "int4"
        if self.s_inv is not None:
            bits += "+awq"
        if self.res_idx is not None:
            bits += f"+hsa(k={self.res_idx.numel()})"
        return f"in={self.in_features}, out={self.out_features}, g={self.group_size}, {bits}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.embodiq.w4a16(
            x, self.packed, self.scales_zeros, self.s_inv,
            self.res_idx, self.res_val, self.bias,
            self.group_size, self.out_features,
        )


class SimQuantLinear(nn.Module):
    """Fake quantization: dequantized weights held in the original dtype.

    Same numerics as QuantLinear (verified by the G1 gate), full fp speed, no
    memory saving. Used for fast accuracy triage before paying for packing.
    """

    def __init__(self, linear: nn.Linear, group_size: int = 128,
                 input_scale: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        w = linear.weight.detach().to(torch.float32)
        if input_scale is not None:
            ws = w * input_scale.to(torch.float32).unsqueeze(0)
            q, s, z = group_quantize_int4(ws, group_size)
            w_hat = dequantize_int4(q, s, z, group_size) / input_scale.to(torch.float32).unsqueeze(0)
        else:
            q, s, z = group_quantize_int4(w, group_size)
            w_hat = dequantize_int4(q, s, z, group_size)
        self.register_buffer("weight", w_hat.to(linear.weight.dtype))
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Int8Linear(nn.Module):
    """Symmetric per-output-channel INT8, the no-divisibility-constraint fallback
    for SigLIP's 4304 dims and for narrow projections."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        q, s = quantize_int8_per_channel(linear.weight.detach().to(torch.float32))
        self.register_buffer("qweight", q)
        self.register_buffer("scales", s.to(torch.float32))
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def dequantized(self) -> torch.Tensor:
        return self.qweight.float() * self.scales.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized().to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)
