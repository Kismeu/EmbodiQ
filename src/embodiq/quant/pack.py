"""INT4 group quantization in torch's native tinygemm layout.

We use ``torch.ops.aten._weight_int4pack_mm`` rather than a vendored CUDA kernel:
it ships inside torch (so it exists in BOTH the torch-2.2 and torch-2.7 prefixes
with no extra build), and it survives ``torch.compile(fullgraph=True,
mode="reduce-overhead")``, which FastWAM turns on by default.

tinygemm's dequantization convention, which we must match exactly:

    scale = (w_max - w_min) / 15
    zero  = w_min + scale * 8
    q     = round((w - w_min) / scale)          in [0, 15]
    w_hat = (q - 8) * scale + zero

Note ``zero`` is a *float offset*, not an integer zero-point, and ``scales_and_zeros``
is laid out [K // group_size, N, 2] -- transposed relative to the weight. Both the
scale and the zero are stored in bf16, so the real cost at g=128 is
4 + 16/128 + 16/128 = **4.25 bit/param**, not the 4.156 you get if you assume the
zero is packed as int4.
"""

from __future__ import annotations

from typing import Optional

import torch

INNER_K_TILES = 8
MAX_INT4 = 15


def group_quantize_int4(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """nn.Linear weight [N, K] -> (q [N, K] uint8 in 0..15, scales, zeros).

    Groups run along K (the input dim), matching tinygemm.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"in_features {k} not divisible by group_size {group_size}")

    w = weight.to(torch.float32).reshape(n, k // group_size, group_size)
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    scales = ((w_max - w_min) / MAX_INT4).clamp(min=1e-6)
    zeros = w_min + scales * 8.0
    q = ((w - w_min) / scales).round().clamp_(0, MAX_INT4).to(torch.uint8)
    return q.reshape(n, k), scales.reshape(n, k // group_size), zeros.reshape(n, k // group_size)


def dequantize_int4(
    q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    n, k = q.shape
    qq = q.to(torch.float32).reshape(n, k // group_size, group_size)
    s = scales.reshape(n, k // group_size, 1).to(torch.float32)
    z = zeros.reshape(n, k // group_size, 1).to(torch.float32)
    return ((qq - 8.0) * s + z).reshape(n, k)


_PACK_INPUT_FORMAT: Optional[str] = None


def _detect_pack_format() -> str:
    """``_convert_weight_to_int4pack`` changed its input format between the two
    torch versions this project must run on, and BOTH accept an argument without
    complaint at the Python level:

      * torch 2.2 (the two Prismatic prefixes): int32 [N, K], values 0..15
      * torch >= 2.5 (the FastWAM prefix):      uint8 [N, K//2], nibble pairs

    The dequant convention and the [K/g, N, 2] scales layout are identical, so
    only the input packing differs. Detect once rather than branching on a
    version string.
    """
    global _PACK_INPUT_FORMAT
    if _PACK_INPUT_FORMAT is not None:
        return _PACK_INPUT_FORMAT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # K must be a multiple of INNER_K_TILES * 16 or the kernel raises for a SHAPE
    # reason, which would otherwise be misread as a dtype signal.
    n, k = 8, INNER_K_TILES * 16 * 2
    try:
        torch.ops.aten._convert_weight_to_int4pack(
            torch.zeros(n, k, dtype=torch.int32, device=dev), INNER_K_TILES
        )
        _PACK_INPUT_FORMAT = "int32_nk"
        return _PACK_INPUT_FORMAT
    except RuntimeError as exc:
        if "dtype" not in str(exc).lower() and "kByte" not in str(exc):
            raise RuntimeError(
                f"tinygemm probe failed for a reason other than dtype: {exc}"
            ) from exc
    torch.ops.aten._convert_weight_to_int4pack(
        torch.zeros(n, k // 2, dtype=torch.uint8, device=dev), INNER_K_TILES
    )
    _PACK_INPUT_FORMAT = "uint8_pairs"
    return _PACK_INPUT_FORMAT


def _pack_codes(q: torch.Tensor) -> torch.Tensor:
    """uint8 codes [N, K] in 0..15 -> whatever this torch's kernel wants."""
    if _detect_pack_format() == "int32_nk":
        return torch.ops.aten._convert_weight_to_int4pack(
            q.to(torch.int32).contiguous(), INNER_K_TILES
        )
    # Nibble order determined empirically against the kernel (see
    # tests/test_pack.py::test_probe_documents_tinygemm_layout): the EVEN K index
    # goes in the HIGH nibble and the odd index in the low nibble. Guessing the
    # other way round transposes adjacent weights and produces a model that runs
    # and is quietly wrong.
    paired = (q[:, 1::2] | (q[:, 0::2] << 4)).contiguous()
    return torch.ops.aten._convert_weight_to_int4pack(paired, INNER_K_TILES)


def to_tinygemm(
    q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack into the layout ``_weight_int4pack_mm`` expects.

    Returns (packed, scales_and_zeros) where scales_and_zeros is
    [K // group_size, N, 2] bf16 -- note the transposed axis order.
    """
    n, k = q.shape
    packed = _pack_codes(q)
    sz = torch.stack(
        [scales.to(torch.bfloat16), zeros.to(torch.bfloat16)], dim=-1
    ).transpose(0, 1).contiguous()  # [K//g, N, 2]
    return packed, sz


def quantize_linear_weight(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot: weight [N, K] -> (packed, scales_and_zeros, dequantized_reference)."""
    q, s, z = group_quantize_int4(weight, group_size)
    packed, sz = to_tinygemm(q, s, z, group_size)
    return packed, sz, dequantize_int4(q, s, z, group_size)


def verify_roundtrip(
    weight: torch.Tensor, group_size: int = 128, atol: float = 2e-2
) -> dict:
    """Prove the packed kernel agrees with an explicit dequantized matmul.

    This is the check that catches a mismatched dequant convention -- the class
    of bug that leaves a model running and silently wrong.
    """
    dev = weight.device
    n, k = weight.shape
    packed, sz, w_hat = quantize_linear_weight(weight, group_size)
    x = torch.randn(4, k, device=dev, dtype=torch.bfloat16)

    y_kernel = torch.ops.aten._weight_int4pack_mm(x, packed, group_size, sz)
    y_ref = torch.nn.functional.linear(x, w_hat.to(torch.bfloat16))

    diff = (y_kernel.float() - y_ref.float()).abs()
    denom = y_ref.float().abs().max().clamp(min=1e-6)
    rel = float((diff.max() / denom).item())
    w_err = (w_hat - weight.float()).abs()
    return {
        "kernel_vs_dequant_rel": rel,
        "kernel_matches": rel < atol,
        "weight_max_abs_err": float(w_err.max()),
        "weight_rel_fro": float(w_err.norm() / weight.float().norm()),
        "bytes_fp": weight.numel() * 2,
        "bytes_q": packed.numel() * packed.element_size() + sz.numel() * 2,
    }


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8. No divisibility constraint at all,
    which is exactly why it is the fallback for SigLIP's 4304 dims."""
    scales = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (weight / scales).round().clamp_(-127, 127).to(torch.int8)
    return q, scales.squeeze(1).to(torch.float32)


def dequantize_int8_per_channel(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scales.unsqueeze(1)
