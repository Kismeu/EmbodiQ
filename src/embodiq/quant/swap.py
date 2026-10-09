"""Replace nn.Linear leaves with quantized modules, and PROVE the swap worked.

The failure this defends against: FastWAM calls ``load_state_dict(strict=False)``
in six places. Swapping a Linear renames ``weight`` to ``packed``/``scales_zeros``,
so a checkpoint load AFTER the swap silently drops every quantized tensor and
leaves random init. The model still runs. It looks exactly like "INT4 destroyed
FastWAM".

Two defenses, both cheap:
  * an ordering rule -- load the checkpoint FIRST, quantize SECOND, asserted here;
  * a post-swap check that every replaced module reconstructs its original weight
    to within the expected quantization error band, and that nothing still looks
    like fresh init.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch
import torch.nn as nn

from embodiq.quant.qlinear import Int8Linear, QuantLinear, SimQuantLinear
from embodiq.quant.spec import LayerRecord, QuantReport, QuantSpec

# Expected relative Frobenius error of group-wise INT4 on real weight matrices.
# Uniform quantization of a roughly-Gaussian group of 128 at 4 bits lands near
# 0.10; anything far above means the swap or the layout is broken, anything at
# ~0 means nothing was actually quantized.
INT4_REL_ERR_BAND = (0.005, 0.35)
INT8_REL_ERR_BAND = (0.0001, 0.05)
# An HSA module restores a slice of its input channels exactly, so its
# reconstruction error sits BELOW the plain-RTN band by design. Keeping the
# upper bound catches a broken layout; the lower bound is relaxed because
# "almost no error" is the intended outcome here, not a sign of a no-op swap --
# the residual-bearing layers are separately asserted to be non-empty by
# quant.artifact._check_method_shape.
INT4_RESIDUAL_REL_ERR_BAND = (0.0, 0.35)


@dataclass
class SwapResult:
    replaced: dict[str, str] = field(default_factory=dict)   # name -> module class
    skipped: list[str] = field(default_factory=list)
    fingerprint_before: str = ""
    fingerprint_after: str = ""
    checks: list[dict] = field(default_factory=list)

    @property
    def n_replaced(self) -> int:
        return len(self.replaced)


def param_fingerprint(roots: Mapping[str, nn.Module]) -> str:
    """sha256 over (name, shape, mean|p|) for every parameter.

    Goes into the result JSON. An accidental re-init is then visible as a diff
    between two runs rather than as an unexplained accuracy cliff.
    """
    h = hashlib.sha256()
    for root_name in sorted(roots):
        for name, p in sorted(roots[root_name].named_parameters()):
            h.update(f"{root_name}.{name}".encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(f"{float(p.detach().float().abs().mean()):.6e}".encode())
    return h.hexdigest()


def _get_parent(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj, parts[-1]


def _set_child(parent: nn.Module, attr: str, mod: nn.Module) -> None:
    if attr.isdigit():
        parent[int(attr)] = mod
    else:
        setattr(parent, attr, mod)


def apply_report(
    roots: Mapping[str, nn.Module],
    report: QuantReport,
    spec: QuantSpec,
    *,
    input_scales: Optional[Mapping[str, torch.Tensor]] = None,
    residuals: Optional[Mapping[str, tuple[torch.Tensor, torch.Tensor]]] = None,
    clips: Optional[Mapping[str, torch.Tensor]] = None,
    verify: bool = True,
) -> SwapResult:
    """Swap in the modules the plan decided on. Checkpoint must ALREADY be loaded."""
    res = SwapResult(fingerprint_before=param_fingerprint(roots))
    input_scales = input_scales or {}
    residuals = residuals or {}
    clips = clips or {}
    by_name = {r.name: r for r in report.layers}

    for name, rec in by_name.items():
        if rec.decision == "bf16":
            res.skipped.append(name)
            continue
        root_name, _, local = name.partition(".") if len(roots) > 1 else ("", "", name)
        root = roots[root_name] if root_name in roots else next(iter(roots.values()))
        local = local or name
        parent, attr = _get_parent(root, local)
        linear = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"{name} is {type(linear).__name__}, not nn.Linear -- "
                               f"was the model already quantized, or the report stale?")
        w_before = linear.weight.detach().float().clone()

        if rec.decision == "int8":
            new = Int8Linear(linear)
            band = INT8_REL_ERR_BAND
        elif spec.simulated:
            new = SimQuantLinear(linear, rec.group_size, input_scales.get(name))
            band = INT4_REL_ERR_BAND
        else:
            ridx, rval = residuals.get(name, (None, None))
            new = QuantLinear.from_linear(
                linear, rec.group_size,
                input_scale=input_scales.get(name),
                clip_max=clips.get(name),
                residual_idx=ridx, residual_val=rval,
            )
            band = INT4_RESIDUAL_REL_ERR_BAND if ridx is not None else INT4_REL_ERR_BAND
            if ridx is not None:
                rec.residual_units = int(ridx.numel())
                rec.residual_bytes = int(rval.numel() * 2)
                rec.index_bytes = int(ridx.numel() * 4)

        _set_child(parent, attr, new)
        res.replaced[name] = type(new).__name__

        if verify:
            w_after = (new.weight.detach().float() if isinstance(new, SimQuantLinear)
                       else new.dequantized())
            rel = float((w_after - w_before).norm() / w_before.norm().clamp(min=1e-12))
            ok = band[0] <= rel <= band[1]
            rec.calib_out_rel_err = rel
            res.checks.append({"name": name, "rel_fro": rel, "band": band, "ok": ok,
                               "decision": rec.decision})
            if not ok:
                raise RuntimeError(
                    f"swap verification failed for {name}: reconstruction error "
                    f"{rel:.4f} outside {band}. Below the band means the weight was "
                    f"not actually quantized; above means the layout or the scales "
                    f"are wrong."
                )

    res.fingerprint_after = param_fingerprint(roots)
    return res


def assert_no_linear_left_in_scope(roots: Mapping[str, nn.Module], report: QuantReport) -> None:
    """Every layer the report marked for quantization must actually be gone.

    A subset match is not good enough -- that is precisely how a partially
    applied swap hides.
    """
    want = {r.name for r in report.layers if r.decision != "bf16"}
    still: list[str] = []
    for root_name, root in roots.items():
        for name, mod in root.named_modules():
            full = f"{root_name}.{name}" if len(roots) > 1 else name
            if full in want and isinstance(mod, nn.Linear):
                still.append(full)
    if still:
        raise RuntimeError(f"{len(still)} layers were planned but not replaced: {still[:10]}")
