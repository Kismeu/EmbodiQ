"""AWQ activation-scale search primitive, explicit per-Linear adaptation.

This is preparation, not a completed official block-level AWQ implementation.
The 20-point activation-exponent grid follows mit-han-lab/llm-awq auto_scale.py.
Objective here is per-Linear output reconstruction, rather than the upstream
attention/MLP block objective. Clipping and runtime integration remain gated.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from embodiq.quant.pack import group_quantize_int4, dequantize_int4

IMPLEMENTATION = "awq_linear_scale_search_v1"


@torch.no_grad()
def search_input_scale(weight, activation, *, group_size=128, n_grid=20):
    if n_grid < 1: raise ValueError("n_grid must be positive")
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError("Unsupported weight shape/group size")
    x = activation["samples"].to(device=weight.device, dtype=torch.float32)
    mean_abs = activation["mean_abs"].to(device=weight.device, dtype=torch.float32)
    if x.ndim != 2 or not len(x) or x.shape[1] != weight.shape[1] or mean_abs.shape != (weight.shape[1],):
        raise ValueError("Activation dimensions do not match weight")
    if not torch.isfinite(x).all() or not torch.isfinite(mean_abs).all() or (mean_abs < 0).any():
        raise ValueError("Invalid calibration activations")
    if not torch.isfinite(weight).all(): raise ValueError("Non-finite weight")
    w = weight.float()
    reference = F.linear(x, w)
    history = []
    best = None
    for i in range(n_grid):
        ratio = i / n_grid
        scale = mean_abs.pow(ratio).clamp(min=1e-4)
        scale = scale / (scale.max() * scale.min()).sqrt()
        q, s, z = group_quantize_int4(w * scale[None], group_size)
        # Include the runtime packed format's BF16 scales/zero offsets.
        reconstructed = dequantize_int4(q, s.bfloat16(), z.bfloat16(), group_size) / scale[None]
        loss = float((F.linear(x, reconstructed) - reference).square().mean())
        if not torch.isfinite(torch.tensor(loss)): raise ValueError("Non-finite reconstruction objective")
        history.append(loss)
        if best is None or loss < best["loss"]:
            best = dict(scale=scale.cpu().clone(), ratio=ratio, loss=loss)
    best.update(implementation=IMPLEMENTATION, rtn_reference_loss=history[0], losses=history,
                group_size=group_size, objective="per_linear_output_mse", clipping=False)
    return best
