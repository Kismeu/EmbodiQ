"""HSA v1 (`local_residual_v1`): layered sensitivity-aware residual compensation.

A first implementation, written so 1.3 can replace it: it registers under
`method="hsa", implementation="local_residual_v1"`, produces an ordinary
`embodiq/quant-artifact/1` file, and the evaluation entry neither knows nor cares
which implementation produced it. Swapping in another one changes a string.

There is no published method called HSA — searching the literature in October
2026 finds sensitivity-aware PTQ, residual/low-rank compensation (ARHQ, MARR,
QuaSAR) and sensitivity-driven rank allocation (HCInfer), but no HSA. So this
module is a *definition*, not a reimplementation, and it is written to be
falsifiable: `tests/test_hsa.py` measures whether each step actually reduces
layer output error rather than asserting that it does.

## The definition

For one `nn.Linear` with weight `W ∈ R^{N×K}`, an AWQ input scale `s ∈ R^K_{>0}`,
group size `g`, and a calibration activation statistic `a ∈ R^K_{≥0}`
(mean |x_j| over calibration tokens):

1. **Scaled weight** `V = W ⊙ s` (broadcast over rows). This is the object the
   main branch quantizes, so everything downstream is defined on `V`, not `W`.

2. **Baseline error** `E₀ = V − dequant(quant_g(V))`.

3. **Per-input-channel sensitivity.** The layer's output error contributed by
   channel `j` is `Σ_t x_{t,j} · E₀[:, j]`, whose magnitude scales as
   `a_j · ‖E₀[:, j]‖₂`. So

       score_j = ‖E₀[:, j]‖₂ · a_j

   This is the "结合校准激活和权重量化误差" of the task statement: neither term
   alone is the right quantity. A channel with large quantization error that the
   data never excites costs nothing; a heavily excited channel that quantizes
   cleanly costs nothing either. The product is what the budget should buy down.

4. **Budget.** `k = align_down(ρ·K, align_k)`, clamped to
   `[align_k, ⌊max_per_layer_rho·K⌋]`. Select the top-`k` channels by score.

5. **Requantize with the selected channels excluded from the group statistics,
   and hand those channels entirely to the residual.** Two parts, both load-bearing:

   a. Groups run along `K`, so a selected outlier still stretches its group's
      min/max and degrades the other `g−1` channels in that group — even though
      its own error is about to be cancelled. Recomputing each group's min/max
      over its *unselected* columns tightens the scale for everything that has to
      live with it. (Measured: `test_step5_improves_the_channels_that_keep_no_residual`.)

   b. The selected columns' int4 codes are then forced to whichever code
      dequantizes closest to **zero**, so the main branch contributes nothing
      there and the residual is the sole carrier of those channels. This is what
      "仅对关键区域保留高精度残差" says literally, and it is also the only
      numerically sound choice. Letting the main branch keep a clamped guess
      instead makes the residual cancel a large wrong value: with the
      selected channels spanning ~4 orders of magnitude across rows, `Q₁` and `R`
      become two large numbers that nearly cancel, and bf16's 0.4% precision —
      which is relative to `R`, not to the small target — destroys the rows where
      `V` is small. Measured on a real layer before this was fixed: fp32 residual
      gave 2e-6 relative error, bf16 gave 1.0e-1. Catastrophic cancellation, not
      a tolerance to be loosened.

       Q₁ = dequant(quant_g(V ; stats over unselected, codes≈0 on idx))
       E₁ = V − Q₁

6. **Residual** `R = E₁[:, idx]`, stored bf16. Because `Q₁[:, idx] ≈ 0`,
   `R ≈ V[:, idx]` and bf16 spends its mantissa on the value itself: relative
   accuracy ~0.4% per element on exactly the channels that contribute most to
   output error. The fused kernel computes
   `(x·s⁻¹) @ Q₁ᵀ + (x·s⁻¹)[:, idx] @ Rᵀ`, so those channels reconstruct `V` —
   and therefore `W` — to bf16 relative precision.

Cost per layer: `k·N·2` bytes of residual plus `k·4` bytes of gather index, both
counted against the method's budget by `quant/artifact.py`. Fixing the base bit
width does not fix the total budget, which is why those bytes are reported.

## Ablation arms

`HSASpec.criterion` selects the scoring rule, so the arms 1.3 needs to beat are
already addressable without touching this module:

| criterion | score_j | what it isolates |
|---|---|---|
| `random` | uniform noise | does *any* 1% high-precision slice help? |
| `magnitude` | ‖W[:, j]‖₂ (**raw** W, not `V`) | is weight magnitude enough? |
| `dw` | ‖E₀[:, j]‖₂ | is quantization error enough, without the data? |
| `dw_act` | ‖E₀[:, j]‖₂ · a_j | the product (step 3) |
| `hsa` | `dw_act` + step 5 | the full definition |

`hsa` and `dw_act` score identically; they differ in whether step 5 runs, which
is why `requantize_after_extraction` is a separate spec field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from embodiq.quant.pack import MAX_INT4, dequantize_int4, group_quantize_int4
from embodiq.quant.spec import HSASpec

IMPLEMENTATION = "local_residual_v1"
IMPLEMENTATION_VERSION = "1.0"
CRITERIA = ("random", "magnitude", "dw", "dw_act", "hsa")


@dataclass
class LayerPlan:
    """What one layer contributes to an HSA artifact."""

    residual_idx: torch.Tensor          # int32 [k], sorted, unique
    residual_val: torch.Tensor          # float32 [N, k]; stored bf16 downstream
    diagnostics: dict = field(default_factory=dict)


def channel_scores(
    scaled_weight: torch.Tensor,
    baseline_error: torch.Tensor,
    activation: Optional[torch.Tensor],
    criterion: str,
    *,
    raw_weight: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-input-channel sensitivity under one of the ablation arms.

    `raw_weight` matters for the `magnitude` arm and only for it. Scoring
    `scaled_weight = W ⊙ s` would make that arm secretly activation-aware,
    because the AWQ scale `s` is itself derived from the activations -- the arm
    would then select the same channels as `dw_act` and the ablation would
    measure nothing. (It did exactly that until a test caught the two arms
    producing bit-identical output MSE.)
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    k = scaled_weight.shape[1]
    if criterion == "random":
        # Seeded so an ablation arm is reproducible; an unseeded baseline would
        # make its own comparison unrepeatable.
        return torch.rand(k, generator=generator, device=scaled_weight.device)
    if criterion == "magnitude":
        basis = raw_weight if raw_weight is not None else scaled_weight
        return basis.float().pow(2).sum(dim=0).sqrt()
    error_norm = baseline_error.float().pow(2).sum(dim=0).sqrt()
    if criterion == "dw":
        return error_norm
    if activation is None:
        raise ValueError(f"criterion {criterion!r} needs a calibration activation "
                         f"statistic; none was supplied")
    if activation.shape != (k,):
        raise ValueError(f"activation must be shape ({k},), got {tuple(activation.shape)}")
    if not torch.isfinite(activation).all() or bool((activation < 0).any()):
        raise ValueError("activation statistic must be finite and non-negative")
    return error_norm * activation.float().to(error_norm.device)


def select_channels(scores: torch.Tensor, spec: HSASpec) -> torch.Tensor:
    """Top-k by score, with the budget aligned and capped.

    Alignment keeps the gather GEMM on a friendly width; the cap is the promise
    that "bounded extra VRAM" means something per layer and not just on average.
    """
    k_total = scores.numel()
    align = max(1, spec.align_k)
    ceiling = int(spec.max_per_layer_rho * k_total)
    wanted = int(spec.rho * k_total)
    k = (wanted // align) * align
    k = max(align, min(k, (ceiling // align) * align, k_total))
    if k < 1:
        raise ValueError(f"budget resolved to {k} channels for in_features={k_total}")
    # Ties broken by index so the selection is deterministic.
    order = torch.argsort(scores, descending=True, stable=True)
    return order[:k].sort().values.to(torch.int32)


def group_quantize_excluding(
    weight: torch.Tensor, group_size: int, exclude: torch.Tensor,
    *, zero_excluded: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """int4 group quantization that ignores `exclude` columns in the statistics
    and, by default, drives their codes to dequantize near zero.

    Mirrors `pack.group_quantize_int4` except for those two things, so the two
    differ in controlled ways and nothing else.

    `zero_excluded=True` is what keeps the residual from having to cancel a large
    wrong value -- see step 5b in the module docstring. `False` reproduces the
    naive variant, kept so the test suite can show the difference rather than
    assert it.

    A group all of whose columns are excluded has no statistics to compute; it
    falls back to the unmasked min/max for that group, which costs nothing
    because every column in it carries a residual anyway.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"in_features {k} not divisible by group_size {group_size}")

    w = weight.to(torch.float32).reshape(n, k // group_size, group_size)
    keep = torch.ones(k, dtype=torch.bool, device=weight.device)
    keep[exclude.long()] = False
    keep_g = keep.reshape(1, k // group_size, group_size)
    empty = ~keep_g.any(dim=-1, keepdim=True)          # [1, K/g, 1]

    masked_max = w.masked_fill(~keep_g, float("-inf")).amax(dim=-1, keepdim=True)
    masked_min = w.masked_fill(~keep_g, float("inf")).amin(dim=-1, keepdim=True)
    w_max = torch.where(empty, w.amax(dim=-1, keepdim=True), masked_max)
    w_min = torch.where(empty, w.amin(dim=-1, keepdim=True), masked_min)

    scales = ((w_max - w_min) / MAX_INT4).clamp(min=1e-6)
    zeros = w_min + scales * 8.0
    q = ((w - w_min) / scales).round().clamp_(0, MAX_INT4)
    if zero_excluded and exclude.numel():
        # The code whose dequantization lands nearest 0: dequant(q) = q*s + w_min,
        # so q* = round(-w_min / s).
        near_zero = ((-w_min) / scales).round().clamp_(0, MAX_INT4)
        q = torch.where(keep_g, q, near_zero.expand_as(q))
    q = q.to(torch.uint8)
    return (q.reshape(n, k), scales.reshape(n, k // group_size),
            zeros.reshape(n, k // group_size))


def plan_layer(
    weight: torch.Tensor,
    *,
    input_scale: Optional[torch.Tensor] = None,
    activation: Optional[torch.Tensor] = None,
    spec: Optional[HSASpec] = None,
    group_size: int = 128,
    generator: Optional[torch.Generator] = None,
) -> LayerPlan:
    """Run the six steps for one layer.

    `input_scale` is the AWQ scale this layer will be deployed with. HSA's main
    branch must be the same object as the AWQ row it is compared against, so the
    residual is computed against a quantization of `W ⊙ s` -- passing the scale
    here is not optional for a comparable HSA row, only for a standalone one.
    """
    spec = spec or HSASpec()
    if spec.granularity != "channel":
        raise NotImplementedError(
            f"{IMPLEMENTATION} implements channel granularity; "
            f"spec asks for {spec.granularity!r}")
    if spec.residual_impl != "gather":
        raise NotImplementedError(f"residual_impl must be 'gather', "
                                  f"got {spec.residual_impl!r}")

    w = weight.detach().float()
    n, k = w.shape
    scaled = w * input_scale.detach().float().to(w.device)[None] if input_scale is not None else w

    # Step 2: the error the main branch would have without any residual.
    q0, s0, z0 = group_quantize_int4(scaled, group_size)
    baseline = dequantize_int4(q0, s0.bfloat16(), z0.bfloat16(), group_size)
    error0 = scaled - baseline

    # Steps 3-4.
    scores = channel_scores(scaled, error0, activation, spec.criterion,
                            raw_weight=w, generator=generator)
    idx = select_channels(scores, spec)

    # Step 5: only the full `hsa` arm requantizes; the others keep the baseline
    # main branch so the ablation isolates the scoring rule.
    if spec.criterion == "hsa" and spec.requantize_after_extraction:
        q1, s1, z1 = group_quantize_excluding(scaled, group_size, idx)
        final = dequantize_int4(q1, s1.bfloat16(), z1.bfloat16(), group_size)
    else:
        final = baseline
    error1 = scaled - final

    # Step 6.
    residual = error1[:, idx.long()].contiguous()

    rel = lambda e: float(e.norm() / scaled.norm().clamp(min=1e-12))  # noqa: E731
    kept = torch.ones(k, dtype=torch.bool, device=w.device)
    kept[idx.long()] = False
    return LayerPlan(
        residual_idx=idx.cpu(),
        residual_val=residual.cpu(),
        diagnostics={
            "in_features": k, "out_features": n,
            "k_residual": int(idx.numel()),
            "share": float(idx.numel() / k),
            "criterion": spec.criterion,
            "requantized": bool(spec.criterion == "hsa"
                                and spec.requantize_after_extraction),
            # Reported so the claim "the main branch improved" is checkable
            # rather than assumed: this is the error on the channels that do NOT
            # get a residual, before and after step 5.
            "rel_err_baseline_all": rel(error0),
            "rel_err_final_all": rel(error1),
            "rel_err_baseline_unselected": rel(error0[:, kept]),
            "rel_err_final_unselected": rel(error1[:, kept]),
            "residual_bytes": int(residual.numel() * 2),
            "index_bytes": int(idx.numel() * 4),
        },
    )


def effective_weight(
    weight: torch.Tensor, plan: LayerPlan, *,
    input_scale: Optional[torch.Tensor] = None, group_size: int = 128,
    spec: Optional[HSASpec] = None,
) -> torch.Tensor:
    """What the deployed layer actually computes with, for verification.

    Reproduces `QuantLinear.dequantized()`: add the residual in the scaled domain,
    then divide the AWQ scale back out. Used by the tests to compare HSA against
    RTN and AWQ on the same footing instead of trusting the arithmetic.
    """
    spec = spec or HSASpec()
    w = weight.detach().float()
    s = input_scale.detach().float().to(w.device) if input_scale is not None else None
    scaled = w * s[None] if s is not None else w
    if spec.criterion == "hsa" and spec.requantize_after_extraction:
        q, sc, z = group_quantize_excluding(scaled, group_size,
                                            plan.residual_idx.to(w.device))
    else:
        q, sc, z = group_quantize_int4(scaled, group_size)
    out = dequantize_int4(q, sc.bfloat16(), z.bfloat16(), group_size)
    idx = plan.residual_idx.long().to(w.device)
    # bf16 round-trip, because that is what the buffer stores.
    out[:, idx] += plan.residual_val.to(w.device).bfloat16().float()
    return out / s[None] if s is not None else out
