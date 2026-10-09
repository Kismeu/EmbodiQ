"""HSA v1, measured rather than asserted.

The point of these tests is not that the code runs. It is that each step of the
definition in `quant/hsa.py` actually buys down layer output error, and that the
scoring rule beats the ablation arms it claims to beat. A quantization method
whose steps are individually unverified is a method-shaped pile of tensors, so
every claim the module docstring makes has a test here that could fail.

Error is always measured on the LAYER OUTPUT against real-ish activations, not on
the weight, because output error is what propagates.
"""

from __future__ import annotations

import torch

import pytest

from embodiq.quant.hsa import (
    IMPLEMENTATION,
    channel_scores,
    effective_weight,
    group_quantize_excluding,
    plan_layer,
    select_channels,
)
from embodiq.quant.pack import dequantize_int4, group_quantize_int4
from embodiq.quant.spec import HSASpec

GROUP = 128
N, K = 256, 512


def make_layer(seed=0, n=N, k=K, n_outlier=12, act_gain=18.0, weight_gain=6.0):
    """A layer in the regime AWQ/HSA target, built so the ablation arms can differ.

    Two DISJOINT outlier populations, which is the whole reason the activation
    term is supposed to matter:

    * `loud`: heavily excited channels of ordinary weight magnitude. These are the
      ones that actually cost output error, and only an activation-aware score
      finds them.
    * `fat`: large-weight channels the data barely excites. A weight-magnitude
      score spends its budget here and buys almost nothing.

    A fixture where the same channels are both loud and fat makes `magnitude` and
    `dw_act` select identically and the comparison between them vacuous -- which
    is exactly what the first version of this file did.
    """
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(n, k, generator=g) * 0.02
    act = torch.distributions.LogNormal(0.0, 0.5).sample((k,))
    perm = torch.randperm(k, generator=g)
    loud, fat = perm[:n_outlier], perm[n_outlier:2 * n_outlier]
    act[loud] *= act_gain
    act[fat] *= 0.05                       # quiet
    weight[:, fat] *= weight_gain          # but fat
    x = torch.randn(512, k, generator=g) * act[None]
    return weight, act, x


def awq_scale(weight, act, x, group_size=GROUP):
    """The project's own AWQ scale search, not a hand-picked exponent.

    A fixed alpha is a strawman: on a layer whose channels differ by orders of
    magnitude it can be *worse* than RTN, which would make "HSA beats AWQ"
    meaningless. `search_input_scale` sweeps the exponent and keeps the one that
    minimizes this layer's output MSE -- the same objective HSA is judged on.
    """
    from embodiq.quant.awq_search import search_input_scale

    best = search_input_scale(
        weight.float(), {"samples": x.float(), "mean_abs": act.float()},
        group_size=group_size)
    return best["scale"], best


def output_mse(weight_eff, weight, x):
    ref = x @ weight.T
    got = x @ weight_eff.T
    return float((got - ref).pow(2).mean())


def rtn_weight(weight, group_size=GROUP):
    q, s, z = group_quantize_int4(weight.float(), group_size)
    return dequantize_int4(q, s.bfloat16(), z.bfloat16(), group_size)


def awq_weight(weight, scale, group_size=GROUP):
    scaled = weight.float() * scale[None]
    q, s, z = group_quantize_int4(scaled, group_size)
    return dequantize_int4(q, s.bfloat16(), z.bfloat16(), group_size) / scale[None]


# -- the method's central claim --------------------------------------------

def test_hsa_beats_awq_beats_rtn_on_output_error():
    """The ordering the method exists to produce. If this fails, HSA v1 is not a
    method, and no amount of plumbing fixes that."""
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    spec = HSASpec(rho=0.02, criterion="hsa")
    plan = plan_layer(weight, input_scale=scale, activation=act, spec=spec,
                      group_size=GROUP)

    mse_rtn = output_mse(rtn_weight(weight), weight, x)
    mse_awq = output_mse(awq_weight(weight, scale), weight, x)
    mse_hsa = output_mse(
        effective_weight(weight, plan, input_scale=scale, spec=spec, group_size=GROUP),
        weight, x)

    assert mse_awq < mse_rtn, f"AWQ {mse_awq:.4g} should beat RTN {mse_rtn:.4g}"
    assert mse_hsa < mse_awq, f"HSA {mse_hsa:.4g} should beat AWQ {mse_awq:.4g}"
    # Record the magnitude so a regression that merely squeaks past is visible.
    print(f"\n  RTN {mse_rtn:.4g} -> AWQ {mse_awq:.4g} -> HSA {mse_hsa:.4g}  "
          f"(HSA/AWQ = {mse_hsa / mse_awq:.3f}, k={plan.diagnostics['k_residual']}"
          f"/{K} = {plan.diagnostics['share']:.1%})")


def test_the_activation_term_earns_its_place():
    """`dw_act` must beat `dw`: weighting the quantization error by how much the
    data excites each channel is the whole claim of step 3."""
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    scores = {}
    for criterion in ("random", "magnitude", "dw", "dw_act"):
        spec = HSASpec(rho=0.02, criterion=criterion, requantize_after_extraction=False)
        plan = plan_layer(weight, input_scale=scale, activation=act, spec=spec,
                          group_size=GROUP,
                          generator=torch.Generator().manual_seed(7))
        scores[criterion] = output_mse(
            effective_weight(weight, plan, input_scale=scale, spec=spec,
                             group_size=GROUP), weight, x)
    print("\n  " + "  ".join(f"{k}={v:.4g}" for k, v in scores.items()))
    assert scores["dw_act"] < scores["dw"], "the activation term must help"
    assert scores["dw_act"] < scores["magnitude"], "must beat weight magnitude"
    assert scores["dw_act"] < scores["random"], "must beat a random slice"


def test_step5_improves_the_channels_that_keep_no_residual():
    """Step 5's justification: excluding the selected outliers from the group
    statistics tightens the scale for the g-1 channels that must live with it."""
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    plan = plan_layer(weight, input_scale=scale, activation=act,
                      spec=HSASpec(rho=0.02, criterion="hsa"), group_size=GROUP)
    d = plan.diagnostics
    assert d["requantized"]
    print(f"\n  unselected rel err {d['rel_err_baseline_unselected']:.5f} -> "
          f"{d['rel_err_final_unselected']:.5f}")
    assert d["rel_err_final_unselected"] < d["rel_err_baseline_unselected"], (
        "requantizing without the outliers must reduce the error on the rest")


# -- exactness of the compensated channels ---------------------------------

def test_selected_channels_reconstruct_exactly():
    """The residual is defined to cancel the error on its channels; anything less
    and the accounting (bytes spent vs error removed) is wrong."""
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    spec = HSASpec(rho=0.02, criterion="hsa")
    plan = plan_layer(weight, input_scale=scale, activation=act, spec=spec,
                      group_size=GROUP)
    eff = effective_weight(weight, plan, input_scale=scale, spec=spec, group_size=GROUP)
    idx = plan.residual_idx.long()
    target = weight[:, idx]
    delta = eff[:, idx] - target
    # Norm-relative, per channel. Per-ELEMENT relative error is the wrong metric
    # here: a selected channel spans ~4 orders of magnitude across rows, and
    # asking for 0.4% on its 1e-5 entries demands absolute precision far finer
    # than the channel contributes. What the method promises -- and what the
    # output error depends on -- is that the channel as a vector is recovered.
    per_channel = delta.norm(dim=0) / target.norm(dim=0).clamp(min=1e-12)
    assert float(per_channel.max()) < 1e-2, (
        f"worst channel norm-relative error {float(per_channel.max()):.4g}")
    # The operational claim, instead of a per-element tolerance: the output error
    # these channels contribute must collapse relative to AWQ. This is what the
    # residual bytes are bought with, and it is measurable directly.
    awq_eff = awq_weight(weight, scale)
    contribution = lambda eff: float(  # noqa: E731
        (x[:, idx] @ (eff[:, idx] - target).T).pow(2).mean())
    awq_part, hsa_part = contribution(awq_eff), contribution(eff)
    print(f"\n  selected-channel output error: AWQ {awq_part:.4g} -> "
          f"HSA {hsa_part:.4g} ({hsa_part / awq_part:.1e}x)")
    assert hsa_part < awq_part / 10, (
        f"compensated channels still contribute {hsa_part:.4g} vs AWQ {awq_part:.4g}")
    # The compensated channels must be far better than the uncompensated ones,
    # or the bytes bought nothing.
    kept = torch.ones(K, dtype=torch.bool)
    kept[idx] = False
    uncompensated = ((eff[:, kept] - weight[:, kept]).norm(dim=0)
                     / weight[:, kept].norm(dim=0).clamp(min=1e-12))
    print(f"\n  compensated worst {float(per_channel.max()):.5f} vs "
          f"uncompensated median {float(uncompensated.median()):.5f}")
    assert float(per_channel.max()) < float(uncompensated.median())



# -- budget ----------------------------------------------------------------

def test_budget_is_aligned_capped_and_nonzero():
    scores = torch.rand(1024)
    spec = HSASpec(rho=0.01, max_per_layer_rho=0.08, align_k=8)
    idx = select_channels(scores, spec)
    assert idx.numel() == 8 and idx.numel() % spec.align_k == 0
    assert idx.dtype == torch.int32
    assert idx.unique().numel() == idx.numel()
    assert bool((idx[1:] > idx[:-1]).all()), "indices must be sorted"


def test_budget_respects_the_per_layer_ceiling():
    """rho above the ceiling must clamp, or "bounded extra VRAM" is not bounded."""
    scores = torch.rand(1024)
    idx = select_channels(scores, HSASpec(rho=0.5, max_per_layer_rho=0.08, align_k=8))
    assert idx.numel() <= int(0.08 * 1024)


def test_a_tiny_layer_still_gets_at_least_one_aligned_block():
    idx = select_channels(torch.rand(128), HSASpec(rho=0.001, align_k=8))
    assert idx.numel() == 8


def test_plan_reports_its_own_cost():
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    plan = plan_layer(weight, input_scale=scale, activation=act,
                      spec=HSASpec(rho=0.02, criterion="hsa"), group_size=GROUP)
    d = plan.diagnostics
    assert d["residual_bytes"] == plan.residual_val.numel() * 2
    assert d["index_bytes"] == plan.residual_idx.numel() * 4
    assert d["share"] <= HSASpec().max_per_layer_rho + 1e-9


# -- group quantization with exclusions ------------------------------------

def test_excluding_columns_changes_only_the_statistics():
    """With nothing excluded it must be bit-identical to pack.group_quantize_int4,
    so the two differ in exactly one controlled way."""
    weight = torch.randn(64, 256)
    a = group_quantize_int4(weight, GROUP)
    b = group_quantize_excluding(weight, GROUP, torch.zeros(0, dtype=torch.int32))
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_a_fully_excluded_group_falls_back_instead_of_producing_inf():
    """Every column of one group selected: there are no statistics to compute."""
    weight = torch.randn(32, 256)
    exclude = torch.arange(GROUP, dtype=torch.int32)        # all of group 0
    q, s, z = group_quantize_excluding(weight, GROUP, exclude)
    assert torch.isfinite(s).all() and torch.isfinite(z).all()
    assert bool((s > 0).all())
    recon = dequantize_int4(q, s.bfloat16(), z.bfloat16(), GROUP)
    assert torch.isfinite(recon).all()


# -- determinism and guards ------------------------------------------------

def test_planning_is_deterministic():
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    spec = HSASpec(rho=0.02, criterion="hsa")
    a = plan_layer(weight, input_scale=scale, activation=act, spec=spec, group_size=GROUP)
    b = plan_layer(weight, input_scale=scale, activation=act, spec=spec, group_size=GROUP)
    assert torch.equal(a.residual_idx, b.residual_idx)
    assert torch.equal(a.residual_val, b.residual_val)


def test_random_arm_is_seeded():
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    spec = HSASpec(rho=0.02, criterion="random")
    plans = [plan_layer(weight, input_scale=scale, activation=act, spec=spec,
                        group_size=GROUP,
                        generator=torch.Generator().manual_seed(3)) for _ in range(2)]
    assert torch.equal(plans[0].residual_idx, plans[1].residual_idx)


def test_activation_criteria_require_an_activation():
    weight, _, _ = make_layer()
    with pytest.raises(ValueError, match="needs a calibration activation"):
        plan_layer(weight, spec=HSASpec(criterion="dw_act"), group_size=GROUP)


def test_a_negative_or_nonfinite_activation_is_rejected():
    weight, act, x = make_layer()
    bad = act.clone(); bad[0] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        channel_scores(weight, torch.zeros_like(weight), bad, "dw_act")


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="criterion must be one of"):
        channel_scores(torch.zeros(4, 8), torch.zeros(4, 8), None, "clairvoyance")


def test_unsupported_granularity_is_refused_not_ignored():
    weight, act, x = make_layer()
    scale, _ = awq_scale(weight, act, x)
    with pytest.raises(NotImplementedError, match="channel granularity"):
        plan_layer(weight, input_scale=scale, activation=act,
                   spec=HSASpec(granularity="block"), group_size=GROUP)


def test_implementation_identity_is_declared():
    """Results must name the implementation so 1.3's version is a different row."""
    assert IMPLEMENTATION == "local_residual_v1"
