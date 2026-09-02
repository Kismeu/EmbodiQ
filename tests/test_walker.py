"""Walker validation on synthetic trees that mirror the real module names.

Runs without any checkpoint, so the scope rules, the SigLIP 4304 fallback, the
tied-weight guard and the accounting are all pinned before a model exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from embodiq.quant.spec import QuantSpec, SkipReason
from embodiq.quant.walker import plan_quantization


def _vit(blocks: int, dim: int, mlp: int, heads_mult: int = 3) -> nn.Module:
    m = nn.Module()
    m.blocks = nn.ModuleList()
    for _ in range(blocks):
        b = nn.Module()
        b.attn = nn.Module()
        b.attn.qkv = nn.Linear(dim, dim * heads_mult)
        b.attn.proj = nn.Linear(dim, dim)
        b.mlp = nn.Module()
        b.mlp.fc1 = nn.Linear(dim, mlp)
        b.mlp.fc2 = nn.Linear(mlp, dim)
        m.blocks.append(b)
    return m


def build_vla_adapter() -> dict:
    """Qwen2.5-0.5B + DINOv2-L + SigLIP-SO400M, plus the loose .pt modules."""
    hf = nn.Module()
    lm = nn.Module(); lm.model = nn.Module(); lm.model.layers = nn.ModuleList()
    for _ in range(24):
        l = nn.Module(); l.self_attn = nn.Module(); l.mlp = nn.Module()
        l.self_attn.q_proj = nn.Linear(896, 896)
        l.self_attn.k_proj = nn.Linear(896, 128)   # GQA kv_heads=2
        l.self_attn.v_proj = nn.Linear(896, 128)
        l.self_attn.o_proj = nn.Linear(896, 896)
        l.mlp.gate_proj = nn.Linear(896, 4864)
        l.mlp.up_proj = nn.Linear(896, 4864)
        l.mlp.down_proj = nn.Linear(4864, 896)
        lm.model.layers.append(l)
    lm.model.embed_tokens = nn.Embedding(151936, 896)
    lm.lm_head = nn.Linear(896, 151936, bias=False)
    lm.lm_head.weight = lm.model.embed_tokens.weight   # tied
    hf.language_model = lm
    hf.vision_backbone = nn.Module()
    hf.vision_backbone.featurizer = _vit(24, 1024, 4096)          # DINOv2-L
    hf.vision_backbone.fused_featurizer = _vit(27, 1152, 4304)    # SigLIP-SO400M
    hf.projector = nn.Module()
    hf.projector.fc1 = nn.Linear(2176, 8704)
    hf.projector.fc2 = nn.Linear(8704, 896)
    hf.projector.fc3 = nn.Linear(896, 896)

    head = nn.Module(); head.model = nn.Module()
    head.model.fc1 = nn.Linear(6272, 896)
    head.model.fc2 = nn.Linear(896, 7)
    head.model.mlp_resnet_blocks = nn.ModuleList()
    for _ in range(24):
        b = nn.Module()
        for nm in ("q_proj","k_self","v_self","k_adapter","v_adapter","k_task","v_task","o_proj"):
            setattr(b, nm, nn.Linear(896, 896))
        b.ffn = nn.Sequential(nn.LayerNorm(896), nn.Linear(896, 896), nn.ReLU())
        b.film_gen = nn.Sequential(nn.Linear(896, 1792))
        head.model.mlp_resnet_blocks.append(b)

    prop = nn.Module(); prop.fc1 = nn.Linear(8, 896); prop.fc2 = nn.Linear(896, 896)
    return {"model": hf, "action_head": head, "proprio_projector": prop}


def build_openvla() -> dict:
    hf = nn.Module()
    lm = nn.Module(); lm.model = nn.Module(); lm.model.layers = nn.ModuleList()
    for _ in range(32):
        l = nn.Module(); l.self_attn = nn.Module(); l.mlp = nn.Module()
        for nm in ("q_proj","k_proj","v_proj","o_proj"):
            setattr(l.self_attn, nm, nn.Linear(4096, 4096))
        l.mlp.gate_proj = nn.Linear(4096, 11008)
        l.mlp.up_proj = nn.Linear(4096, 11008)
        l.mlp.down_proj = nn.Linear(11008, 4096)
        lm.model.layers.append(l)
    lm.lm_head = nn.Linear(4096, 32064, bias=False)
    hf.language_model = lm
    hf.vision_backbone = nn.Module()
    hf.vision_backbone.featurizer = _vit(24, 1024, 4096)
    hf.vision_backbone.fused_featurizer = _vit(27, 1152, 4304)
    hf.projector = nn.Module()
    hf.projector.fc1 = nn.Linear(2176, 8704)
    hf.projector.fc2 = nn.Linear(8704, 4096)
    hf.projector.fc3 = nn.Linear(4096, 4096)
    return {"model": hf}


def _by_name(report):
    return {r.name: r for r in report.layers}


def test_vla_adapter_multiroot_and_defaults():
    rep = plan_quantization(build_vla_adapter(), "vla_adapter", QuantSpec())
    recs = _by_name(rep)
    assert {r.root for r in rep.layers} == {"model", "action_head", "proprio_projector"}
    assert not any(r.scope == "other" for r in rep.layers), \
        [r.name for r in rep.layers if r.scope == "other"]
    # default scope is llm only
    assert recs["model.language_model.model.layers.0.mlp.up_proj"].decision == "int4"
    assert recs["model.vision_backbone.featurizer.blocks.0.mlp.fc1"].decision == "bf16"


def test_tied_lm_head_is_never_quantized():
    rep = plan_quantization(build_vla_adapter(), "vla_adapter", QuantSpec(),
                            enabled_scopes={"llm", "lm_head"})
    r = _by_name(rep)["model.language_model.lm_head"]
    assert r.decision == "bf16" and r.skip_reason is SkipReason.TIED_WEIGHT


def test_siglip_4304_falls_back_to_int8_not_silence():
    """in=4304 = 2^4 * 269 divides no allowed group size; fc1's out=4304 is fine
    (4304 % 8 == 0). The layer must be demoted with a recorded reason, not dropped."""
    rep = plan_quantization(build_vla_adapter(), "vla_adapter", QuantSpec(),
                            enabled_scopes={"vision_siglip"})
    recs = _by_name(rep)
    fc2 = recs["model.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"]
    assert fc2.in_features == 4304
    assert fc2.decision == "int8"
    assert fc2.skip_reason is SkipReason.IN_NOT_DIVISIBLE
    fc1 = recs["model.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"]
    assert fc1.decision == "int4" and fc1.group_size == 128


def test_dead_and_constant_input_layers_are_flagged():
    rep = plan_quantization(build_vla_adapter(), "vla_adapter", QuantSpec(),
                            enabled_scopes={"action_head", "action_head_io", "dead"})
    recs = _by_name(rep)
    film = recs["action_head.model.mlp_resnet_blocks.0.film_gen.0"]
    assert film.decision == "bf16" and film.skip_reason is SkipReason.DEAD_WEIGHT
    fc1 = recs["action_head.model.fc1"]
    assert fc1.skip_reason is SkipReason.CONSTANT_INPUT
    # the live attention Linears in the head DO get quantized
    assert recs["action_head.model.mlp_resnet_blocks.0.q_proj"].decision == "int4"


def test_openvla_lm_head_excluded_and_llm_covered():
    rep = plan_quantization(build_openvla(), "openvla", QuantSpec(),
                            enabled_scopes={"llm", "lm_head"})
    recs = _by_name(rep)
    head = recs["language_model.lm_head"]
    assert head.decision == "bf16" and head.skip_reason is SkipReason.NUMERICALLY_CRITICAL
    assert recs["language_model.model.layers.31.mlp.down_proj"].decision == "int4"
    a = rep.accounting()
    # In-scope compression is the INT4 claim; whole-model is what VRAM sees.
    # They differ because the 0.9 B vision tower and the 131 M lm_head stay bf16.
    assert a["compression_in_scope"] > 3.5, a["compression_in_scope"]
    assert 4.1 < a["bits_per_param_in_scope"] < 4.3, a["bits_per_param_in_scope"]
    assert 2.5 < a["compression_whole_model"] < 3.2, a["compression_whole_model"]
    assert a["compression_whole_model"] < a["compression_in_scope"]
    assert a["by_scope"]["llm"]["compression"] > 3.5
    assert a["by_scope"]["vision_siglip"]["compression"] == 1.0


def test_every_linear_is_accounted_for():
    """No silent omission: the report must cover exactly the Linear leaves."""
    from embodiq.quant.walker import find_linear_leaves

    for family, roots in (("vla_adapter", build_vla_adapter()), ("openvla", build_openvla())):
        rep = plan_quantization(roots, family, QuantSpec())
        n = sum(len(find_linear_leaves(r)) for r in roots.values())
        assert len(rep.layers) == n, f"{family}: {len(rep.layers)} records for {n} Linears"
        assert all(r.decision == "bf16" or r.skip_reason is None for r in rep.layers)
