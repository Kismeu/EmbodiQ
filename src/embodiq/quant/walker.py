"""Module-tree walker: find nn.Linear leaves, assign scopes, decide eligibility.

Multi-root by construction. Only OpenVLA is a single HF model; VLA-Adapter adds
loose .pt modules (action head, proprio projector) that live outside the HF
model entirely, and FastWAM has four roots (video DiT, ActionDiT, UMT5, VAE).
A walker piloted on a single-root model gets this wrong and then breaks on two
of the three.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Sequence

from embodiq.quant.spec import Decision, LayerRecord, QuantReport, QuantSpec, SkipReason

ScopeRule = tuple[str, str]  # (regex, scope)

# First match wins. Unmatched -> "other", which is never quantized and always
# reported. Verified against the actual module names in each repo.
SCOPE_RULES: dict[str, list[ScopeRule]] = {
    "openvla": [
        (r"^language_model\.lm_head$", "lm_head"),
        (r"^language_model\.model\.layers\.", "llm"),
        (r"^vision_backbone\.featurizer\.", "vision_dino"),
        (r"^vision_backbone\.fused_featurizer\.", "vision_siglip"),
        (r"^projector\.", "projector"),
    ],
    "vla_adapter": [
        # Qwen2.5-0.5B ties lm_head to embed_tokens; the tie check catches it,
        # but scope it explicitly so it is visible in the report.
        (r"^language_model\.lm_head$", "lm_head"),
        (r"^language_model\.model\.layers\.", "llm"),
        (r"^vision_backbone\.featurizer\.", "vision_dino"),
        (r"^vision_backbone\.fused_featurizer\.", "vision_siglip"),
        (r"^projector\.", "projector"),
        # 24 x Linear(896, 1792) kept only for checkpoint compatibility; the
        # forward that used it is commented out. 38.5 M params = 19% of the head.
        (r"^model\.mlp_resnet_blocks\.\d+\.film_gen\.", "dead"),
        (r"^model\.mlp_resnet_blocks\.", "action_head"),
        (r"^model\.fc1$", "action_head_io"),   # zero input at inference
        (r"^model\.fc2$", "action_head_io"),
        (r"^fc[12]$", "proprio"),
    ],
    "fastwam": [
        # Real FastWAM registers these aliases before mot/dit. named_modules()
        # deduplicates shared experts and exposes the first registered names.
        (r"^video_expert\.blocks\.", "video_dit"),
        (r"^action_expert\.blocks\.", "action_dit"),
        (r"^(video_expert|action_expert)\.(text_embedding|time_embedding|time_projection|head|action_encoder)", "dit_io"),
        (r"^mot\.mixtures\.video\.blocks\.", "video_dit"),
        (r"^mot\.mixtures\.action\.blocks\.", "action_dit"),
        (r"^mot\.mixtures\.\w+\.(text_embedding|time_embedding|time_projection|head|action_encoder)", "dit_io"),
        (r"^text_encoder\.blocks\.", "text_encoder"),
        (r"^proprio_encoder", "proprio"),
    ],
}

# Never quantized regardless of scope selection, each for a specific reason.
HARD_SKIP: dict[str, list[tuple[str, SkipReason, str]]] = {
    "openvla": [
        (r"^language_model\.lm_head$", SkipReason.NUMERICALLY_CRITICAL,
         "action correctness is pure argmax ranking over the last 256 vocab logits; "
         "131 M params = 3.5% of the model, so keeping bf16 is nearly free"),
    ],
    "vla_adapter": [
        (r"film_gen", SkipReason.DEAD_WEIGHT,
         "forward is commented out in action_heads.py; report as disclosure, "
         "never credit the saving to quantization"),
        (r"^model\.fc1$", SkipReason.CONSTANT_INPUT,
         "rearranged_actions_hidden_states is all-zero at inference, so the "
         "quantization error is a deterministic bias -- fold it into .bias instead"),
    ],
    "fastwam": [
        (r"time_projection", SkipReason.NUMERICALLY_CRITICAL,
         "feeds AdaLN modulation for every token in every block"),
        (r"\.head(\.head)?$", SkipReason.NUMERICALLY_CRITICAL, "output projection"),
        (r"action_encoder", SkipReason.NUMERICALLY_CRITICAL, "7-dim input"),
    ],
}

DEFAULT_SCOPES: dict[str, set[str]] = {
    "openvla": {"llm"},
    "vla_adapter": {"llm"},
    "fastwam": {"video_dit", "action_dit"},
}


def find_linear_leaves(root, prefix: str = "") -> dict[str, object]:
    import torch.nn as nn

    out = {}
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Linear):
            out[f"{prefix}{name}" if prefix else name] = mod
    return out


def assign_scope(name: str, rules: Sequence[ScopeRule]) -> str:
    for pattern, scope in rules:
        if re.search(pattern, name):
            return scope
    return "other"


def resolve_group_size(in_features: int, spec: QuantSpec) -> Optional[int]:
    """Largest allowed group dividing in_features, or None."""
    if spec.group_size != "auto":
        g = int(spec.group_size)
        return g if in_features % g == 0 else None
    for g in (128, 64, 32):
        if in_features % g == 0:
            return g
    return None


def plan_quantization(
    roots: Mapping[str, object],
    family: str,
    spec: QuantSpec,
    *,
    enabled_scopes: Optional[Iterable[str]] = None,
    extra_skip: Sequence[str] = (),
) -> QuantReport:
    """Decide int4 / int8 / bf16 for every Linear leaf across every root."""
    import torch

    scopes = set(enabled_scopes) if enabled_scopes is not None else set(DEFAULT_SCOPES[family])
    rules = SCOPE_RULES[family]
    hard = HARD_SKIP.get(family, [])
    report = QuantReport(spec_id=spec.id, spec_hash=spec.spec_hash)

    # Detect tied weights across ALL roots: quantizing one half of a tie
    # corrupts the other (Qwen2.5-0.5B shares lm_head with embed_tokens).
    seen_ids: dict[int, str] = {}
    for root_name, root in roots.items():
        for pname, p in root.named_parameters():
            seen_ids.setdefault(id(p), f"{root_name}.{pname}")

    for root_name, root in roots.items():
        for name, mod in find_linear_leaves(root).items():
            full = f"{root_name}.{name}" if len(roots) > 1 else name
            scope = assign_scope(name, rules)
            in_f, out_f = mod.in_features, mod.out_features
            numel = in_f * out_f
            rec = LayerRecord(
                name=full, root=root_name, scope=scope,
                in_features=in_f, out_features=out_f, n_params=numel,
                bytes_fp=numel * 2,
            )

            def finish(decision: Decision, reason: Optional[SkipReason], note: str = "",
                       g: Optional[int] = None) -> None:
                rec.decision = decision
                rec.skip_reason = reason
                rec.note = note
                rec.group_size = g
                if decision == "int4":
                    rec.bytes_q = int(numel * spec.bits_per_param(g) / 8)
                    rec.kernel = "tinygemm"
                elif decision == "int8":
                    rec.bytes_q = int(numel * (8.0 + 16.0 / in_f) / 8)
                    rec.kernel = "int8_per_channel"
                else:
                    rec.bytes_q = rec.bytes_fp
                report.add(rec)

            # --- ordered eligibility checks --------------------------------
            hit = next(((r, n) for pat, r, n in hard if re.search(pat, name)), None)
            if hit is not None:
                finish("bf16", hit[0], hit[1]); continue
            if any(re.search(p, full) for p in extra_skip):
                finish("bf16", SkipReason.IN_SKIPLIST); continue
            if scope not in scopes:
                finish("bf16", SkipReason.SCOPE_DISABLED); continue
            if scope == "dead":
                finish("bf16", SkipReason.DEAD_WEIGHT); continue
            owner = seen_ids.get(id(mod.weight))
            if owner and not owner.endswith(f"{name}.weight"):
                finish("bf16", SkipReason.TIED_WEIGHT, f"shares storage with {owner}"); continue
            if numel < spec.min_numel:
                finish("bf16", SkipReason.TOO_SMALL); continue

            g = resolve_group_size(in_f, spec)
            if g is None:
                # SigLIP mlp.fc2: in=4304 = 2^4 * 269.
                if spec.pad_policy == "fallback" and spec.fallback == "int8_per_channel":
                    finish("int8", SkipReason.IN_NOT_DIVISIBLE,
                           f"in_features={in_f} indivisible; per-channel INT8 has no such constraint")
                elif spec.pad_policy == "fallback" and spec.fallback == "bf16":
                    finish("bf16", SkipReason.IN_NOT_DIVISIBLE)
                else:
                    raise ValueError(f"{full}: in_features={in_f} not divisible and pad_policy={spec.pad_policy}")
                continue
            # tinygemm packs 8 int4 along the output dim.
            if out_f % 8 != 0:
                finish("int8", SkipReason.OUT_NOT_ALIGNED, f"out_features={out_f} not a multiple of 8")
                continue

            finish("int4" if spec.w_bit == 4 else "int8", None, g=g)

    return report
