"""Quantization specs and per-layer bookkeeping.

Design rule that makes silent omission structurally impossible: EVERY nn.Linear
in every root lands in exactly one scope bucket and gets exactly one LayerRecord
with a decision and, if skipped, a reason. Anything unmatched falls into "other",
which is never quantized and always printed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Optional


class SkipReason(str, Enum):
    IN_NOT_DIVISIBLE = "in_features not divisible by any allowed group size"
    OUT_NOT_ALIGNED = "out_features not kernel-aligned"
    TOO_SMALL = "numel below min_numel"
    IN_SKIPLIST = "explicit skip-list"
    TIED_WEIGHT = "weight tensor is shared with another module"
    NUMERICALLY_CRITICAL = "modulation / action-head / embedding path"
    NO_CALIB_ACTIVATION = "hook never fired on calibration data"
    DEAD_WEIGHT = "module unreachable at inference"
    CONSTANT_INPUT = "input is constant at inference; error folds into bias"
    SCOPE_DISABLED = "scope not selected"


Decision = Literal["int4", "int8", "bf16"]


@dataclass(frozen=True)
class QuantSpec:
    method: Literal["rtn", "awq", "hsa"] = "rtn"
    w_bit: int = 4
    group_size: int | Literal["auto"] = 128
    sym: bool = False
    # Keep the AWQ input scale inside the module instead of folding it into a
    # preceding op. Folding targets mostly do not exist here: FastWAM's norms are
    # elementwise_affine=False, its self_attn.o is applied outside the block, and
    # Qwen2.5-0.5B is GQA(14/2) so o_proj cannot fold into v_proj.
    explicit_input_scale: bool = True
    # SigLIP's 4304 (= 2^4 * 269) divides by no power of two above 16.
    pad_policy: Literal["pad", "nonpow2_group", "fallback"] = "fallback"
    fallback: Literal["int8_per_channel", "bf16", "error"] = "int8_per_channel"
    min_numel: int = 1 << 16
    simulated: bool = False
    kernel: Literal["auto", "tinygemm", "dequant"] = "auto"
    hsa: Optional["HSASpec"] = None

    @property
    def id(self) -> str:
        g = self.group_size if isinstance(self.group_size, int) else "auto"
        base = f"{self.method}-w{self.w_bit}g{g}"
        if self.hsa is not None:
            base += f"-rho{self.hsa.rho * 100:g}"
        if self.simulated:
            base += "-sim"
        return base

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def bits_per_param(self, group_size: int) -> float:
        """Actual tinygemm cost: b + 16/g (bf16 scale) + 16/g (bf16 zero).

        4.25 at w4/g128. The zero is a bf16 float offset, not a packed int4
        zero-point -- assuming the latter under-reports the footprint by 0.09
        bit/param, which would then disagree with the measured weights_bytes.
        """
        if self.w_bit == 8:
            return 8.0 + 16.0 / group_size
        return self.w_bit + 32.0 / group_size


@dataclass(frozen=True)
class HSASpec:
    granularity: Literal["layer", "channel", "block"] = "channel"
    block: tuple[int, int] = (128, 128)
    criterion: Literal["random", "magnitude", "dw", "dw_act", "hsa"] = "hsa"
    rho: float = 0.01
    # gather, not a contiguous slice: pre-permuting input channels to make the
    # residual contiguous saves 2.3 us but shifts g=128 group boundaries, which
    # changes the main branch's quantization error -- so the HSA main branch would
    # no longer be the same object as the AWQ row it is compared against.
    residual_impl: Literal["gather"] = "gather"
    residual_dtype: str = "float16"
    layer_sens: Literal["leave_one_in", "fisher", "uniform"] = "leave_one_in"
    max_per_layer_rho: float = 0.08
    align_k: int = 8
    requantize_after_extraction: bool = True


@dataclass
class LayerRecord:
    name: str
    root: str
    scope: str
    in_features: int
    out_features: int
    decision: Decision = "bf16"
    skip_reason: Optional[SkipReason] = None
    group_size: Optional[int] = None
    kernel: Optional[str] = None
    n_params: int = 0
    bytes_fp: int = 0
    bytes_q: int = 0
    residual_units: int = 0
    residual_bytes: int = 0
    index_bytes: int = 0
    layer_sensitivity: Optional[float] = None
    calib_out_rel_err: Optional[float] = None
    calib_out_cos: Optional[float] = None
    act_absmax: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skip_reason"] = self.skip_reason.value if self.skip_reason else None
        return d


@dataclass
class QuantReport:
    spec_id: str
    spec_hash: str
    layers: list[LayerRecord] = field(default_factory=list)

    def add(self, rec: LayerRecord) -> None:
        self.layers.append(rec)

    def accounting(self) -> dict:
        by_decision: dict[str, dict] = {}
        by_reason: dict[str, dict] = {}
        for r in self.layers:
            d = by_decision.setdefault(r.decision, {"n": 0, "params": 0})
            d["n"] += 1
            d["params"] += r.n_params
            if r.skip_reason is not None:
                b = by_reason.setdefault(r.skip_reason.value, {"n": 0, "params": 0})
                b["n"] += 1
                b["params"] += r.n_params
        total = sum(r.n_params for r in self.layers) or 1
        quantized = sum(r.n_params for r in self.layers if r.decision != "bf16")
        bytes_fp = sum(r.bytes_fp for r in self.layers)
        bytes_q = sum(r.bytes_q for r in self.layers)
        # Whole-model and in-scope compression are different numbers and both
        # must be reported. LLM-only quantization of OpenVLA is 3.6x on the
        # scope but 2.8x on the model, because the 0.9 B vision tower and
        # lm_head stay bf16. Quoting either one alone misstates the result --
        # and for VLA-Adapter, where the LLM is only 0.5 B of 1.2 B, the gap is
        # far wider.
        inscope = [r for r in self.layers if r.decision != "bf16"]
        in_fp = sum(r.bytes_fp for r in inscope)
        in_q = sum(r.bytes_q for r in inscope)
        # Per-scope breakdown so "which part did we actually shrink" is visible.
        by_scope: dict[str, dict] = {}
        for r in self.layers:
            e = by_scope.setdefault(r.scope, {"n": 0, "params": 0, "bytes_fp": 0, "bytes_q": 0, "quantized_params": 0})
            e["n"] += 1
            e["params"] += r.n_params
            e["bytes_fp"] += r.bytes_fp
            e["bytes_q"] += r.bytes_q
            if r.decision != "bf16":
                e["quantized_params"] += r.n_params
        for e in by_scope.values():
            e["compression"] = (e["bytes_fp"] / e["bytes_q"]) if e["bytes_q"] else None
        return {
            "compression_in_scope": (in_fp / in_q) if in_q else None,
            "bits_per_param_in_scope": (8.0 * in_q / sum(r.n_params for r in inscope)) if inscope else None,
            "by_scope": by_scope,
            "n_layers": len(self.layers),
            "total_params": total,
            "quantized_param_fraction": quantized / total,
            "by_decision": by_decision,
            "by_skip_reason": by_reason,
            "bytes_fp": bytes_fp,
            "bytes_q": bytes_q,
            "compression_whole_model": (bytes_fp / bytes_q) if bytes_q else None,
            "bits_per_param_measured": (8.0 * bytes_q / total) if total else None,
            "residual_bytes": sum(r.residual_bytes for r in self.layers),
            "index_bytes": sum(r.index_bytes for r in self.layers),
            "excluded_modules": sorted(
                r.name for r in self.layers
                if r.decision == "bf16" and r.skip_reason is not SkipReason.SCOPE_DISABLED
            )[:64],
        }

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            **self.accounting(),
            "layers": [r.to_dict() for r in self.layers],
        }

    def render(self, limit: int = 0) -> str:
        a = self.accounting()
        lines = [
            f"quant report {self.spec_id} ({self.spec_hash})",
            f"  layers={a['n_layers']}  params={a['total_params']:,}  "
            f"quantized={100 * a['quantized_param_fraction']:.1f}%",
            f"  bytes {a['bytes_fp'] / 2**20:,.0f} MiB -> {a['bytes_q'] / 2**20:,.0f} MiB"
            f"  whole-model {a['compression_whole_model']:.2f}x"
            f" | in-scope {a['compression_in_scope']:.2f}x"
            f" ({a['bits_per_param_in_scope']:.3f} bit/param)"
            if a["compression_whole_model"] else "  (nothing quantized)",
        ]
        for dec, v in sorted(a["by_decision"].items()):
            lines.append(f"    {dec:<5} {v['n']:>5} layers  {v['params']:>14,} params")
        if a["by_skip_reason"]:
            lines.append("  skipped:")
            for reason, v in sorted(a["by_skip_reason"].items(), key=lambda kv: -kv[1]["params"]):
                lines.append(f"    {v['n']:>5} layers  {v['params']:>14,} params  {reason}")
        return "\n".join(lines)
