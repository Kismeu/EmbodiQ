"""Fail-closed loading of AWQ / HSA quantization artifacts produced elsewhere.

This module is the contract surface for task 1.4. The AWQ scale search (1.2) and
the HSA sensitivity/residual selection (1.3) are produced outside this evaluation
path; the shared entry accepts their output ONLY through the checks below, and
there is deliberately no other way to run `--quant awq` or `--quant hsa`.

What the checks are defending against, each one a way a 48-cell table silently
becomes meaningless:

* **A relabelled RTN run.** `--quant awq` with no artifact used to be allowed by
  one code path in this repo and would have produced a full set of AWQ-labelled
  numbers that were really RTN. An artifact is now mandatory, and an AWQ artifact
  carrying no input scales -- or an HSA artifact carrying no residuals -- is
  rejected rather than quietly degraded.
* **A different module range.** The protocol requires RTN/AWQ/HSA to share one
  per-model scope; otherwise the matrix is a collage of whichever range happened
  to score best. `bind()` therefore demands the artifact cover *exactly* the
  layers this model's plan marks int4 -- no extras, no omissions.
* **A different checkpoint.** Scales and residuals are only valid for the weights
  they were searched on. Every layer carries the sha256 of the weight and bias it
  was produced from, checked against the tensors actually loaded.
* **Partial application.** Every layer is validated before any module is touched,
  so a bad artifact leaves the model untouched instead of half-converted.

The accounting fields (residual bytes, index bytes, scale bytes) are filled in so
that an HSA row's real footprint -- low-bit main branch *plus* high-precision
residual *plus* its gather index -- lands in the memory column, as the protocol
requires. Fixing the base bit width does not equalize the total budget.

Format (`torch.save` of a plain dict, `weights_only=True`-loadable)::

    {
      "format": "embodiq/quant-artifact/1",
      "method": "awq" | "hsa",
      "implementation": "awq_block_v1",       # free-form, recorded in results
      "implementation_version": "1.0",
      "producer_code_sha256": "<hex>",        # the code that produced this
      "model_family": "vla_adapter",
      "suite": "libero_spatial",
      "w_bit": 4,
      "group_size": 128,
      "source_checkpoint_sha256": "<hex>",    # digest of the checkpoint inputs
      "calibration": {"manifest_sha256": "<hex>", "n_calibration": 1280,
                      "n_development": 320, "source": "..."},
      "scope": ["llm", "action_head"],
      "layers": {
        "<module name>": {
          "weight_sha256": "<hex>",
          "bias_sha256": "<hex>" | None,
          "input_scale": Tensor[in_features],              # AWQ, required for awq
          "clip_max":    Tensor[out_features, in/g] | None, # AWQ, optional
          "residual_idx": Tensor[k] int,                   # HSA, required for hsa
          "residual_val": Tensor[out_features, k],         # HSA, required for hsa
        }, ...
      }
    }

`quant/awq_artifact.py` keeps the narrower single-block pilot format and is not
used by the evaluation entry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from embodiq.quant.spec import QuantReport, QuantSpec

FORMAT = "embodiq/quant-artifact/1"
METHODS = ("awq", "hsa")
REQUIRED_META = (
    "format", "method", "implementation", "implementation_version",
    "producer_code_sha256", "model_family", "w_bit", "group_size",
    "source_checkpoint_sha256", "calibration", "scope", "layers",
)


class ArtifactError(ValueError):
    """Raised for any artifact that cannot be trusted. Never downgraded."""


def tensor_sha256(tensor: Optional[torch.Tensor]) -> Optional[str]:
    """Content hash of a tensor's raw bytes, dtype-independent."""
    if tensor is None:
        return None
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class BoundArtifact:
    """Validated tensors, keyed by module name, ready for swap.apply_report."""

    input_scales: dict[str, torch.Tensor] = field(default_factory=dict)
    clips: dict[str, torch.Tensor] = field(default_factory=dict)
    residuals: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class QuantArtifact:
    path: Path
    payload: dict

    # -- loading ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "QuantArtifact":
        path = Path(path)
        if not path.is_file():
            raise ArtifactError(f"quantization artifact not found: {path}")
        # weights_only: an artifact is data, and must never be able to execute.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ArtifactError(f"{path}: artifact must be a dict, got {type(payload).__name__}")
        missing = [k for k in REQUIRED_META if k not in payload]
        if missing:
            raise ArtifactError(f"{path}: artifact is missing required keys {missing}")
        if payload["format"] != FORMAT:
            raise ArtifactError(
                f"{path}: unsupported artifact format {payload['format']!r}; "
                f"this evaluation entry reads {FORMAT!r}"
            )
        if payload["method"] not in METHODS:
            raise ArtifactError(
                f"{path}: method {payload['method']!r} is not one of {METHODS}; "
                f"fp16/rtn need no artifact and must not be supplied as one"
            )
        for key in ("implementation", "implementation_version",
                    "producer_code_sha256", "source_checkpoint_sha256"):
            if not isinstance(payload[key], str) or not payload[key].strip():
                raise ArtifactError(f"{path}: {key} must be a non-empty string")
        if not isinstance(payload["layers"], dict) or not payload["layers"]:
            raise ArtifactError(f"{path}: artifact contains no layers")
        if not isinstance(payload["calibration"], dict) or \
                not payload["calibration"].get("manifest_sha256"):
            raise ArtifactError(
                f"{path}: calibration.manifest_sha256 is required -- an AWQ/HSA row "
                f"without a recorded calibration set is not reproducible"
            )
        return cls(path=path, payload=payload)

    # -- accessors -------------------------------------------------------------

    @property
    def method(self) -> str:
        return self.payload["method"]

    @property
    def layers(self) -> dict:
        return self.payload["layers"]

    def metadata(self) -> dict:
        """Everything that must travel into the result JSON (minus tensors)."""
        return {
            "artifact_path": str(self.path),
            "artifact_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "format": self.payload["format"],
            "method": self.payload["method"],
            "implementation": self.payload["implementation"],
            "implementation_version": self.payload["implementation_version"],
            "producer_code_sha256": self.payload["producer_code_sha256"],
            "model_family": self.payload["model_family"],
            "suite": self.payload.get("suite"),
            "w_bit": self.payload["w_bit"],
            "group_size": self.payload["group_size"],
            "source_checkpoint_sha256": self.payload["source_checkpoint_sha256"],
            "calibration": self.payload["calibration"],
            "scope": list(self.payload["scope"]),
            "n_layers": len(self.layers),
            # Missing flags mean eligibility has not been attested. Development
            # artifacts remain loadable for diagnostics but cannot enter a report.
            "formal": self.payload.get("formal"),
            "validation_only": self.payload.get("validation_only"),
        }

    # -- validation ------------------------------------------------------------

    def bind(
        self,
        roots: Mapping[str, nn.Module],
        report: QuantReport,
        spec: QuantSpec,
        family: str,
        *,
        scopes: Optional[set[str]] = None,
    ) -> BoundArtifact:
        """Validate the whole artifact against the loaded model; mutate nothing.

        Returns the tensors keyed by module name. Any failure raises, leaving the
        model exactly as it was -- a half-applied artifact is the one outcome that
        would be invisible in the results.
        """
        p = self.payload
        if p["method"] != spec.method:
            raise ArtifactError(
                f"{self.path}: artifact method {p['method']!r} != requested "
                f"{spec.method!r}; refusing to run one method under another's label"
            )
        if p["model_family"] != family:
            raise ArtifactError(
                f"{self.path}: artifact is for {p['model_family']!r}, loaded model "
                f"is {family!r}"
            )
        if p["w_bit"] != spec.w_bit or p["group_size"] != spec.group_size:
            raise ArtifactError(
                f"{self.path}: artifact is w{p['w_bit']}g{p['group_size']}, run spec "
                f"is w{spec.w_bit}g{spec.group_size}"
            )
        if scopes is not None and set(p["scope"]) != set(scopes):
            raise ArtifactError(
                f"{self.path}: artifact scope {sorted(p['scope'])} != run scope "
                f"{sorted(scopes)}. RTN/AWQ/HSA must share one per-model module "
                f"range, or the matrix mixes ranges."
            )

        # The artifact must cover exactly the int4 layers the plan selected.
        # int8-fallback layers are excluded on purpose: a per-output-channel INT8
        # module takes neither a group-wise input scale nor a group clip.
        planned = {r.name for r in report.layers if r.decision == "int4"}
        have = set(self.layers)
        if have != planned:
            extra, absent = sorted(have - planned), sorted(planned - have)
            raise ArtifactError(
                f"{self.path}: artifact layer set does not match the plan's int4 "
                f"layers ({len(planned)} planned, {len(have)} in artifact). "
                f"{len(extra)} not in plan (e.g. {extra[:3]}), "
                f"{len(absent)} missing from artifact (e.g. {absent[:3]})."
            )

        group = spec.group_size if isinstance(spec.group_size, int) else None
        bound = BoundArtifact(meta=self.metadata())
        n_scale_bytes = n_residual_bytes = n_index_bytes = 0
        n_residual_units = 0
        records = {r.name: r for r in report.layers}

        for name in sorted(self.layers):
            row = self.layers[name]
            if not isinstance(row, dict):
                raise ArtifactError(f"{self.path}: layer {name} is not a dict")
            module = _resolve(roots, name)
            if not isinstance(module, nn.Linear):
                raise ArtifactError(
                    f"{self.path}: {name} is {type(module).__name__}, not a plain "
                    f"nn.Linear -- was the model already quantized?"
                )
            if tensor_sha256(module.weight) != row.get("weight_sha256"):
                raise ArtifactError(
                    f"{self.path}: {name} weight hash mismatch; the artifact was "
                    f"produced from a different checkpoint"
                )
            if tensor_sha256(module.bias) != row.get("bias_sha256"):
                raise ArtifactError(f"{self.path}: {name} bias hash mismatch")

            scale = row.get("input_scale")
            if scale is not None:
                _check_finite(scale, f"{name}.input_scale", self.path)
                if tuple(scale.shape) != (module.in_features,):
                    raise ArtifactError(
                        f"{self.path}: {name}.input_scale shape {tuple(scale.shape)} "
                        f"!= ({module.in_features},)"
                    )
                if not bool((scale > 0).all()):
                    raise ArtifactError(
                        f"{self.path}: {name}.input_scale must be strictly positive "
                        f"(it is inverted at execution time)"
                    )
                bound.input_scales[name] = scale
                n_scale_bytes += scale.numel() * 2  # stored bf16 in QuantLinear

            clip = row.get("clip_max")
            if clip is not None:
                _check_finite(clip, f"{name}.clip_max", self.path)
                if group is None:
                    raise ArtifactError(
                        f"{self.path}: {name}.clip_max needs a fixed group size"
                    )
                want = (module.out_features, module.in_features // group)
                if tuple(clip.shape) != want:
                    raise ArtifactError(
                        f"{self.path}: {name}.clip_max shape {tuple(clip.shape)} != {want}"
                    )
                if bool((clip < 0).any()):
                    raise ArtifactError(f"{self.path}: {name}.clip_max must be >= 0")
                bound.clips[name] = clip

            idx, val = row.get("residual_idx"), row.get("residual_val")
            if (idx is None) != (val is None):
                raise ArtifactError(
                    f"{self.path}: {name} has only one of residual_idx/residual_val"
                )
            if idx is not None:
                if idx.dtype not in (torch.int16, torch.int32, torch.int64):
                    raise ArtifactError(
                        f"{self.path}: {name}.residual_idx must be an integer tensor, "
                        f"got {idx.dtype}"
                    )
                if idx.ndim != 1 or idx.numel() == 0:
                    raise ArtifactError(f"{self.path}: {name}.residual_idx must be 1-D non-empty")
                if int(idx.min()) < 0 or int(idx.max()) >= module.in_features:
                    raise ArtifactError(
                        f"{self.path}: {name}.residual_idx out of range for "
                        f"in_features={module.in_features}"
                    )
                if idx.unique().numel() != idx.numel():
                    raise ArtifactError(
                        f"{self.path}: {name}.residual_idx has duplicate channels; "
                        f"the gather would double-count the correction"
                    )
                _check_finite(val, f"{name}.residual_val", self.path)
                want = (module.out_features, idx.numel())
                if tuple(val.shape) != want:
                    raise ArtifactError(
                        f"{self.path}: {name}.residual_val shape {tuple(val.shape)} != {want}"
                    )
                share = idx.numel() / module.in_features
                cap = spec.hsa.max_per_layer_rho if spec.hsa is not None else None
                if cap is not None and share > cap + 1e-9:
                    raise ArtifactError(
                        f"{self.path}: {name} keeps {share:.3%} of input channels in "
                        f"high precision, above max_per_layer_rho={cap:.3%}"
                    )
                bound.residuals[name] = (idx, val)
                n_residual_units += idx.numel()
                n_residual_bytes += val.numel() * 2   # bf16 residual
                n_index_bytes += idx.numel() * 4      # int32 gather index
                rec = records.get(name)
                if rec is not None:
                    rec.residual_units = int(idx.numel())
                    rec.residual_bytes = int(val.numel() * 2)
                    rec.index_bytes = int(idx.numel() * 4)

        _check_method_shape(self, bound)
        bound.meta.update(
            n_layers_with_input_scale=len(bound.input_scales),
            n_layers_with_clip=len(bound.clips),
            n_layers_with_residual=len(bound.residuals),
            residual_units=n_residual_units,
            residual_bytes=n_residual_bytes,
            index_bytes=n_index_bytes,
            input_scale_bytes=n_scale_bytes,
            # The protocol counts residual + index + scale against the method's
            # budget: equal base bit width is not equal total budget.
            method_overhead_bytes=n_residual_bytes + n_index_bytes + n_scale_bytes,
        )
        return bound


def _check_method_shape(artifact: "QuantArtifact", bound: BoundArtifact) -> None:
    """An AWQ artifact that is really RTN, or an HSA artifact that is really AWQ,
    would both produce plausible numbers under the wrong label. Reject both."""
    if artifact.method == "awq":
        if not bound.input_scales:
            raise ArtifactError(
                f"{artifact.path}: method 'awq' but no layer carries an input_scale; "
                f"that is RTN under an AWQ label"
            )
        if bound.residuals:
            raise ArtifactError(
                f"{artifact.path}: method 'awq' must not carry high-precision "
                f"residuals ({len(bound.residuals)} layers do); that is HSA"
            )
    elif artifact.method == "hsa":
        if not bound.residuals:
            raise ArtifactError(
                f"{artifact.path}: method 'hsa' but no layer carries a residual; "
                f"without residual compensation this is the AWQ/RTN main branch"
            )


def _check_finite(tensor: torch.Tensor, label: str, path: Path) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ArtifactError(f"{path}: {label} must be a tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ArtifactError(f"{path}: {label} contains non-finite values")


def _resolve(roots: Mapping[str, nn.Module], dotted: str) -> nn.Module:
    """Resolve a plan-style module name against the same roots swap.py uses."""
    root_name, _, local = dotted.partition(".") if len(roots) > 1 else ("", "", dotted)
    root = roots[root_name] if root_name in roots else next(iter(roots.values()))
    try:
        return root.get_submodule(local or dotted)
    except AttributeError as exc:
        raise ArtifactError(f"artifact names a module that does not exist: {dotted}") from exc
