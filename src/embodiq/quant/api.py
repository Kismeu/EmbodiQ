"""The single entry point a backend Policy calls to quantize its roots.

This is the "统一评测入口" of task 1.4: all four precision conditions reach the
same rollout through this one callable, and the only thing that differs between
them is the spec and (for AWQ/HSA) a validated artifact.

    fp16  ->  fp16_apply                     (no swap, still reports layer accounting)
    rtn   ->  make_quant_apply(spec)         (weight-only, no calibration input)
    awq   ->  make_quant_apply(spec, artifact=<path>)   per-channel input scales
    hsa   ->  make_quant_apply(spec, artifact=<path>)   AWQ main branch + residual

Usage from a Policy::

    apply = make_quant_apply(QuantSpec(method="rtn"), scopes={"llm"})
    policy = VLAAdapterPolicy(ckpt, suite, quant_apply=apply)

The callable takes ``(roots, family)`` and returns a QuantReport. It must be
invoked AFTER the checkpoint is loaded -- see swap.apply_report.

Two rules this module enforces rather than documents:

* **No method runs under another method's label.** AWQ and HSA require an
  artifact that passes every check in :mod:`embodiq.quant.artifact`; a missing or
  rejected artifact raises. There is no fallback to RTN, because a silent
  fallback produces a complete, plausible, wrong set of results.
* **One module range per model, shared by all methods.** The artifact must cover
  exactly the layers this model's plan marks int4 (checked in ``artifact.bind``),
  so an AWQ row cannot be a different scope than the RTN row it is compared to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from embodiq.quant.spec import QuantReport, QuantSpec
from embodiq.quant.swap import apply_report, assert_no_linear_left_in_scope
from embodiq.quant.walker import DEFAULT_SCOPES, plan_quantization

ARTIFACT_METHODS = ("awq", "hsa")


def make_quant_apply(
    spec: QuantSpec,
    *,
    scopes: Optional[Iterable[str]] = None,
    artifact: Optional[str | Path] = None,
    extra_skip: tuple[str, ...] = (),
    verbose: bool = True,
) -> Callable[[Mapping[str, object], str], QuantReport]:
    """Build the quantize-on-load callable for one run.

    Args:
        spec: method, bit width, group size, and (for HSA) the residual budget.
        scopes: the module range. Defaults to this family's DEFAULT_SCOPES.
        artifact: required for ``awq``/``hsa``; rejected for ``rtn``.
    """
    if spec.method in ARTIFACT_METHODS and artifact is None:
        raise ValueError(
            f"method {spec.method!r} requires a quantization artifact produced by "
            f"its own calibration and scale/residual search. Pass "
            f"--quant-artifact. Running without one would execute RTN under an "
            f"{spec.method.upper()} label."
        )
    if spec.method not in ARTIFACT_METHODS and artifact is not None:
        raise ValueError(
            f"method {spec.method!r} takes no artifact; {artifact} would be ignored"
        )
    if spec.method not in ("rtn",) + ARTIFACT_METHODS:
        raise ValueError(f"unknown quantization method {spec.method!r}")

    def apply(roots: Mapping[str, object], family: str) -> QuantReport:
        from embodiq.quant.artifact import QuantArtifact

        before_bytes = tensor_storage_bytes(roots)
        chosen = set(scopes) if scopes is not None else set(DEFAULT_SCOPES[family])
        report = plan_quantization(roots, family, spec, enabled_scopes=chosen,
                                   extra_skip=extra_skip)
        if not any(r.decision in ("int4", "int8") for r in report.layers):
            raise ValueError(f"No layers selected for {family}: {sorted(chosen)}")

        bound = None
        if artifact is not None:
            # Loads, then validates the entire artifact against the loaded
            # weights BEFORE anything is replaced. A rejected artifact leaves the
            # model untouched rather than half-converted.
            loaded = QuantArtifact.load(artifact)
            bound = loaded.bind(roots, report, spec, family, scopes=chosen)
            if verbose:
                m = bound.meta
                print(f"  artifact {m['implementation']} v{m['implementation_version']} "
                      f"({m['artifact_sha256'][:8]}): {m['n_layers']} layers, "
                      f"{m['n_layers_with_input_scale']} input scales, "
                      f"{m['n_layers_with_clip']} clipped, "
                      f"{m['n_layers_with_residual']} with residual; "
                      f"method overhead {m['method_overhead_bytes'] / 2**20:,.1f} MiB")

        result = apply_report(
            roots, report, spec, verify=True,
            input_scales=bound.input_scales if bound else None,
            clips=bound.clips if bound else None,
            residuals=bound.residuals if bound else None,
        )
        assert_no_linear_left_in_scope(roots, report)
        report.swap = result  # type: ignore[attr-defined]
        report.tensor_storage_before = before_bytes
        report.tensor_storage_after = tensor_storage_bytes(roots)
        # Travels into the result JSON so an AWQ/HSA row names the implementation,
        # its version, its calibration set and its source checkpoint.
        report.artifact = bound.meta if bound else None  # type: ignore[attr-defined]
        if verbose:
            print(report.render())
            print(f"  swapped {result.n_replaced} modules; "
                  f"param fingerprint {result.fingerprint_before[:8]} -> "
                  f"{result.fingerprint_after[:8]}")
        return report

    return apply


def tensor_storage_bytes(roots):
    seen = set()
    total = 0
    for root in roots.values():
        for tensor in list(root.parameters()) + list(root.buffers()):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def fp16_apply(roots: Mapping[str, object], family: str) -> QuantReport:
    """No-op baseline that still produces a report, so FP16 rows carry the same
    layer accounting as quantized ones."""
    spec = QuantSpec(method="rtn", w_bit=16)
    return plan_quantization(roots, family, spec, enabled_scopes=set())
