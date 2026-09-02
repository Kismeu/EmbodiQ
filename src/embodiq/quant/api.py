"""The single entry point a backend Policy calls to quantize its roots.

Usage from a Policy::

    apply = make_quant_apply(QuantSpec(method="rtn"), scopes={"llm"})
    policy = VLAAdapterPolicy(ckpt, suite, quant_apply=apply)

The callable takes ``(roots, family)`` and returns a QuantReport. It must be
invoked AFTER the checkpoint is loaded -- see swap.apply_report.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional

from embodiq.quant.spec import QuantReport, QuantSpec
from embodiq.quant.swap import apply_report, assert_no_linear_left_in_scope
from embodiq.quant.walker import DEFAULT_SCOPES, plan_quantization


def make_quant_apply(
    spec: QuantSpec,
    *,
    scopes: Optional[Iterable[str]] = None,
    extra_skip: tuple[str, ...] = (),
    verbose: bool = True,
) -> Callable[[Mapping[str, object], str], QuantReport]:
    def apply(roots: Mapping[str, object], family: str) -> QuantReport:
        chosen = set(scopes) if scopes is not None else set(DEFAULT_SCOPES[family])
        report = plan_quantization(roots, family, spec, enabled_scopes=chosen,
                                   extra_skip=extra_skip)
        result = apply_report(roots, report, spec, verify=True)
        assert_no_linear_left_in_scope(roots, report)
        report.swap = result  # type: ignore[attr-defined]
        if verbose:
            print(report.render())
            print(f"  swapped {result.n_replaced} modules; "
                  f"param fingerprint {result.fingerprint_before[:8]} -> "
                  f"{result.fingerprint_after[:8]}")
        return report

    return apply


def fp16_apply(roots: Mapping[str, object], family: str) -> QuantReport:
    """No-op baseline that still produces a report, so FP16 rows carry the same
    layer accounting as quantized ones."""
    spec = QuantSpec(method="rtn", w_bit=16)
    return plan_quantization(roots, family, spec, enabled_scopes=set())
