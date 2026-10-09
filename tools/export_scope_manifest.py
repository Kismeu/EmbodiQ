"""Export the exact layer list an AWQ/HSA artifact must cover, per model.

`quant_artifact_contract.md` tells 1.2/1.3 that their artifact must cover exactly
the layers this model's plan marks int4 — and `QuantArtifact.bind` enforces it.
Telling them a *count* is not enough: a count they can match with the wrong set
still fails, after they have spent the calibration. So this exports the layer
names themselves, taken from the per-layer reports the completed RTN runs already
wrote, which is the same plan the evaluator will compute at load time.

Two things it verifies rather than assumes:

* **The layer set is identical across a model's four suites.** The checkpoints
  differ per suite but the architecture does not, so a divergence would mean the
  scope is not actually a per-model property and the matrix could not share one
  module range across suites.
* **The counts match what the contract document publishes.** A silent drift
  between the doc and the plan is exactly what would make a correct artifact get
  rejected on arrival.

Usage:
    python tools/export_scope_manifest.py
    python tools/export_scope_manifest.py --out artifacts/scope_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from embodiq.entry.run_adapter import MODEL_SCOPES  # noqa: E402

REPORT_ROOT = ROOT / "results/stage5_rtn_full"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# Published in quant_artifact_contract.md; an artifact is validated against these.
DOCUMENTED_INT4 = {"openvla": 224, "vla_adapter": 568, "fastwam": 600}
DOCUMENTED_INT8 = {"openvla": 0, "vla_adapter": 28, "fastwam": 0}


def load_reports(model: str) -> dict[str, dict]:
    out = {}
    for suite in SUITES:
        path = REPORT_ROOT / model / "quant_reports" / f"{suite}_rtn.json"
        if path.is_file():
            out[suite] = json.loads(path.read_text())
    return out


def layer_sets(report: dict) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"int4": [], "int8": [], "bf16": []}
    for layer in report["layers"]:
        buckets.setdefault(layer["decision"], []).append(layer["name"])
    return {k: sorted(v) for k, v in buckets.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/scope_manifest.json")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails (for CI)")
    args = ap.parse_args()

    doc = {
        "schema": "embodiq/scope_manifest/1",
        "generated": time.strftime("%FT%T%z"),
        "purpose": ("The exact layer set an AWQ/HSA artifact must cover, per model. "
                    "Derived from the per-layer reports of the completed RTN runs, "
                    "which are the same plan the evaluator recomputes at load time."),
        "contract": "docs/quant_artifact_contract.md",
        "group_size": 128,
        "w_bit": 4,
        "models": {},
    }
    problems: list[str] = []

    for model, scope in MODEL_SCOPES.items():
        reports = load_reports(model)
        if not reports:
            problems.append(f"{model}: no quant report found under {REPORT_ROOT}")
            continue

        sets = {suite: layer_sets(r) for suite, r in reports.items()}
        reference_suite = sorted(sets)[0]
        reference = sets[reference_suite]

        # The architecture, hence the plan, must not depend on the suite.
        for suite, other in sets.items():
            for decision in ("int4", "int8"):
                if other[decision] != reference[decision]:
                    only_a = sorted(set(other[decision]) - set(reference[decision]))
                    only_b = sorted(set(reference[decision]) - set(other[decision]))
                    problems.append(
                        f"{model}: {decision} layer set differs between "
                        f"{reference_suite} and {suite} "
                        f"(+{len(only_a)} e.g. {only_a[:2]}, "
                        f"-{len(only_b)} e.g. {only_b[:2]})")

        n4, n8 = len(reference["int4"]), len(reference["int8"])
        if DOCUMENTED_INT4.get(model) != n4:
            problems.append(f"{model}: plan has {n4} int4 layers, the contract "
                            f"document publishes {DOCUMENTED_INT4.get(model)}")
        if DOCUMENTED_INT8.get(model) != n8:
            problems.append(f"{model}: plan has {n8} int8 fallbacks, the contract "
                            f"document publishes {DOCUMENTED_INT8.get(model)}")

        scope_in_report = sorted(reports[reference_suite].get("by_scope", {}))
        doc["models"][model] = {
            "scope": list(scope),
            "scopes_present_in_plan": scope_in_report,
            "suites_checked": sorted(sets),
            "n_int4": n4,
            "n_int8_fallback": n8,
            "quantized_param_fraction": reports[reference_suite].get(
                "quantized_param_fraction"),
            "compression_in_scope": reports[reference_suite].get("compression_in_scope"),
            # The artifact must cover exactly this list.
            "artifact_must_cover": reference["int4"],
            # And must NOT include these: per-output-channel INT8 takes neither a
            # group-wise input scale nor a group clip.
            "artifact_must_exclude_int8_fallback": reference["int8"],
            "int4_sha256": hashlib.sha256(
                "\n".join(reference["int4"]).encode()).hexdigest(),
        }
        print(f"{model:12s} int4={n4:4d} int8={n8:3d}  suites checked: "
              f"{len(sets)}  scope={'+'.join(scope)}")

    doc["problems"] = problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    for p in problems:
        print(f"  PROBLEM {p}")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
