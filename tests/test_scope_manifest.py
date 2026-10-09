"""The layer set an AWQ/HSA artifact must cover, pinned against reality.

`QuantArtifact.bind` rejects an artifact whose layer set is not exactly the plan's
int4 set, and `quant_artifact_contract.md` publishes the counts 1.2/1.3 build
against. If the published numbers and the plan drift apart, a correct artifact is
rejected on arrival — after the calibration has been spent. These tests tie the
document, the tool and the saved reports of the real runs together.

They read the per-layer reports written by the completed RTN runs and skip when
those are absent, so the suite still passes on a fresh checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import export_scope_manifest as ex  # noqa: E402


def reports_for(model):
    found = ex.load_reports(model)
    if not found:
        pytest.skip(f"no saved quant report for {model}")
    return found


@pytest.mark.parametrize("model", sorted(ex.DOCUMENTED_INT4))
def test_plan_matches_the_published_layer_counts(model):
    """The contract document's numbers are what upstream builds against."""
    reports = reports_for(model)
    sets = ex.layer_sets(reports[sorted(reports)[0]])
    assert len(sets["int4"]) == ex.DOCUMENTED_INT4[model]
    assert len(sets["int8"]) == ex.DOCUMENTED_INT8[model]


@pytest.mark.parametrize("model", sorted(ex.DOCUMENTED_INT4))
def test_the_layer_set_does_not_depend_on_the_suite(model):
    """Checkpoints differ per suite, the architecture does not. If the plan
    differed, the scope would not be a per-model property and the four suites of
    one row could not share a module range."""
    reports = reports_for(model)
    if len(reports) < 2:
        pytest.skip("need at least two suites to compare")
    sets = {suite: ex.layer_sets(r) for suite, r in reports.items()}
    reference = sets[sorted(sets)[0]]
    for suite, other in sets.items():
        assert other["int4"] == reference["int4"], f"{model}/{suite} int4 set differs"
        assert other["int8"] == reference["int8"], f"{model}/{suite} int8 set differs"


@pytest.mark.parametrize("model", sorted(ex.DOCUMENTED_INT4))
def test_int4_and_int8_sets_are_disjoint(model):
    """An artifact covering an int8-fallback layer is rejected; the two sets must
    therefore be unambiguous."""
    sets = ex.layer_sets(reports_for(model)[sorted(reports_for(model))[0]])
    assert not (set(sets["int4"]) & set(sets["int8"]))


@pytest.mark.parametrize("model", sorted(ex.DOCUMENTED_INT4))
def test_every_covered_layer_is_inside_the_declared_scope(model):
    """A layer outside MODEL_SCOPES in the int4 set would mean the plan quantized
    something the scope does not name."""
    reports = reports_for(model)
    report = reports[sorted(reports)[0]]
    declared = set(ex.MODEL_SCOPES[model])
    by_name = {layer["name"]: layer for layer in report["layers"]}
    for name in ex.layer_sets(report)["int4"]:
        assert by_name[name]["scope"] in declared, f"{name} is in scope {by_name[name]['scope']!r}"


def test_the_exported_manifest_is_consistent(tmp_path):
    """The exporter's own checks must pass on the saved reports."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/export_scope_manifest.py"),
         "--out", str(tmp_path / "scope.json"), "--strict"],
        capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stdout + result.stderr
    doc = json.loads((tmp_path / "scope.json").read_text())
    assert doc["problems"] == []
    for model, entry in doc["models"].items():
        assert entry["n_int4"] == len(entry["artifact_must_cover"])
        assert entry["n_int8_fallback"] == len(entry["artifact_must_exclude_int8_fallback"])
        assert entry["int4_sha256"]
