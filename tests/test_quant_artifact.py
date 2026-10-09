"""The AWQ/HSA artifact contract, pinned on a synthetic tree.

Runs with no checkpoint and no CUDA: every check in quant.artifact.bind happens
before a single module is replaced, so the whole contract surface is testable on
CPU. The packed-kernel side of the swap is covered by the GPU gate tests.

The tests are grouped by the failure each one makes impossible, because that is
what the contract is for -- 1.2 and 1.3 deliver the artifacts, and this
evaluation entry has to be able to reject a wrong one rather than produce a
complete, plausible, mislabelled set of results.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from embodiq.quant.api import make_quant_apply
from embodiq.quant.artifact import (
    FORMAT,
    ArtifactError,
    QuantArtifact,
    tensor_sha256,
)
from embodiq.quant.spec import HSASpec, QuantSpec
from embodiq.quant.walker import plan_quantization

GROUP = 128


def build_openvla_llm(layers: int = 2) -> dict:
    """A minimal `openvla` tree: only language_model.model.layers.* is in scope."""
    hf = nn.Module()
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.layers = nn.ModuleList()
    for _ in range(layers):
        blk = nn.Module()
        blk.self_attn = nn.Module()
        blk.mlp = nn.Module()
        blk.self_attn.q_proj = nn.Linear(1024, 1024)
        blk.self_attn.o_proj = nn.Linear(1024, 1024)
        blk.mlp.up_proj = nn.Linear(1024, 2560)
        blk.mlp.down_proj = nn.Linear(2560, 1024)
        lm.model.layers.append(blk)
    lm.lm_head = nn.Linear(1024, 32064)
    hf.language_model = lm
    return {"hf": hf}


def plan(roots, spec, scopes=frozenset({"llm"})):
    return plan_quantization(roots, "openvla", spec, enabled_scopes=set(scopes))


def int4_names(report) -> list[str]:
    return [r.name for r in report.layers if r.decision == "int4"]


def module_by_name(roots, name) -> nn.Linear:
    return next(iter(roots.values())).get_submodule(name)


def layer_row(roots, name, **extra) -> dict:
    mod = module_by_name(roots, name)
    row = {
        "weight_sha256": tensor_sha256(mod.weight),
        "bias_sha256": tensor_sha256(mod.bias),
    }
    row.update(extra)
    return row


def awq_payload(roots, names, **over) -> dict:
    payload = {
        "format": FORMAT,
        "method": "awq",
        "implementation": "awq_block_v1",
        "implementation_version": "1.0",
        "producer_code_sha256": "a" * 64,
        "model_family": "openvla",
        "suite": "libero_spatial",
        "w_bit": 4,
        "group_size": GROUP,
        "source_checkpoint_sha256": "b" * 64,
        "calibration": {"manifest_sha256": "c" * 64, "n_calibration": 1280,
                        "n_development": 320, "source": "libero training demos"},
        "scope": ["llm"],
        "layers": {
            n: layer_row(roots, n,
                         input_scale=torch.rand(module_by_name(roots, n).in_features) + 0.5)
            for n in names
        },
    }
    payload.update(over)
    return payload


def hsa_payload(roots, names, k=8, **over) -> dict:
    payload = awq_payload(roots, names, method="hsa",
                          implementation="hsa_local_residual_v1")
    for n in names:
        mod = module_by_name(roots, n)
        payload["layers"][n]["residual_idx"] = torch.arange(k, dtype=torch.int32)
        payload["layers"][n]["residual_val"] = torch.randn(mod.out_features, k)
    payload.update(over)
    return payload


def save(tmp_path, payload, name="artifact.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


@pytest.fixture
def roots():
    torch.manual_seed(0)
    return build_openvla_llm()


# -- the happy path ---------------------------------------------------------

def test_valid_awq_artifact_binds_and_accounts(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    assert names, "the synthetic tree must produce int4 layers"
    art = QuantArtifact.load(save(tmp_path, awq_payload(roots, names)))
    bound = art.bind(roots, report, spec, "openvla", scopes={"llm"})

    assert set(bound.input_scales) == set(names)
    assert not bound.residuals
    assert bound.meta["n_layers"] == len(names)
    # Scale bytes are counted against the method's budget, not ignored.
    expected = sum(module_by_name(roots, n).in_features * 2 for n in names)
    assert bound.meta["input_scale_bytes"] == expected
    assert bound.meta["method_overhead_bytes"] == expected
    assert bound.meta["artifact_sha256"]


def test_valid_hsa_artifact_accounts_residual_and_index(roots, tmp_path):
    spec = QuantSpec(method="hsa", hsa=HSASpec(rho=0.01))
    report = plan(roots, spec)
    names = int4_names(report)
    art = QuantArtifact.load(save(tmp_path, hsa_payload(roots, names, k=8)))
    bound = art.bind(roots, report, spec, "openvla", scopes={"llm"})

    assert set(bound.residuals) == set(names)
    units = sum(8 for _ in names)
    assert bound.meta["residual_units"] == units
    assert bound.meta["index_bytes"] == units * 4
    assert bound.meta["residual_bytes"] == sum(
        module_by_name(roots, n).out_features * 8 * 2 for n in names)
    # The per-layer record carries it too, so it reaches the memory column.
    recs = {r.name: r for r in report.layers if r.name in set(names)}
    assert all(recs[n].residual_units == 8 for n in names)
    assert all(recs[n].index_bytes == 32 for n in names)


# -- a relabelled run is impossible ----------------------------------------

def test_awq_without_artifact_is_refused():
    with pytest.raises(ValueError, match="requires a quantization artifact"):
        make_quant_apply(QuantSpec(method="awq"), scopes={"llm"})


def test_hsa_without_artifact_is_refused():
    with pytest.raises(ValueError, match="requires a quantization artifact"):
        make_quant_apply(QuantSpec(method="hsa", hsa=HSASpec()), scopes={"llm"})


def test_rtn_rejects_an_artifact(tmp_path):
    with pytest.raises(ValueError, match="takes no artifact"):
        make_quant_apply(QuantSpec(method="rtn"), scopes={"llm"},
                         artifact=tmp_path / "whatever.pt")


def test_awq_artifact_carrying_residuals_is_rejected_as_hsa(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    payload = hsa_payload(roots, names)
    payload["method"] = "awq"            # residuals present, labelled AWQ
    payload["implementation"] = "awq_block_v1"
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="must not carry high-precision"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_hsa_artifact_without_residuals_is_rejected_as_awq(roots, tmp_path):
    spec = QuantSpec(method="hsa", hsa=HSASpec())
    report = plan(roots, spec)
    names = int4_names(report)
    payload = awq_payload(roots, names)
    payload["method"] = "hsa"            # scales only, labelled HSA
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="no layer carries a residual"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_artifact_method_must_match_the_run_spec(roots, tmp_path):
    spec = QuantSpec(method="hsa", hsa=HSASpec())
    report = plan(roots, spec)
    art = QuantArtifact.load(save(tmp_path, awq_payload(roots, int4_names(report))))
    with pytest.raises(ArtifactError, match="refusing to run one method under"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


# -- a different module range is impossible --------------------------------

def test_scope_mismatch_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    payload = awq_payload(roots, int4_names(report), scope=["llm", "vision_dino"])
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="must share one per-model module"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_artifact_missing_a_planned_layer_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    art = QuantArtifact.load(save(tmp_path, awq_payload(roots, names[:-1])))
    with pytest.raises(ArtifactError, match="does not match the plan's int4"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_artifact_with_an_out_of_plan_layer_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    payload = awq_payload(roots, names)
    # lm_head is hard-skipped: action correctness is argmax over its logits.
    payload["layers"]["language_model.lm_head"] = layer_row(
        roots, "language_model.lm_head", input_scale=torch.ones(1024))
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="does not match the plan's int4"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


# -- a different checkpoint is impossible ----------------------------------

def test_weight_hash_mismatch_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    path = save(tmp_path, awq_payload(roots, names))
    art = QuantArtifact.load(path)
    with torch.no_grad():                       # simulate a different checkpoint
        module_by_name(roots, names[0]).weight.add_(1.0)
    with pytest.raises(ArtifactError, match="different checkpoint"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_model_family_mismatch_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    payload = awq_payload(roots, int4_names(report), model_family="fastwam")
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="artifact is for 'fastwam'"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_bit_width_mismatch_is_rejected(roots, tmp_path):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    payload = awq_payload(roots, int4_names(report), w_bit=8)
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match="artifact is w8"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


# -- malformed metadata ----------------------------------------------------

def test_unknown_format_is_rejected(roots, tmp_path):
    payload = awq_payload(roots, ["language_model.model.layers.0.self_attn.q_proj"])
    payload["format"] = "embodiq/awq-block-pilot/1"
    with pytest.raises(ArtifactError, match="unsupported artifact format"):
        QuantArtifact.load(save(tmp_path, payload))


def test_missing_required_key_is_rejected(roots, tmp_path):
    payload = awq_payload(roots, ["language_model.model.layers.0.self_attn.q_proj"])
    del payload["producer_code_sha256"]
    with pytest.raises(ArtifactError, match="missing required keys"):
        QuantArtifact.load(save(tmp_path, payload))


def test_artifact_without_calibration_manifest_is_rejected(roots, tmp_path):
    payload = awq_payload(roots, ["language_model.model.layers.0.self_attn.q_proj"])
    payload["calibration"] = {"n_calibration": 1280}
    with pytest.raises(ArtifactError, match="calibration.manifest_sha256 is required"):
        QuantArtifact.load(save(tmp_path, payload))


def test_empty_layer_set_is_rejected(roots, tmp_path):
    payload = awq_payload(roots, [])
    with pytest.raises(ArtifactError, match="contains no layers"):
        QuantArtifact.load(save(tmp_path, payload))


def test_fp16_or_rtn_method_in_an_artifact_is_rejected(roots, tmp_path):
    payload = awq_payload(roots, ["language_model.model.layers.0.self_attn.q_proj"])
    payload["method"] = "rtn"
    with pytest.raises(ArtifactError, match="need no artifact"):
        QuantArtifact.load(save(tmp_path, payload))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        QuantArtifact.load(tmp_path / "absent.pt")


# -- malformed tensors -----------------------------------------------------

@pytest.mark.parametrize("mutate,match", [
    (lambda m, r: r.update(input_scale=torch.ones(7)), "input_scale shape"),
    (lambda m, r: r.update(input_scale=torch.zeros(m.in_features)), "strictly positive"),
    (lambda m, r: r.update(input_scale=torch.full((m.in_features,), float("nan"))),
     "non-finite"),
    (lambda m, r: r.update(clip_max=torch.ones(3, 3)), "clip_max shape"),
    (lambda m, r: r.update(
        clip_max=torch.full((m.out_features, m.in_features // GROUP), -1.0)),
     r"clip_max must be >= 0"),
])
def test_malformed_awq_tensor_is_rejected(roots, tmp_path, mutate, match):
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    payload = awq_payload(roots, names)
    mutate(module_by_name(roots, names[0]), payload["layers"][names[0]])
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match=match):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


@pytest.mark.parametrize("mutate,match", [
    (lambda m, r: r.update(residual_idx=torch.tensor([0, 0, 1, 2], dtype=torch.int32),
                           residual_val=torch.randn(m.out_features, 4)),
     "duplicate channels"),
    (lambda m, r: r.update(residual_idx=torch.tensor([m.in_features], dtype=torch.int32),
                           residual_val=torch.randn(m.out_features, 1)),
     "out of range"),
    (lambda m, r: r.update(residual_idx=torch.zeros(4), # float, not integer
                           residual_val=torch.randn(m.out_features, 4)),
     "must be an integer tensor"),
    (lambda m, r: r.update(residual_val=torch.randn(m.out_features, 99)),
     "residual_val shape"),
    (lambda m, r: r.pop("residual_val"), "only one of residual_idx"),
])
def test_malformed_hsa_tensor_is_rejected(roots, tmp_path, mutate, match):
    spec = QuantSpec(method="hsa", hsa=HSASpec())
    report = plan(roots, spec)
    names = int4_names(report)
    payload = hsa_payload(roots, names)
    mutate(module_by_name(roots, names[0]), payload["layers"][names[0]])
    art = QuantArtifact.load(save(tmp_path, payload))
    with pytest.raises(ArtifactError, match=match):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


def test_residual_over_the_per_layer_budget_is_rejected(roots, tmp_path):
    """rho is a budget, not a suggestion: residual bytes are real VRAM."""
    spec = QuantSpec(method="hsa", hsa=HSASpec(rho=0.01, max_per_layer_rho=0.02))
    report = plan(roots, spec)
    names = int4_names(report)
    # 1024 * 0.02 = 20 channels allowed; ask for 64.
    art = QuantArtifact.load(save(tmp_path, hsa_payload(roots, names, k=64)))
    with pytest.raises(ArtifactError, match="above max_per_layer_rho"):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})


# -- nothing is half-applied -----------------------------------------------

def test_rejected_artifact_leaves_the_model_untouched(roots, tmp_path):
    """The one failure that would be invisible in the results: a partial swap."""
    spec = QuantSpec(method="awq")
    report = plan(roots, spec)
    names = int4_names(report)
    payload = awq_payload(roots, names)
    # Last layer malformed: everything before it validated fine.
    payload["layers"][names[-1]]["input_scale"] = torch.ones(3)
    art = QuantArtifact.load(save(tmp_path, payload))
    before = {n: module_by_name(roots, n).weight.clone() for n in names}
    with pytest.raises(ArtifactError):
        art.bind(roots, report, spec, "openvla", scopes={"llm"})
    for n in names:
        mod = module_by_name(roots, n)
        assert isinstance(mod, nn.Linear), f"{n} was replaced despite a rejection"
        assert torch.equal(mod.weight, before[n])


def test_metadata_reaches_the_result_json(roots, tmp_path):
    """Every field a reproducible AWQ/HSA row needs must survive into quant_meta."""
    spec = QuantSpec(method="hsa", hsa=HSASpec())
    report = plan(roots, spec)
    art = QuantArtifact.load(save(tmp_path, hsa_payload(roots, int4_names(report))))
    meta = art.bind(roots, report, spec, "openvla", scopes={"llm"}).meta
    for key in ("implementation", "implementation_version", "producer_code_sha256",
                "source_checkpoint_sha256", "calibration", "scope", "artifact_sha256",
                "w_bit", "group_size", "method_overhead_bytes"):
        assert meta.get(key), f"{key} missing from artifact metadata"
