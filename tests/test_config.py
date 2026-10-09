import json
from pathlib import Path

import pytest

from embodiq.config import load_config, validate_config, validate_file


ROOT = Path(__file__).resolve().parents[1]


def test_stage_configs_validate():
    for path in sorted((ROOT / "configs").glob("stage_*.json")):
        validate_file(path)


def test_disabled_hsa_is_reserved_only():
    cfg = load_config(ROOT / "configs/stage_full_pending_hsa.json")
    assert "hsa" in cfg["enabled_methods"]
    assert cfg["quantization"]["hsa"]["enabled"] is False
    assert cfg["quantization"]["hsa"]["status"] == "pending_external_implementation"
    assert not validate_config(cfg)


def test_bad_scope_is_rejected():
    cfg = load_config(ROOT / "configs/stage_awq.json")
    cfg["models"]["fastwam"]["scopes"] = ["llm"]
    assert any("models.fastwam.scopes" in e for e in validate_config(cfg))


def test_yaml_requires_optional_dependency(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text("schema: embodiq/experiment-config/1\n")
    # This test only checks the suffix dispatch when yaml is available in the
    # test environment; JSON remains the dependency-free interchange format.
    try:
        import yaml  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="PyYAML"):
            load_config(p)


def test_plan_skips_pending_hsa():
    from embodiq.config import build_plan, load_config
    cfg = load_config(ROOT / "configs/stage_full_pending_hsa.json")
    jobs = build_plan(cfg)
    assert len(jobs) == 36
    assert {j["quant"] for j in jobs} == {"fp16", "rtn", "awq"}


def test_managed_run_plan_is_durable(tmp_path):
    from embodiq.runner import create_run, load_run, summarize_run
    run_dir, state = create_run(ROOT / "configs/stage_awq.json", tmp_path)
    saved = load_run(run_dir)
    assert saved["schema"] == "embodiq/managed-run/1"
    assert len(saved["jobs"]) == 36
    assert summarize_run(saved)["planned"] == 36
    assert (run_dir / "run.json").is_file()


def test_runner_preflight_blocks_missing_awq_artifact(tmp_path):
    from embodiq.runner import create_run, execute_job, load_run
    run_dir, state = create_run(ROOT / "configs/stage_awq.json", tmp_path)
    awq_job = next(j for j in state["jobs"] if j["quant"] == "awq")
    code = execute_job(run_dir, awq_job["job_id"], gpu=0)
    assert code == 2
    saved = load_run(run_dir)
    row = next(j for j in saved["jobs"] if j["job_id"] == awq_job["job_id"])
    assert row["status"] == "blocked"


@pytest.mark.parametrize('method',['awq','hsa'])
@pytest.mark.parametrize('explicit',[True,False])
def test_artifact_method_command_and_preflight(tmp_path,method,explicit):
    from embodiq.runner import command_for_job, preflight_job
    cfg=load_config(ROOT/'configs/stage_fp16_rtn.json')
    from embodiq.config import build_plan
    job=next(j for j in build_plan(cfg) if j['model']=='openvla')
    job.update(quant=method,hsa_rho=.02,artifact_root=str(tmp_path),artifact=None)
    artifact=tmp_path/'openvla'/'libero_spatial.pt'
    if explicit:
        artifact=tmp_path/'explicit.pt';job['artifact']=str(artifact)
    assert f'{method}_artifact_missing' in preflight_job(job)
    with pytest.raises(ValueError,match='existing artifact'):
        command_for_job(job,output=tmp_path/'out',gpu=3)
    artifact.parent.mkdir(exist_ok=True);artifact.write_bytes(b'preflight-existence-only')
    assert not any('artifact' in r for r in preflight_job(job))
    cmd,env=command_for_job(job,output=tmp_path/'out',gpu=3)
    assert cmd[cmd.index('--quant-artifact')+1]==str(artifact)
    assert env['CUDA_VISIBLE_DEVICES']=='3'
    if method=='hsa':assert cmd[cmd.index('--hsa-rho')+1]=='0.02'


def test_hsa_enabled_plan_preserves_budget_and_artifact(tmp_path):
    from embodiq.config import build_plan
    cfg=load_config(ROOT/'configs/stage_fp16_rtn.json')
    cfg['enabled_methods']=['hsa']
    cfg['quantization']['hsa'].update(enabled=True,status='ready',artifact_root=str(tmp_path),rho=.02)
    jobs=build_plan(cfg)
    assert len(jobs)==12 and all(j['hsa_rho']==.02 for j in jobs)
    cfg['quantization']['hsa']['rho']=.2
    assert any('rho' in e for e in validate_config(cfg))
