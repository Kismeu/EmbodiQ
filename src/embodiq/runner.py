"""Managed experiment planning and durable run state.

This layer deliberately separates planning from execution.  It gives the CLI a
stable run ID, immutable config snapshot and per-job state before model code is
started.  Existing model-specific evaluators remain the execution backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from embodiq.config import build_plan, load_config, validate_config

SCHEMA = "embodiq/managed-run/1"


def _atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _run_id(config_path: Path, config: dict, jobs: list[dict]) -> str:
    payload = json.dumps({"config": config, "jobs": jobs}, sort_keys=True).encode()
    return time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha256(payload).hexdigest()[:10]


def create_run(config_path: str | Path, out_root: str | Path | None = None) -> tuple[Path, dict]:
    path = Path(config_path).resolve()
    config = load_config(path)
    errors = validate_config(config)
    if errors:
        raise ValueError("invalid config: " + "; ".join(errors))
    jobs = build_plan(config)
    root = Path(out_root or config["outputs"]["root"])
    run_id = _run_id(path, config, jobs)
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    rows = []
    for i, job in enumerate(jobs):
        row = dict(job)
        row.update(job_id=f"job_{i:04d}", status="planned", attempts=0)
        rows.append(row)
    state = {
        "schema": SCHEMA, "run_id": run_id, "created": time.strftime("%FT%T%z"),
        "updated": time.strftime("%FT%T%z"), "phase": "planned",
        "config_path": str(path), "config": config, "jobs": rows,
        "summary": {"planned": len(rows), "running": 0, "complete": 0, "blocked": 0, "failed": 0},
    }
    _atomic(run_dir / "run.json", state)
    return run_dir, state


def load_run(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir(): p = p / "run.json"
    if not p.is_file(): raise FileNotFoundError(p)
    return json.loads(p.read_text())


def save_run(path: str | Path, state: dict) -> None:
    p = Path(path)
    if p.is_dir(): p = p / "run.json"
    state["updated"] = time.strftime("%FT%T%z")
    _atomic(p, state)


def summarize_run(state: dict) -> dict:
    counts = {k: 0 for k in ("planned", "running", "complete", "blocked", "failed")}
    for job in state.get("jobs", []): counts[job.get("status", "planned")] = counts.get(job.get("status", "planned"), 0) + 1
    state["summary"] = counts
    return counts


SUITE_CKPT = {
    "libero_spatial": "LIBERO-Spatial-Pro",
    "libero_object": "LIBERO-Object-Pro",
    "libero_goal": "LIBERO-Goal-Pro",
    "libero_10": "LIBERO-Long-Pro",
}
OPENVLA_CKPT = {
    "libero_spatial": "openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla-7b-finetuned-libero-10",
}
ENV_NAME = {"openvla": "openvla", "vla_adapter": "vlaadapter", "fastwam": "fastwam"}


def _python_for_job(job: dict) -> str:
    """Choose the model-specific conda interpreter, with an explicit override."""
    if job.get('python'):
        return job['python']
    model = job["model"]
    key = "EMBODIQ_PYTHON_" + model.upper().replace("-", "_")
    override = os.environ.get(key)
    if override:
        return override
    return f"/data/jyc/tools/anaconda3/envs/{ENV_NAME[model]}/bin/python"


def _resolve_checkpoint(job: dict) -> Path | None:
    if job.get("checkpoint"):
        return Path(job["checkpoint"])
    root = job.get("checkpoint_root")
    if not root or "${" in root:
        return None
    names = OPENVLA_CKPT if job.get("model") == "openvla" else SUITE_CKPT
    name = names.get(job["suite"])
    return Path(root) / name if name else None


def preflight_job(job: dict) -> list[str]:
    """Return blocking reasons without importing a model repository."""
    reasons: list[str] = []
    python_bin = Path(_python_for_job(job))
    if not python_bin.is_file():
        reasons.append(f"python_missing:{python_bin}")
    repo = job.get("repo_root")
    if repo and "${" not in repo and not Path(repo).is_dir():
        reasons.append(f"repo_root_missing:{repo}")
    checkpoint = _resolve_checkpoint(job)
    if checkpoint is None:
        reasons.append("checkpoint_unresolved")
    elif not checkpoint.exists():
        reasons.append(f"checkpoint_missing:{checkpoint}")
    if job.get("quant") in {"awq", "hsa"}:
        artifact = _resolve_artifact(job)
        if artifact is None:
            reasons.append(f"{job['quant']}_artifact_unresolved")
        elif not artifact.is_file():
            reasons.append(f"{job['quant']}_artifact_missing")
    if job.get("model") == "fastwam":
        stats = job.get("dataset_stats")
        if not stats or not Path(stats).is_file():
            reasons.append("dataset_stats_missing")
    return reasons


def _resolve_artifact(job: dict) -> Path | None:
    explicit = job.get("artifact")
    if explicit:
        return None if "${" in str(explicit) else Path(explicit)
    root = job.get("artifact_root")
    if not root or "${" in str(root):
        return None
    candidates = [Path(root) / job["model"] / f"{job['suite']}.pt",
                  Path(root) / f"{job['model']}_{job['suite']}.pt"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def command_for_job(job: dict, *, output: Path, gpu: int) -> tuple[list[str], dict]:
    """Build the isolated environment command for one evaluator job."""
    ckpt = _resolve_checkpoint(job)
    cmd = [_python_for_job(job), "-m", "embodiq.entry.evaluate",
           "--model", job["model"], "--suite", job["suite"],
           "--protocol", job["protocol"], "--quant", job["quant"],
           "--trials", str(job["trials"]), "--seed", str(job["seed"]),
           "--gpu", "0", "--out", str(output)]
    if ckpt is not None:
        cmd += ["--checkpoint", str(ckpt)]
    if job.get("repo_root"):
        cmd += ["--repo-root", str(job["repo_root"])]
    if job.get("dataset_stats"):
        cmd += ["--dataset-stats", str(job["dataset_stats"])]
    if job["model"] == "vla_adapter" and job.get("checkpoint_root"):
        cmd += ["--checkpoint-root", str(job["checkpoint_root"])]
    if job["model"] == "fastwam":
        cmd += ["--fastwam-mode", job.get("fastwam_mode", "first_frame")]
    if job.get("task_ids") is not None:
        cmd += ["--tasks", ",".join(map(str, job["task_ids"]))]
    cmd += ["--checkpoint-every", str(job.get("checkpoint_every", 1))]
    if job.get("save_videos"):
        cmd += ["--save-videos"]
    if job["quant"] in {"awq", "hsa"}:
        artifact = _resolve_artifact(job)
        if artifact is None or not artifact.is_file():
            raise ValueError(f"{job['quant']} requires an existing artifact")
        cmd += ["--quant-artifact", str(artifact)]
        if job["quant"] == "hsa":
            cmd += ["--hsa-rho", str(job.get("hsa_rho", 0.01))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl",
               PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), QVLA_TF_CPU_ONLY="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    root = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    return cmd, env


def execute_job(run_path: str | Path, job_id: str, *, gpu: int = 0, resume: bool = False) -> int:
    import subprocess
    run_dir = Path(run_path)
    state = load_run(run_dir)
    job = next((j for j in state["jobs"] if j["job_id"] == job_id), None)
    if job is None: raise KeyError(f"unknown job: {job_id}")
    if state.get("phase") == "stopped":
        if not resume:
            return 3
        state["phase"] = "running"
        state.pop("stop_requested", None)
        save_run(run_dir, state)
    if job.get("status") == "complete": return 0
    reasons = preflight_job(job)
    if reasons:
        job.update(status="blocked", blocked_reasons=reasons)
        summarize_run(state); save_run(run_dir, state)
        return 2
    job["status"] = "running"; job["attempts"] = int(job.get("attempts", 0)) + 1
    summarize_run(state); save_run(run_dir, state)
    output = run_dir / "results" / job_id
    cmd, env = command_for_job(job, output=output, gpu=gpu)
    log = run_dir / "logs" / f"{job_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        job.update(status="complete", result_dir=str(output), log=str(log))
    else:
        job.update(status="failed", returncode=result.returncode, result_dir=str(output), log=str(log))
    state["phase"] = "complete" if all(j["status"] in {"complete", "blocked"} for j in state["jobs"]) else "running"
    summarize_run(state); save_run(run_dir, state)
    return result.returncode


def stop_run(run_path: str | Path) -> dict:
    """Mark a durable run stopped; a running child process is left untouched."""
    state = load_run(run_path)
    state["phase"] = "stopped"
    state["stop_requested"] = True
    summarize_run(state)
    save_run(run_path, state)
    return state
