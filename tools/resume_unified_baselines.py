"""Audited continuation of the interrupted v3 baseline; no baseline code edits.

Only the specifically audited comment-only OpenVLA change is admitted. Inputs
are rehashed before execution, existing complete suites are checked and reused.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import run_unified_baselines as base

TARGET = base.EMB / "OpenVLA/prismatic/models/vlas/openvla.py"
REVISION = "c8f03f48af692657d3060c19588038c7220e9af9"
GPUS = {"openvla": 1, "vla_adapter": 2, "fastwam": 3}

# The roots the freeze digest covers. `embodiq/src` is listed first and is the
# reason the digest needs a scope argument: an RTN/AWQ/HSA run loads the
# ISOLATED tree (experiments/quant_stage*/src) via PYTHONPATH and never imports
# `src` at all, so edits there cannot reach the run -- yet they change the
# digest and would block a backfill that is byte-identical to the cells it
# completes. See audited_source(scope=...).
NATIVE_ROOTS = ["OpenVLA/prismatic", "OpenVLA/experiments",
                "VLA-Adapter/prismatic", "VLA-Adapter/experiments",
                "FastWAM/src", "FastWAM/experiments", "FastWAM/configs"]
MODEL_REPOS = {
    "OpenVLA": "c8f03f48af692657d3060c19588038c7220e9af9",
    "VLA-Adapter": "23fa0c9c159e2aa04341cdd3e924f44061311060",
    "FastWAM": "7faa71108368fbb3b6885649f112af607427a2d4",
    "LIBERO": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
}
# Tracked files already modified when the v3 baseline was frozen on 2026-09-19,
# so their current bytes are part of source_sha256 34f6b349... Anything else
# showing up as modified is real drift.
AUDITED_DIRTY = {
    "OpenVLA": {"prismatic/models/vlas/openvla.py", "experiments/robot/openvla_utils.py"},
    "VLA-Adapter": set(), "FastWAM": set(), "LIBERO": set(),
}
# Native-only digest pinned on 2026-10-04 after verifying every model repo sits
# at its frozen HEAD with no unaudited modification. It exists because the
# pre-freeze manifest recorded only the combined src+native digest, which cannot
# be split; from here on this value detects native drift on its own.
NATIVE_DIGEST = "c6e5a30655d0d188c7beacb0ef05b3fdca4031ad2cec763475d8db6809e4c090"


def repo_state() -> dict:
    """HEAD and modified tracked files for each model repo, from git itself.

    Independent of any digest we compute, and strictly stronger for the question
    being asked: has the native code moved since the freeze?
    """
    import subprocess as sp

    out = {}
    for name, revision in MODEL_REPOS.items():
        repo = base.EMB / name
        head = sp.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                               text=True).strip()
        porcelain = sp.check_output(["git", "-C", str(repo), "status", "--porcelain"],
                                    text=True).splitlines()
        modified = sorted(line[3:].strip() for line in porcelain
                          if not line.startswith("??"))
        unaudited = sorted(set(modified) - AUDITED_DIRTY[name])
        out[name] = {"head": head, "frozen_revision": revision,
                     "head_matches": head == revision,
                     "modified_tracked": modified, "unaudited_drift": unaudited}
    return out


def native_digest() -> str:
    """Digest over the model repos only, with the audited OpenVLA file replaced
    by its frozen bytes."""
    import subprocess as sp

    original = sp.check_output(["git", "-C", str(base.EMB / "OpenVLA"), "show",
                               REVISION + ":prismatic/models/vlas/openvla.py"])
    h = hashlib.sha256()
    for rel in NATIVE_ROOTS:
        for q in sorted((base.EMB / rel).rglob("*")):
            if q.is_file() and q.suffix in (".py", ".yaml"):
                h.update(str(q).encode())
                h.update(original if q == TARGET else q.read_bytes())
    return h.hexdigest()


def audited_source(scope="full"):
    """scope="full": the original check, digest over embodiq/src + native repos.
    scope="native": for runs that execute from an isolated tree -- native repos
    must be pristine (git-verified), and drift confined to embodiq/src is
    reported rather than treated as a stop."""
    original = subprocess.check_output(["git", "-C", str(base.EMB / "OpenVLA"),
                                        "show", REVISION + ":prismatic/models/vlas/openvla.py"])
    current = TARGET.read_bytes()
    if ast.dump(ast.parse(original), include_attributes=False) != ast.dump(ast.parse(current), include_attributes=False):
        raise RuntimeError("The audited OpenVLA file has executable changes")
    h = hashlib.sha256()
    roots = [base.ROOT / "src", base.EMB / "OpenVLA/prismatic", base.EMB / "OpenVLA/experiments",
             base.EMB / "VLA-Adapter/prismatic", base.EMB / "VLA-Adapter/experiments",
             base.EMB / "FastWAM/src", base.EMB / "FastWAM/experiments", base.EMB / "FastWAM/configs"]
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml"):
                h.update(str(p).encode()); h.update(original if p == TARGET else p.read_bytes())
    expected = json.loads((base.OUT / "environment_manifest.json").read_text())["source_sha256"]
    result = dict(path=str(TARGET), original_sha256=hashlib.sha256(original).hexdigest(),
                  current_sha256=hashlib.sha256(current).hexdigest(), ast_equal=True,
                  scope=scope, reconstructed_source_sha256=h.hexdigest(),
                  frozen_source_sha256=expected,
                  current_source_sha256=base.source_digest())
    if h.hexdigest() == expected:
        return result

    if scope == "full":
        raise RuntimeError("Additional source/config drift beyond the audited comment edit")

    # scope="native": the combined digest moved, so establish by git -- not by a
    # hash that embodiq/src contaminates -- that the model repos did not.
    repos = repo_state()
    result["model_repos"] = repos
    offenders = {name: info for name, info in repos.items()
                 if not info["head_matches"] or info["unaudited_drift"]}
    if offenders:
        raise RuntimeError(
            f"native model source drifted beyond the audited edit: {offenders}")
    result["native_source_sha256"] = native_digest()
    if NATIVE_DIGEST is not None and result["native_source_sha256"] != NATIVE_DIGEST:
        raise RuntimeError(
            f"native digest {result['native_source_sha256'][:12]} != pinned "
            f"{NATIVE_DIGEST[:12]}; the model repos changed")
    # Recorded, not hidden: the combined digest no longer matches the frozen
    # value, and the reason is edits under embodiq/src which this run does not
    # import. The executing tree is verified separately by the caller.
    result["main_src_drift"] = True
    result["main_src_drift_note"] = (
        "embodiq/src differs from the 2026-09-19 freeze. This run executes from "
        "an isolated experiments/quant_stage*/src tree, verified separately by "
        "its own fingerprint, so src edits cannot affect it. Model repos are at "
        "their frozen HEADs with no unaudited modification (git-verified).")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    folder = base.OUT / "recovery"
    folder.mkdir(exist_ok=True)
    if args.background:
        with (folder / "controller.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__], cwd=base.ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(json.dumps({"pid": proc.pid, "status": str(folder / "status.json")})); return
    lock = (base.OUT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "status.json").exists():
        raise RuntimeError("Recovery already has a status file; inspect it before another attempt")
    state = dict(pid=os.getpid(), phase="verifying_inputs", jobs=[], started=time.strftime("%FT%T%z"))
    guard = threading.Lock()
    def save():
        state["updated"] = time.strftime("%FT%T%z")
        base.atomic_json(folder / "status.json", state)
    save()
    try:
        audit = audited_source()
        base.atomic_json(folder / "source_audit.json", audit)
        environment = json.loads((base.OUT / "environment_manifest.json").read_text())
        for model in GPUS:
            query = "import json,importlib.metadata as m; print(json.dumps({k:m.version(k) for k in ['torch','transformers','numpy','mujoco','robosuite','PyOpenGL','Pillow']}))"
            versions = json.loads(subprocess.check_output([str(base.ENVS / base.ENV[model] / "bin/python"), "-c", query], text=True))
            if versions != environment["versions"][model]: raise RuntimeError(f"Environment changed: {model}")
        inputs = json.loads((base.OUT / "input_manifest.json").read_text())
        if not inputs["complete"]: raise RuntimeError("Original input manifest incomplete")
        state["input_files_total"] = len(inputs["files"])
        for i, row in enumerate(inputs["files"]):
            path = Path(row["path"])
            if path.stat().st_size != row["bytes"] or base.digest(path) != row["sha256"]:
                raise RuntimeError(f"Input drift: {path}")
            state["input_files_verified"] = i + 1; save()
        tasks = json.loads((base.OUT / "episode_manifest.json").read_text())["tasks"]
        expected = {(t["suite"], t["task_id"]): t for t in tasks}
        for task in tasks:
            if base.digest(task["initial_state_path"]) != task["initial_state_file_sha256"]:
                raise RuntimeError("Initial-state file changed")

        def inspect(model, suite):
            files = list((base.OUT / "full" / model / suite).glob("*_results.json"))
            rows = [json.loads(p.read_text()) for p in files]
            ids = set()
            for d in rows:
                task_id = d["task_id"]
                if task_id in ids: raise RuntimeError("Duplicate task results")
                ids.add(task_id)
                if d["policy"]["family"] != model or d["protocol"]["name"] != "unified.v3" or d["protocol"]["seed_base"] != 42:
                    raise RuntimeError("Result identity mismatch")
                if d["total_episodes"] != 50 or d["error_episodes"] or len(d["episodes"]) != 50:
                    raise RuntimeError("Incomplete/error task needs explicit review")
                if {e["episode_idx"] for e in d["episodes"]} != set(range(50)):
                    raise RuntimeError("Episode coverage mismatch")
                for e in d["episodes"]:
                    ref = expected[(suite, task_id)]["episodes"][e["episode_idx"]]
                    if e["seed"] != ref["seed"] or e["aux"]["initial_state_sha256"] != ref["state_sha256"]:
                        raise RuntimeError("Episode manifest mismatch")
            return rows, ids

        state["phase"] = "full"; save()
        def worker(model):
            for suite in base.SUITES:
                if base.source_digest() != audit["current_source_sha256"]:
                    raise RuntimeError("Further source/config change during recovery")
                rows, ids = inspect(model, suite)
                if ids == set(range(10)):
                    with guard:
                        state["jobs"].append(dict(model=model, suite=suite, status="reused",
                                                episodes=500, successes=sum(d["successes"] for d in rows))); save()
                    continue
                missing = sorted(set(range(10)) - ids)
                cmd = [str(base.ENVS / base.ENV[model] / "bin/python"), "-u", "-m", "embodiq.entry.evaluate",
                       "--model", model, "--checkpoint", str(base.checkpoint(model, suite)),
                       "--suite", suite, "--quant", "fp16", "--protocol", "unified.v3",
                       "--trials", "50", "--tasks", ",".join(map(str, missing)),
                       "--gpu", str(GPUS[model]), "--out", str(base.OUT / "full" / model)]
                if model == "fastwam": cmd += ["--dataset-stats", str(base.EMB / "FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")]
                env = base.environment(model)
                env.update(CUDA_VISIBLE_DEVICES=str(GPUS[model]), MUJOCO_EGL_DEVICE_ID=str(GPUS[model]))
                row = dict(model=model, suite=suite, status="running", command=cmd)
                with (folder / f"{model}_{suite}.log").open("w") as log:
                    p = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    with guard: row["pid"] = p.pid; state["jobs"].append(row); save()
                    rc = p.wait()
                rows, ids = inspect(model, suite)
                with guard:
                    row.update(status="complete" if rc == 0 and ids == set(range(10)) else "failed",
                               exit_code=rc, episodes=sum(d["total_episodes"] for d in rows),
                               successes=sum(d["successes"] for d in rows)); save()
                if row["status"] != "complete": return False
            return True
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(worker, GPUS))
        state["phase"] = "complete" if all(results) else "partial_failure"; save()
    except Exception as exc:
        state.update(phase="failed", error=f"{type(exc).__name__}: {exc}"); save(); raise


if __name__ == "__main__": main()
