"""Offline baseline pipeline: asset manifest -> smoke gate -> 6000 episodes.

Workers use reserved physical GPUs 2/3; jobs sharing a GPU are serialized.
Failed jobs halt that model pipeline;
there is no automatic experiment retry and no modification of other processes.
"""
from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT.parent
OUT = ROOT / "results/baseline_unified_v3_20260919"
ENVS = Path("/data/jyc/tools/anaconda3/envs")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
GPU = {"openvla": 2, "vla_adapter": 3, "fastwam": 3}
ENV = {"openvla": "openvla", "vla_adapter": "vlaadapter", "fastwam": "fastwam"}
REPO = {"openvla": "OpenVLA", "vla_adapter": "VLA-Adapter", "fastwam": "FastWAM"}
NAMES = ["Spatial", "Object", "Goal", "Long"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    h = hashlib.sha256()
    roots = [ROOT / "src", EMB / "OpenVLA/prismatic", EMB / "OpenVLA/experiments",
             EMB / "VLA-Adapter/prismatic", EMB / "VLA-Adapter/experiments",
             EMB / "FastWAM/src", EMB / "FastWAM/experiments", EMB / "FastWAM/configs"]
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml"):
                h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def atomic_json(path, data):
    """Durable same-directory replace.

    Three faults this used to have, all of which cost a running experiment on
    2026-09-20 when the project directory was renamed under a live controller:

    * ``with_suffix(".tmp")`` rewrites the LAST dotted component, so
      ``status.json`` became ``status.tmp`` -- fine by luck, but
      ``a.b.json`` would become ``a.tmp`` and collide.
    * a missing parent directory raised FileNotFoundError out of a telemetry
      write and killed the controller that was supervising nine finished jobs.
    * no fsync, so a machine that died between replace() and flush could leave
      a truncated status file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique, same-filesystem temp so replace() is atomic and two writers to the
    # same path cannot clobber each other's partial file.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def status_writer(path, state, log=print):
    """A save() that can fail without destroying the experiment it reports on.

    Status is telemetry; the results are the work, and they are already durable
    on disk by the time this is called. A controller must never die because it
    could not write its own progress file -- that is how a single interrupted
    job turned into two permanently missing suites.

    Write failures are counted and surfaced in the state itself, so a status
    file that stopped being updated is distinguishable from one that is lying.
    """
    failures: list[str] = []

    def save():
        state["updated"] = time.strftime("%FT%T%z")
        if failures:
            state["status_write_failures"] = len(failures)
            state["status_write_last_error"] = failures[-1]
        try:
            atomic_json(path, state)
        except Exception as exc:          # noqa: BLE001 -- deliberately broad
            failures.append(f"{type(exc).__name__}: {exc}")
            log(f"[status] WRITE FAILED ({len(failures)}x): {exc}", flush=True)

    return save


def mark_crashed(path, state, exc, log=print):
    """Never leave a dead controller's phase saying "running".

    A status file frozen at "running" with no live PID is indistinguishable from
    a job that is merely slow, which is exactly how the OpenVLA Goal suite sat
    in "running" for two weeks after its controller had died.
    """
    import traceback

    state.update(phase="crashed", error=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc(),
                 crashed_at=time.strftime("%FT%T%z"))
    try:
        atomic_json(path, state)
    except Exception as inner:            # noqa: BLE001
        log(f"[status] could not record crash: {inner}", flush=True)


def cell_tasks(directory, expect=10, episodes=50):
    """-> (present task ids, missing task ids, complete?) for one suite cell."""
    present, bad = set(), set()
    for path in Path(directory).glob("*_results.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:                 # noqa: BLE001
            continue
        if d.get("total_episodes") == episodes and not d.get("error_episodes"):
            present.add(d["task_id"])
        else:
            bad.add(d.get("task_id"))
    missing = sorted(set(range(expect)) - present)
    return sorted(present), missing, (not missing and not bad)


def environment(model):
    env = dict(os.environ)
    for k in list(env):
        if "proxy" in k.lower(): env.pop(k)
    env.update(CUDA_VISIBLE_DEVICES=str(GPU[model]), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MUJOCO_EGL_DEVICE_ID=str(GPU[model]), PYTHONPATH=str(ROOT / "src"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    return env


def export_assets():
    from embodiq.backends.prismatic_env import PrismaticEnvFactory
    factory = PrismaticEnvFactory()
    rows = []
    from libero.libero import get_libero_path
    for suite in SUITES:
        for task_id in range(factory.n_tasks(suite)):
            task = factory._suite(suite).get_task(task_id)
            states = factory.initial_states(suite, task_id, 50)
            path = factory.initial_state_path(suite, task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            rows.append(dict(suite=suite, task_id=task_id, instruction=task.language,
                             initial_state_path=str(path), initial_state_file_sha256=digest(path),
                             bddl_sha256=digest(bddl),
                             episodes=[dict(episode_idx=i, initial_state_id=i, seed=42,
                                            state_sha256=hashlib.sha256(states[i].tobytes()).hexdigest(),
                                            dtype=str(states.dtype)) for i in range(50)]))
    atomic_json(OUT / "episode_manifest.json", {"protocol": "unified.v3", "tasks": rows})


def checkpoint(model, suite):
    index = SUITES.index(suite)
    if model == "openvla":
        return Path("/data/jyc/models") / ("openvla-7b-finetuned-libero-" + ["spatial", "object", "goal", "10"][index])
    if model == "vla_adapter":
        return Path("/data/jyc/data/VLA-Adapter") / f"LIBERO-{NAMES[index]}-Pro"
    return EMB / "FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--export-assets", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.export_assets:
        export_assets(); return
    if args.background:
        with (OUT / "controller.log").open("ab", buffering=0) as log:
            p = subprocess.Popen([sys.executable, "-u", __file__], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": p.pid, "status": str(OUT / "status.json")})); return
    lock = (OUT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "status.json").exists():
        raise RuntimeError("Existing pipeline state; review it before starting a new run")
    state = dict(pid=os.getpid(), phase="preflight", jobs=[], started=time.strftime("%FT%T%z"))
    guard = threading.Lock()
    def save():
        state["updated"] = time.strftime("%FT%T%z")
        atomic_json(OUT / "status.json", state)
    save()
    try:
        versions = {}
        query = "import json,importlib.metadata as m; print(json.dumps({k:m.version(k) for k in ['torch','transformers','numpy','mujoco','robosuite','PyOpenGL','Pillow']}))"
        for model in GPU:
            python = str(ENVS / ENV[model] / "bin/python")
            versions[model] = json.loads(subprocess.check_output([python, "-c", query], text=True))
            assert versions[model]["mujoco"] == "3.3.2" and versions[model]["robosuite"] == "1.4.1"
        env = environment("fastwam"); env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([str(ENVS / "fastwam/bin/python"), __file__, "--export-assets"],
                       cwd=ROOT, env=env, check=True)
        source = source_digest()
        atomic_json(OUT / "environment_manifest.json", dict(versions=versions, source_sha256=source,
                    protocol="unified.v3", seed=42, trials=50, GPUs=GPU,
                    action_compile=False, future_video=False))

        gpu_locks = {gpu: threading.Lock() for gpu in set(GPU.values())}

        def _run_job(model, suite, phase):
            if source_digest() != source:
                raise RuntimeError("Source/config changed after pipeline freeze")
            output = OUT / phase / model
            cmd = [str(ENVS / ENV[model] / "bin/python"), "-u", "-m", "embodiq.entry.evaluate",
                   "--model", model, "--checkpoint", str(checkpoint(model, suite)),
                   "--suite", suite, "--quant", "fp16", "--protocol", "unified.v3",
                   "--gpu", str(GPU[model]), "--out", str(output),
                   "--trials", "2" if phase == "smoke" else "50"]
            if phase == "smoke": cmd += ["--tasks", "0"]
            if model == "fastwam":
                cmd += ["--dataset-stats", str(EMB / "FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")]
            row = dict(model=model, suite=suite, phase=phase, command=cmd, status="running")
            logdir = OUT / "logs"; logdir.mkdir(exist_ok=True)
            logfile = logdir / f"{phase}_{model}_{suite}.log"
            with guard:
                state["jobs"].append(row); save()
            with logfile.open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(model), stdout=log, stderr=subprocess.STDOUT)
                with guard:
                    row.update(pid=proc.pid, log=str(logfile)); save()
                rc = proc.wait()
            files = sorted((output / suite).glob("*_results.json"))
            records = [json.loads(p.read_text()) for p in files]
            expected_tasks = 1 if phase == "smoke" else 10
            expected_trials = 2 if phase == "smoke" else 50
            complete = (rc == 0 and len(records) == expected_tasks and
                        all(d["total_episodes"] == expected_trials and not d["error_episodes"] for d in records))
            with guard:
                row.update(exit_code=rc, status="complete" if complete else "failed",
                           successes=sum(d["successes"] for d in records),
                           episodes=sum(d["total_episodes"] for d in records)); save()
            return complete

        def run_job(model, suite, phase):
            with gpu_locks[GPU[model]]:
                return _run_job(model, suite, phase)

        state["phase"] = "smoke"; save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            checks = list(pool.map(lambda model: run_job(model, "libero_spatial", "smoke"), GPU))
        if not all(checks): raise RuntimeError("Smoke gate failed; full evaluation not started")

        state["phase"] = "hashing_inputs"; save()
        paths = set()
        for model in GPU:
            for suite in SUITES:
                p = checkpoint(model, suite)
                if p.is_file(): paths.add(p)
                else:
                    paths.update(f for f in p.iterdir() if f.is_file() and f.suffix in (".json", ".py", ".pt", ".safetensors", ".model", ".txt"))
        paths.update((EMB / "FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B").glob("*.pth"))
        paths.update(p for p in (EMB / "FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl").rglob("*") if p.is_file())
        paths.add(EMB / "FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
        hashes = []
        for path in sorted(paths):
            hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))
            atomic_json(OUT / "input_manifest.json", dict(complete=False, files=hashes))
        atomic_json(OUT / "input_manifest.json", dict(complete=True, files=hashes))
        state["phase"] = "full"; save()
        def model_queue(model):
            for suite in SUITES:
                if not run_job(model, suite, "full"): return False
            return True
        with ThreadPoolExecutor(max_workers=3) as pool:
            checks = list(pool.map(model_queue, GPU))
        state["phase"] = "complete" if all(checks) else "partial_failure"
        save()
    except Exception as exc:
        state.update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
