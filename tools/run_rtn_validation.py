"""Run paired RTN suites from an isolated source tree without touching FP jobs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
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
from resume_unified_baselines import audited_source

SOURCE = base.ROOT / "experiments/quant_stage5/src"
OUT = base.ROOT / "results/stage5_rtn_full"
GPUS = {"openvla": 0, "vla_adapter": 2, "fastwam": 3}

# Live evaluator subprocesses, so a stop signal can take them down with us
# instead of leaving a 17 GiB process holding a GPU.
SHARDS: "dict[str, list[int]]" = {}
LIVE: "dict[int, subprocess.Popen]" = {}
LIVE_LOCK = threading.Lock()
STOPPING = threading.Event()


def install_stop_handlers(status_path, state, save):
    """Turn SIGTERM/SIGINT into a recorded, child-terminating shutdown.

    Resume granularity is one task: the evaluator writes a task's JSON only when
    all its episodes are done, so a stop discards the task in flight and keeps
    every task already written. The status says which.
    """
    import signal

    def handler(signum, _frame):
        if STOPPING.is_set():
            return
        STOPPING.set()
        name = signal.Signals(signum).name
        print(f"[stop] {name} received; terminating workers", flush=True)
        with LIVE_LOCK:
            children = list(LIVE.values())
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=60)
            except Exception:                      # noqa: BLE001
                child.kill()
        state.update(phase="stopped", stopped_by=name,
                     stopped_at=time.strftime("%FT%T%z"),
                     stop_note=("Tasks already written are kept and will be skipped "
                                "on the next --resume. The task in flight was "
                                "discarded: results are written per task, not per "
                                "episode."))
        for row in state.get("jobs", []):
            if row.get("status") in ("running", "waiting_for_gpu", "waiting_for_baseline"):
                row["status"] = f"stopped_while_{row['status']}"
        save()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def fingerprint():
    h = hashlib.sha256()
    for p in sorted((SOURCE / "embodiq").rglob("*.py")):
        h.update(str(p.relative_to(base.EMB)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def free_gib(gpu):
    """Free VRAM on one device, via nvidia-smi so no CUDA context is created."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True, timeout=30, check=True)
    return int(out.stdout.strip().splitlines()[0]) / 1024.0


def wait_for_gpu(gpu, need_gib, deadline, log):
    """Park until the device has room. Someone else's job is not ours to evict."""
    while True:
        try:
            free = free_gib(gpu)
        except Exception as exc:                  # noqa: BLE001
            log(f"[gpu{gpu}] query failed: {exc}", flush=True)
            free = -1.0
        if free >= need_gib:
            log(f"[gpu{gpu}] {free:.1f} GiB free >= {need_gib} GiB; starting", flush=True)
            return True
        if time.monotonic() > deadline:
            log(f"[gpu{gpu}] still only {free:.1f} GiB free at deadline", flush=True)
            return False
        if STOPPING.wait(120):
            log(f"[gpu{gpu}] stop requested while waiting for capacity", flush=True)
            return False


def baseline_ready(model, suite):
    rows = [json.loads(p.read_text()) for p in (base.OUT / "full" / model / suite).glob("*_results.json")]
    return (len(rows) == 10 and {r["task_id"] for r in rows} == set(range(10)) and
            all(r["total_episodes"] == 50 and not r["error_episodes"] and
                r["protocol"]["name"] == "unified.v3" for r in rows))


def print_status():
    """What is finished, what is in flight, and what a resume would run."""
    path = OUT / "status.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    pid = state.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    print(f"controller pid={pid} alive={alive} phase={state.get('phase')}")
    if state.get("stopped_by"):
        print(f"  stopped by {state['stopped_by']} at {state.get('stopped_at')}")
    done = pending = 0
    for model in sorted(set(state.get("models") or GPUS)):
        for suite in base.SUITES:
            present, missing, complete = base.cell_tasks(OUT / model / suite)
            mark = "complete" if complete else ("empty" if not present else "partial")
            print(f"  {model:12s} {suite:15s} {mark:9s} tasks={len(present):2d}/10"
                  + (f"  resume would run {missing}" if missing else ""))
            done += len(present)
            pending += len(missing)
    print(f"tasks done={done} pending={pending}")
    print("resume: python tools/run_rtn_validation.py --background --resume "
          "--models openvla")
    return 0


def stop_running():
    """Ask the controller to stop. It records the stop and takes its worker with it."""
    import signal

    path = OUT / "status.json"
    if not path.is_file():
        print(f"no status file at {path}")
        return 1
    state = json.loads(path.read_text())
    pid = state.get("pid")
    if not pid:
        print("status file has no pid")
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"controller {pid} is not running; phase={state.get('phase')}")
        return 1
    print(f"SIGTERM sent to controller {pid}; it terminates its worker, records "
          f"the stop, and keeps every task already written.")
    for _ in range(90):
        time.sleep(2)
        try:
            os.kill(pid, 0)
        except OSError:
            current = json.loads(path.read_text())
            print(f"stopped: phase={current.get('phase')} by={current.get('stopped_by')}")
            return 0
    print("controller still alive after 180s; inspect it before using SIGKILL")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="keep finished suite cells, run only missing/partial ones")
    parser.add_argument("--models", default="",
                        help="comma-separated subset, e.g. 'openvla' for a backfill")
    parser.add_argument("--gpu", default="",
                        help="override the GPU for a single-model run, e.g. 'openvla=3'")
    parser.add_argument("--gpus", default="",
                        help="shard a model's remaining tasks across devices, e.g. "
                             "'openvla=0,1,2,3'. Tasks are independent, so this is "
                             "pure wall-clock parallelism; one worker per device "
                             "keeps each on an exclusive GPU.")
    parser.add_argument("--min-free-gib", type=float, default=20.0,
                        help="park until the target GPU has this much free VRAM")
    parser.add_argument("--gpu-wait-hours", type=float, default=72.0)
    parser.add_argument("--stop", action="store_true",
                        help="signal the running controller to stop cleanly and exit")
    parser.add_argument("--status", action="store_true",
                        help="print what is done, running and still missing")
    args = parser.parse_args()

    if args.status:
        return print_status()
    if args.stop:
        return stop_running()
    OUT.mkdir(exist_ok=True)
    if args.background:
        child = [sys.executable, "-u", __file__] + (["--resume"] if args.resume else [])
        for flag, value in (("--models", args.models), ("--gpu", args.gpu)):
            if value:
                child += [flag, value]
        if args.gpus:
            child += ["--gpus", args.gpus]
        child += ["--min-free-gib", str(args.min_free_gib),
                  "--gpu-wait-hours", str(args.gpu_wait_hours)]
        with (OUT / "controller.log").open("ab", buffering=0) as log:
            p = subprocess.Popen(child, cwd=base.ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": p.pid, "status": str(OUT / "status.json")})); return
    models = [m for m in (args.models.split(",") if args.models else list(GPUS)) if m]
    unknown = [m for m in models if m not in GPUS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")
    for item in filter(None, args.gpu.split(",")):
        name, _, value = item.partition("=")
        if name not in GPUS:
            raise SystemExit(f"--gpu: unknown model {name!r}")
        GPUS[name] = int(value)
    for item in filter(None, args.gpus.split(";")):
        name, _, value = item.partition("=")
        if name not in GPUS:
            raise SystemExit(f"--gpus: unknown model {name!r}")
        SHARDS[name] = [int(x) for x in value.split(",") if x != ""]
        if not SHARDS[name]:
            raise SystemExit(f"--gpus: no devices for {name!r}")

    lock = (OUT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = OUT / "status.json"
    if status_path.exists() and not args.resume:
        raise RuntimeError("Existing RTN run; inspect it, then pass --resume to fill "
                           "only the missing cells")
    if args.resume and status_path.exists():
        # Keep the failed run's record instead of overwriting it; the history of
        # what died and why is evidence.
        prior = status_path.with_name(f"status.superseded.{time.strftime('%Y%m%d_%H%M%S')}.json")
        status_path.replace(prior)
        print(f"[resume] previous status preserved at {prior.name}", flush=True)
    source_hash = fingerprint()
    audit = audited_source(scope="native")
    native_hash = audit["native_source_sha256"]
    for model in models:
        files = list((base.ROOT / "results/stage5_rtn_pilot" / model / "libero_spatial").glob("*_results.json"))
        if len(files) != 1: raise RuntimeError(f"Missing pilot: {model}")
        pilot = json.loads(files[0].read_text())
        if pilot["total_episodes"] != 2 or pilot["error_episodes"] or pilot["run"]["git"]["embodiq_source_sha256"] != source_hash:
            raise RuntimeError(f"Pilot/source gate failed: {model}")
    state = dict(pid=os.getpid(), phase="running", source_sha256=source_hash,
                 native_source_sha256=native_hash, jobs=[], started=time.strftime("%FT%T%z"))
    state["source_audit"] = {k: v for k, v in audit.items() if k != "model_repos"}
    state["model_repos"] = audit.get("model_repos")
    state["isolated_source_sha256"] = source_hash
    # The claim that actually decides whether a backfilled cell is comparable
    # with the cells already finished: the tree being executed must hash to the
    # same value those cells recorded. Checked, not asserted.
    recorded = sorted({
        json.loads(p.read_text())["run"]["git"]["embodiq_source_sha256"]
        for model in models
        for p in (OUT / model).glob("*/*_results.json")
    })
    state["completed_cell_source_sha256"] = recorded
    if recorded and recorded != [source_hash]:
        raise RuntimeError(
            f"the isolated tree hashes to {source_hash[:12]} but the cells already "
            f"completed recorded {[r[:12] for r in recorded]}; a backfill from a "
            f"different source would not be comparable with them")
    state["isolated_source_matches_completed_cells"] = bool(recorded)
    state.update(resume=bool(args.resume), models=models,
                 gpus={m: GPUS[m] for m in models},
                 shards={m: SHARDS[m] for m in models if m in SHARDS},
                 min_free_gib=args.min_free_gib)
    mutex = threading.Lock()
    save = base.status_writer(status_path, state)
    install_stop_handlers(status_path, state, save)
    save()
    def worker(model):
        for suite in base.SUITES:
            present, missing, complete = base.cell_tasks(OUT / model / suite)
            if complete:
                with mutex:
                    state["jobs"].append(dict(model=model, suite=suite, status="reused",
                                              episodes=len(present) * 50))
                    save()
                print(f"[{model}/{suite}] reused: {len(present)} tasks already complete",
                      flush=True)
                continue
            row = dict(model=model, suite=suite, status="waiting_for_baseline",
                       tasks_present=present, tasks_to_run=missing)
            with mutex: state["jobs"].append(row); save()
            deadline = time.monotonic() + 48 * 3600
            while not baseline_ready(model, suite):
                if time.monotonic() > deadline:
                    with mutex: row["status"] = "baseline_wait_timeout"; save()
                    return False
                time.sleep(60)
                with mutex: save()
            if STOPPING.is_set():
                return False
            with mutex: row["status"] = "waiting_for_gpu"; save()
            deadline_gpu = time.monotonic() + args.gpu_wait_hours * 3600
            for gpu in (SHARDS.get(model) or [GPUS[model]]):
                if not wait_for_gpu(gpu, args.min_free_gib, deadline_gpu, print):
                    with mutex: row["status"] = "gpu_wait_timeout"; save()
                    return False
            if fingerprint() != source_hash: raise RuntimeError("Isolated RTN source changed")
            if audited_source(scope="native")["native_source_sha256"] != native_hash:
                raise RuntimeError("Native model/baseline source changed")

            # Shard the missing tasks across the model's GPUs. Tasks are fully
            # independent -- their own env, their own 50 initial states, and a
            # seed derived from (task_id, episode_idx) -- so which device runs a
            # task cannot change its result. One worker per GPU keeps each
            # process on an exclusive device, the same condition under which the
            # already-finished cells were measured, so the per-task latency and
            # VRAM figures stay comparable.
            devices = SHARDS.get(model) or [GPUS[model]]
            shards = [(devices[i % len(devices)], missing[i::len(devices)])
                      for i in range(min(len(devices), len(missing)))]
            shards = [(gpu, tasks) for gpu, tasks in shards if tasks]
            row["shards"] = [{"gpu": gpu, "tasks": tasks} for gpu, tasks in shards]
            with mutex: row["status"] = "launching"; save()

            procs = []
            for index, (gpu, tasks) in enumerate(shards):
                cmd = [str(base.ENVS / base.ENV[model] / "bin/python"), "-u", "-m", "embodiq.entry.evaluate",
                       "--model", model, "--checkpoint", str(base.checkpoint(model, suite)),
                       "--suite", suite, "--protocol", "unified.v3", "--quant", "rtn", "--trials", "50",
                       "--gpu", str(gpu), "--out", str(OUT / model)]
                # Only the tasks this shard owns; finished task files are left
                # untouched (one JSON per task, named by its own gpu id).
                cmd += ["--tasks", ",".join(str(i) for i in tasks)]
                if model == "fastwam": cmd += ["--dataset-stats", str(base.EMB / "FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")]
                env = base.environment(model)
                env.update(PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES=str(gpu),
                           MUJOCO_EGL_DEVICE_ID=str(gpu))
                # Include the shard index: two shards can share a device, and a
                # shared log path means two writers at independent offsets.
                suffix = f"_gpu{gpu}_s{index}" if len(shards) > 1 else ""
                entry_log = OUT / f"{model}_{suite}{suffix}.log"
                log = entry_log.open("w")
                child = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                with LIVE_LOCK: LIVE[child.pid] = child
                procs.append((gpu, tasks, child, log, entry_log.name))
                print(f"[{model}/{suite}] gpu{gpu} <- tasks {tasks} "
                      f"(pid {child.pid}, log {entry_log.name})", flush=True)
            with mutex:
                for entry, (gpu, tasks, child, _, name) in zip(row["shards"], procs):
                    entry["pid"] = child.pid
                    entry["log"] = name
                row["status"] = "running"
                save()

            codes = []
            for gpu, tasks, child, log, _name in procs:
                try:
                    codes.append(child.wait())
                finally:
                    with LIVE_LOCK: LIVE.pop(child.pid, None)
                    log.close()
            for entry, code in zip(row["shards"], codes):
                entry["exit_code"] = code
            rc = max(codes) if codes else 1
            if STOPPING.is_set():
                return False
            rows = [json.loads(p.read_text()) for p in (OUT / model / suite).glob("*_results.json")]
            now, still_missing, good = base.cell_tasks(OUT / model / suite)
            good = good and rc == 0
            with mutex:
                row.update(status="complete" if good else ("partial" if now else "failed"),
                           exit_code=rc, tasks_done=now, tasks_still_missing=still_missing,
                           episodes=sum(d["total_episodes"] for d in rows),
                           successes=sum(d["successes"] for d in rows)); save()
            if not good: return False
        return True
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
            checks = list(pool.map(worker, models))
        state["phase"] = "complete" if all(checks) else "partial_failure"; save()
    except BaseException as exc:
        # BaseException so a KeyboardInterrupt or SystemExit also leaves a
        # truthful status file rather than a frozen "running". A deliberate stop
        # already recorded itself and raised SystemExit to unwind; overwriting
        # that with "crashed" would report an intentional stop as a failure.
        if not STOPPING.is_set():
            base.mark_crashed(status_path, state, exc)
        raise


if __name__ == "__main__": main()
