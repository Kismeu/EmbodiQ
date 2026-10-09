"""Fixed-input resource measurement (protocol section 8), one configuration per run.

Why this exists separately from the closed-loop runs: a rollout's latency and
VRAM depend on which episodes it happened to get, how many steps they ran, and
whether anything else was on the device. Those numbers are reported per cell and
are fine for "did this configuration work", but they are not a deployment claim.
This tool removes every one of those degrees of freedom:

* **The same inputs, in the same order, for every configuration.** 40 tasks x 5
  observations = 200 fixed inputs, listed with their sha256 in a manifest that is
  written once and reused. A configuration that saw different inputs is not
  comparable, so the manifest hash is recorded in the output and checked on load.
* **Warmup excluded, explicitly.** 20 warmup queries per checkpoint, then 3
  measured passes over its inputs. Cold-start and first-query cost are reported
  as their own numbers rather than averaged into the steady state.
* **batch = 1, one request at a time, CUDA-synced around each query.** Latency is
  measured from a complete raw observation to an executable action chunk --
  preprocessing, model, and post-processing included, simulator excluded.
* **Exclusive GPU, or the result is marked contended and stays out of the main
  table.** Checked before and after; `--allow-contention` records the fact
  instead of refusing, for a diagnostic run.

Two numbers, not one, for each of the two metrics:

* latency: `query_s` is comparable only within a model; `per_query_action_s`
  (query / executed actions) is the cross-model figure, because the three models
  query at 1 : 1/8 : 1/10 density.
* memory: the load phase and the steady state are separate -- quantization runs
  in-process after the bf16 checkpoint is resident, so the process peak during
  load is bounded by the ORIGINAL precision. MemoryProbe releases the allocator
  pool after quantization and reports both.

Usage:
    # once, offline: build the shared input manifest
    python tools/run_resource_bench.py --make-inputs

    # then, per configuration, on an idle GPU
    python tools/run_resource_bench.py --model vla_adapter --quant rtn --gpu 2
    python tools/run_resource_bench.py --model fastwam --quant hsa \\
        --quant-artifact /path/to/hsa_fastwam.pt --gpu 2
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_unified_baselines as base  # noqa: E402

OBS_ROOT = ROOT / "calib/observations_training_v1"
INPUTS = ROOT / "results/resource_bench/inputs_v1.json"
OUT_ROOT = ROOT / "results/resource_bench"
OBS_PER_TASK = 5
WARMUP_QUERIES = 20
MEASURED_PASSES = 3
NVML_SAMPLER_HZ = 100.0  # 10 ms, the protocol's candidate interval

# Observations are re-rendered from the official training demonstrations, which
# are successful by construction. Protocol section 8 asks for inputs that are not
# drawn only from successful trajectories, so this source does not satisfy that
# clause and the limitation travels with every result produced from it. It is
# sound for what is being measured here -- these models' per-query cost is
# input-independent by construction (OpenVLA emits exactly 7 tokens, VLA-Adapter
# runs one forward, FastWAM takes a fixed 10 Euler steps) -- but a success-rate
# claim must never be built on it.
INPUT_SOURCE_CAVEAT = (
    "观测由官方训练演示重渲染，训练演示按构造均为成功轨迹，"
    "不满足协议 §8 “不得只挑成功轨迹”。本工具只测资源："
    "三模型的单次查询代价按构造与输入内容无关"
    "（OpenVLA 固定 7 个 token、VLA-Adapter 单次前向、FastWAM 固定 10 步 Euler），"
    "故该来源对显存/延迟测量成立；但不得用于任何成功率结论。"
)


# -- the shared input manifest ---------------------------------------------

def make_inputs(out: Path = INPUTS) -> dict:
    """Pick 5 observations per task, deterministically, and hash every one.

    Taken from the development split so the calibration set that AWQ/HSA tune on
    is not also the set their cost is measured on. Selection is by sorted
    filename, so the manifest is reproducible without a random seed.
    """
    tasks = []
    for suite in base.SUITES:
        for task_id in range(10):
            manifest_path = OBS_ROOT / suite / f"task{task_id}" / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"missing {manifest_path}; run tools/prepare_all_calibration.py first")
            manifest = json.loads(manifest_path.read_text())
            if manifest["suite"] != suite or manifest["task_id"] != task_id:
                raise ValueError(f"{manifest_path}: suite/task mismatch")
            dev = sorted((o for o in manifest["observations"]
                          if o["split"] == "development"),
                         key=lambda o: o["file"])
            if len(dev) < OBS_PER_TASK:
                raise ValueError(
                    f"{manifest_path}: {len(dev)} development observations, "
                    f"need {OBS_PER_TASK}")
            chosen = dev[:OBS_PER_TASK]
            rows = []
            for obs in chosen:
                path = OBS_ROOT / suite / f"task{task_id}" / obs["file"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != obs["sha256"]:
                    raise ValueError(f"{path}: observation drifted from its manifest")
                rows.append({
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "demo_id": obs["demo_id"],
                    "frame_id": obs["frame_id"],
                    "split": obs["split"],
                })
            tasks.append({
                "suite": suite,
                "task_id": task_id,
                "instruction": manifest["instruction"],
                "source_sha256": manifest["source_sha256"],
                "observations": rows,
            })
    doc = {
        "schema": "embodiq/resource_bench_inputs/1",
        "created": time.strftime("%FT%T%z"),
        "source": "calib/observations_training_v1 development split",
        "selection": "sorted by filename, first 5 per task; no randomness",
        "obs_per_task": OBS_PER_TASK,
        "n_tasks": len(tasks),
        "n_observations": sum(len(t["observations"]) for t in tasks),
        "caveat": INPUT_SOURCE_CAVEAT,
        "tasks": tasks,
    }
    doc["manifest_sha256"] = hashlib.sha256(
        json.dumps(doc["tasks"], sort_keys=True).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(out, doc)
    return doc


def load_inputs(path: Path = INPUTS) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run --make-inputs first")
    doc = json.loads(path.read_text())
    recomputed = hashlib.sha256(
        json.dumps(doc["tasks"], sort_keys=True).encode()).hexdigest()
    if recomputed != doc.get("manifest_sha256"):
        raise ValueError(f"{path}: manifest hash mismatch; the input list was edited")
    return doc


# -- contention ------------------------------------------------------------

def other_processes(gpu: int) -> list[dict]:
    """Processes on the device that are not us. Protocol section 8: resource
    numbers from a shared GPU do not enter the main table."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:                      # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    mine = {os.getpid()}
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, mem = (x.strip() for x in line.split(","))
        if int(pid) not in mine:
            rows.append({"pid": int(pid), "used_mib": int(mem)})
    return rows


# -- the measurement -------------------------------------------------------

def bench_suite(model: str, suite: str, quant: str, artifact, gpu: int,
                task_rows: list[dict], fastwam_mode: str, *, job=None,
                warmup_queries=WARMUP_QUERIES, measured_passes=MEASURED_PASSES) -> dict:
    """Load one checkpoint, warm it up, then time three passes over its inputs."""
    import numpy as np
    import torch

    from embodiq.core.memory import MemoryProbe
    from embodiq.core.rollout import build_observation
    from embodiq.core.timing import LatencyAccumulator

    # Imported here so a --make-inputs run needs no model repo on the path.
    repo = Path(job['repo_root']) if job else base.EMB / base.REPO[model]
    sys.path.insert(0, str(repo.resolve()))
    if model == "openvla":
        from embodiq.backends.openvla.policy import OpenVLAPolicy as PolicyClass
        kwargs = {}
    elif model == "fastwam":
        from embodiq.backends.fastwam.policy import FastWAMPolicy as PolicyClass
        kwargs = dict(
            repo_root=repo,
            dataset_stats=base.EMB / "FastWAM/checkpoints/fastwam_release"
                                     "/libero_uncond_2cam224_dataset_stats.json",
            compile_infer=False, seed=42, action_infer_mode=fastwam_mode)
        if job:
            kwargs['dataset_stats'] = Path(job['dataset_stats'])
            kwargs['seed'] = job['seed']
    else:
        from embodiq.backends.adapter.policy import VLAAdapterPolicy as PolicyClass
        kwargs = {}

    quant_apply = None
    quant_meta = {"method": "fp16", "scope": []}
    if quant != "fp16":
        from embodiq.entry.run_adapter import MODEL_SCOPES
        from embodiq.quant.api import make_quant_apply
        from embodiq.quant.spec import HSASpec, QuantSpec

        spec = QuantSpec(method=quant, hsa=HSASpec() if quant == "hsa" else None)
        scope = MODEL_SCOPES[model]
        quant_apply = make_quant_apply(spec, scopes=set(scope), artifact=artifact,
                                       verbose=False)
        quant_meta = {"method": quant, "scope": list(scope), "spec_id": spec.id,
                      "spec_hash": spec.spec_hash}

    checkpoint = Path(job['checkpoint']) if job else base.checkpoint(model, suite)
    policy = PolicyClass(str(checkpoint), suite, device_index=0,
                         quant_apply=quant_apply, **kwargs)
    # Cold start is its own number, not part of the steady state.
    t0 = time.perf_counter()
    policy.load()
    load_s = time.perf_counter() - t0
    load_sampler_hz = policy._probe.sampler_hz
    policy._probe.sampler_hz = NVML_SAMPLER_HZ

    try:
        observations = []
        for task in task_rows:
            for row in task["observations"]:
                path = ROOT / row["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError(f"{path}: drifted from the input manifest")
                with np.load(path, allow_pickle=False) as archive:
                    raw = {k: archive[k] for k in archive.files}
                observations.append(
                    build_observation(raw, row["frame_id"], task["instruction"]))
        if not observations:
            raise ValueError(f"no inputs for {model}/{suite}")

        def one_query(obs) -> float:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                chunk = policy.act(obs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if chunk is None or chunk.num_valid < 1:
                raise RuntimeError("policy returned no executable action")
            return elapsed

        # Warmup: 20 queries, cycling the real inputs. The first is reported on
        # its own -- it carries lazy kernel/autotune cost that does not recur.
        warm = [one_query(observations[i % len(observations)])
                for i in range(warmup_queries)]
        first_query_s, warmup_tail_s = warm[0], (sum(warm[1:]) / max(1, len(warm) - 1))

        # Steady state starts after warmup: drop the allocator's warmup
        # transients so the reported peak is the inference peak.
        policy._probe.reset_peak()

        acc = LatencyAccumulator(warmup_queries_excluded=warmup_queries,
                                 compile_overhead_s=first_query_s)
        per_pass = []
        for _ in range(measured_passes):
            pass_start = time.perf_counter()
            for obs in observations:
                acc.add_query(one_query(obs))
            per_pass.append(time.perf_counter() - pass_start)

        replan = policy.spec.default_replan_steps
        # add_env_steps lets summary() derive per_env_step_s on the same basis
        # the closed-loop runs use: one query covers `replan` executed actions.
        acc.add_env_steps(len(acc.queries) * replan, sum(per_pass))
        latency = acc.summary(replan)
        memory = policy.memory_footprint()
        report = {
            "suite": suite,
            "checkpoint": str(checkpoint),
            "n_inputs": len(observations),
            "n_queries_measured": len(acc.queries),
            "passes": measured_passes,
            "pass_seconds": per_pass,
            "chunk_size": policy.spec.chunk_size,
            "replan_steps": replan,
            "cold_start": {"load_s": load_s, "first_query_s": first_query_s,
                           "warmup_tail_mean_s": warmup_tail_s,
                           "warmup_queries": warmup_queries,
                           "nvml_sampler_hz": load_sampler_hz},
            "latency": latency,
            # Serialized exactly as the closed-loop results do, derived
            # properties included, so the two sources are directly comparable.
            "memory": {**dataclasses.asdict(memory),
                       "activation_peak_bytes": memory.activation_peak_bytes,
                       "runtime_overhead_bytes": memory.runtime_overhead_bytes},
            "policy": {"model_id": policy.spec.model_id,
                       "torch_dtype": policy.spec.torch_dtype,
                       "extra": policy.spec.extra},
            "quant": quant_meta,
        }
        if getattr(policy, "quant_report", None) is not None:
            acct = policy.quant_report.accounting()
            acct.pop("layers", None)
            report["quant"]["accounting"] = acct
            report["quant"]["artifact"] = getattr(policy.quant_report, "artifact", None)
        return report
    finally:
        policy.unload()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-inputs", action="store_true",
                    help="build the shared 200-input manifest and exit")
    ap.add_argument("--model", choices=base.REPO.keys())
    ap.add_argument("--quant", default="fp16", choices=("fp16", "rtn", "awq", "hsa"))
    ap.add_argument("--quant-artifact", type=Path)
    ap.add_argument("--fastwam-mode", default="first_frame",
                    choices=("first_frame", "idm"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--suites", default="", help="comma-separated subset; default all four")
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-contention", action="store_true",
                    help="measure anyway and mark the result contended")
    ap.add_argument("--job-plan", type=Path, help="immutable managed plan with explicit per-suite model paths")
    ap.add_argument("--smoke", action="store_true", help="development only: one input, two warmups, one measured query per suite")
    args = ap.parse_args()

    if args.make_inputs:
        doc = make_inputs(args.inputs)
        print(f"wrote {args.inputs}: {doc['n_observations']} observations "
              f"over {doc['n_tasks']} tasks, manifest {doc['manifest_sha256'][:12]}")
        return 0

    if args.model is None:
        ap.error("--model is required unless --make-inputs")
    if args.quant in ("awq", "hsa") and args.quant_artifact is None:
        ap.error(f"--quant {args.quant} requires --quant-artifact")
    if args.quant not in ("awq", "hsa") and args.quant_artifact is not None:
        ap.error(f"--quant {args.quant} takes no artifact")
    if args.fastwam_mode != "first_frame" and args.model != "fastwam":
        ap.error("--fastwam-mode applies only to --model fastwam")

    inputs = load_inputs(args.inputs)
    suites = [s for s in (args.suites.split(",") if args.suites else base.SUITES) if s]
    unknown = [s for s in suites if s not in base.SUITES]
    if unknown:
        ap.error(f"unknown suites: {unknown}")
    jobs = {}
    if args.job_plan:
        plan = json.loads(args.job_plan.read_text())
        jobs = {j['suite']: j for j in plan['jobs']
                if j['model'] == args.model and j['quant'] == args.quant}
        if not set(suites).issubset(jobs):
            ap.error('job plan is missing a requested model/method/suite')
        if args.quant not in {'fp16', 'rtn'}:
            ap.error('managed resource execution currently supports fp16/rtn only')
        if any(j['fastwam_mode'] != args.fastwam_mode for j in jobs.values()):
            ap.error('temporal mode differs from managed plan')

    before = other_processes(args.gpu)
    if before and not args.allow_contention:
        print(f"GPU {args.gpu} is not idle: {before}\n"
              f"Protocol section 8 keeps contended resource numbers out of the main "
              f"table. Wait for an exclusive window, or pass --allow-contention to "
              f"record a diagnostic run.", file=sys.stderr)
        return 2

    env_gpu = str(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=env_gpu, MUJOCO_EGL_DEVICE_ID=env_gpu,
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      DIFFSYNTH_SKIP_DOWNLOAD="true", TOKENIZERS_PARALLELISM="false")
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()

    out = args.out or (OUT_ROOT / f"{args.model}_{args.quant}.json")
    doc = {
        "schema": "embodiq/resource_bench/1",
        "started": time.strftime("%FT%T%z"),
        "model": args.model,
        "quant": args.quant,
        "quant_artifact": str(args.quant_artifact) if args.quant_artifact else None,
        "fastwam_mode": args.fastwam_mode if args.model == "fastwam" else None,
        "gpu": args.gpu,
        "inputs_manifest_sha256": inputs["manifest_sha256"],
        "inputs_caveat": inputs["caveat"],
        "warmup_queries": WARMUP_QUERIES,
        "measured_passes": MEASURED_PASSES,
        "nvml_sampler_hz": NVML_SAMPLER_HZ,
        "validation_only": args.smoke,
        "plan_sha256": hashlib.sha256(args.job_plan.read_bytes()).hexdigest() if args.job_plan else None,
        "contended_before": before,
        "suites": {},
    }
    by_suite = {s: [t for t in inputs["tasks"] if t["suite"] == s] for s in suites}
    warmup, passes = (2, 1) if args.smoke else (WARMUP_QUERIES, MEASURED_PASSES)
    doc.update(warmup_queries=warmup, measured_passes=passes)
    if args.smoke:
        by_suite = {s: [{**rows[0], 'observations': rows[0]['observations'][:1]}]
                    for s, rows in by_suite.items()}
    try:
        for suite in suites:
            print(f"[{args.model}/{args.quant}] {suite}: "
                  f"{sum(len(t['observations']) for t in by_suite[suite])} inputs",
                  flush=True)
            doc["suites"][suite] = bench_suite(
                args.model, suite, args.quant, args.quant_artifact, args.gpu,
                by_suite[suite], args.fastwam_mode, job=jobs.get(suite),
                warmup_queries=warmup, measured_passes=passes)
            base.atomic_json(out, doc)
    except BaseException as exc:
        doc.update(phase="crashed", error=f"{type(exc).__name__}: {exc}")
        base.atomic_json(out, doc)
        raise

    after = other_processes(args.gpu)
    doc["contended_after"] = after
    doc["contended"] = bool(before or after)
    doc["phase"] = "complete"
    doc["finished"] = time.strftime("%FT%T%z")
    # One aggregate over all measured queries, which is the 12-row table's cell.
    queries = [q for s in doc["suites"].values() for q in [s["latency"]["query_s"]]]
    doc["aggregate"] = {
        "n_suites": len(doc["suites"]),
        "n_queries": sum(s["n_queries_measured"] for s in doc["suites"].values()),
        "query_s_mean": (sum(q["mean"] * q["n"] for q in queries)
                         / sum(q["n"] for q in queries)) if queries else None,
        "query_s_p95_max": max((q["p95"] for q in queries), default=None),
        "per_env_step_s_mean": (
            sum(s["latency"]["per_env_step_s"] for s in doc["suites"].values())
            / len(doc["suites"])) if doc["suites"] else None,
        "peak_alloc_bytes_max": max(
            (s["memory"]["peak_alloc_bytes"] for s in doc["suites"].values()), default=None),
        "nvml_peak_bytes_max": max(
            (s["memory"]["nvml_peak_bytes"] for s in doc["suites"].values()), default=None),
        "load_peak_alloc_bytes_max": max(
            (s["memory"].get("load_peak_alloc_bytes", 0)
             for s in doc["suites"].values()), default=None),
        "weights_bytes_max": max(
            (s["memory"]["weights_bytes"] for s in doc["suites"].values()), default=None),
    }
    base.atomic_json(out, doc)
    agg = doc["aggregate"]
    print(f"wrote {out}\n"
          f"  queries={agg['n_queries']}  query_s={agg['query_s_mean']:.4f}  "
          f"per_env_step_s={agg['per_env_step_s_mean']:.4f}\n"
          f"  steady peak={agg['peak_alloc_bytes_max'] / 2**30:.2f} GiB  "
          f"load peak={(agg['load_peak_alloc_bytes_max'] or 0) / 2**30:.2f} GiB  "
          f"contended={doc['contended']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
