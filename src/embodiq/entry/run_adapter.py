"""Run one VLA-Adapter suite end to end and emit per-task result JSON.

One worker owns one suite, because VLA-Adapter ships a DIFFERENT checkpoint per
suite -- so a worker that hopped between suites would reload a 3 GB model every
10 tasks for nothing. Four suites across four GPUs is naturally balanced.

    python -m embodiq.entry.run_adapter --suite libero_spatial --gpu 0 \
        --quant fp16 --trials 50 --out results/<run_id>
"""

from __future__ import annotations

# must precede anything that imports tensorflow
from embodiq.backends.compat import tf_cpu_guard

tf_cpu_guard.apply()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

# Resolved from this file, not hardcoded: the project directory was renamed on
# 2026-09-20 and a literal here is what killed the running pipeline. The
# compatibility symlink keeps string paths working, but rglob()/resolve() return
# the canonical path, so anything comparing against a literal still breaks.
EMB_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = Path(__file__).resolve().parents[1]
# Where the VLA-Adapter Pro checkpoints actually live. EMB_ROOT/checkpoints has
# never existed, so the default always failed and every real invocation passed
# --checkpoint by hand. Overridable with EMBODIQ_CKPT_ROOT.
CKPT_ROOT = Path(os.environ.get("EMBODIQ_CKPT_ROOT", "/data/jyc/data/VLA-Adapter"))
SUITE_TO_CKPT = {
    "libero_spatial": "LIBERO-Spatial-Pro",
    "libero_object": "LIBERO-Object-Pro",
    "libero_goal": "LIBERO-Goal-Pro",
    "libero_10": "LIBERO-Long-Pro",
}

# The module range for the 1.4 main matrix, one list per model, SHARED by
# rtn/awq/hsa. Protocol section 6: the three methods must cover the same range,
# or the 48-cell table is a collage of whichever range scored best. The per-model
# asymmetry (OpenVLA llm-only) is deliberate and reported alongside
# quantized_param_fraction -- see docs/task14_gap_analysis.md section 3.3.
MODEL_SCOPES = {
    "openvla": ["llm"],
    "vla_adapter": ["llm", "vision_dino", "vision_siglip", "projector", "action_head"],
    "fastwam": ["video_dit", "action_dit"],
}

# Methods that cannot run without a validated calibration artifact from 1.2/1.3.
ARTIFACT_QUANTS = ("awq", "awq-llm", "hsa")

QUANT_PRESETS = {
    "fp16": None,
    "rtn": {"method": "rtn"},
    "awq": {"method": "awq"},
    # Supplementary experiment: public-AWQ-style language-only adaptation.
    # It is intentionally outside the formal shared-scope matrix.
    "awq-llm": {"method": "awq", "scopes": ["llm"]},
    "hsa": {"method": "hsa"},
    # Supplementary scope rows, not part of the main matrix.
    "rtn-llm": {"method": "rtn", "scopes": ["llm"]},
    "rtn-all": {"method": "rtn", "scopes": ["llm", "vision_dino", "vision_siglip",
                                            "projector", "action_head"]},
}


def _git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _detect_contention(device_index: int) -> dict:
    """Is another process sharing this GPU?

    Contention does not affect success rate, but it inflates latency by up to
    2.2x (measured). Recording it per result file means the aggregator can keep
    contended rows out of the latency table without discarding their SR.
    """
    try:
        import pynvml
        from embodiq.core.memory import nvml_device_index

        pynvml.nvmlInit()
        physical_index = nvml_device_index(device_index)
        h = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        import os

        others = [p for p in procs if p.pid != os.getpid()]
        return {
            "contended": bool(others),
            "nvml_device_index": physical_index,
            "n_other_procs": len(others),
            "other_mem_gib": round(sum(p.usedGpuMemory or 0 for p in others) / 2**30, 2),
        }
    except Exception:
        return {"contended": None, "n_other_procs": None}


def _versions() -> dict:
    import numpy, torch, transformers

    out = {"python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": numpy.__version__}
    for mod in ("mujoco", "robosuite"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "?"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("vla_adapter", "openvla", "fastwam"), default="vla_adapter")
    ap.add_argument("--checkpoint", type=Path, help="explicit checkpoint directory")
    ap.add_argument("--repo-root", type=Path, help="native model source checkout")
    ap.add_argument("--dataset-stats", type=Path, help="FastWAM dataset statistics JSON")
    ap.add_argument("--compile-action-infer", action="store_true", help="FastWAM compiled inference (off for functional validation)")
    ap.add_argument("--fastwam-mode", default="first_frame", choices=("first_frame", "idm"),
                    help="FastWAM temporal prediction: 'idm' denoises future frames with "
                         "the video DiT first (needs the optional-IDM checkpoint); "
                         "'first_frame' skips it. The main matrix is first_frame.")
    ap.add_argument("--seed", type=int, help="override protocol seed explicitly")
    ap.add_argument("--suite", required=True, choices=sorted(SUITE_TO_CKPT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quant", default="fp16", choices=sorted(QUANT_PRESETS))
    ap.add_argument("--quant-artifact", type=Path,
                    help="AWQ/HSA artifact from 1.2/1.3 (embodiq/quant-artifact/1); "
                         "required for --quant awq|hsa, rejected otherwise")
    ap.add_argument("--hsa-rho", type=float, default=0.01,
                    help="HSA residual budget: share of input channels kept in "
                         "high precision (default 0.01)")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, default=CKPT_ROOT,
                    help="directory containing the four LIBERO-*-Pro checkpoints")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="write an in-flight checkpoint every N episodes so a stop "
                         "costs N episodes instead of a whole task (0 disables)")
    args = ap.parse_args(argv)
    if args.fastwam_mode != "first_frame" and args.model != "fastwam":
        ap.error("--fastwam-mode applies only to --model fastwam")
    if args.quant in ARTIFACT_QUANTS and args.quant_artifact is None:
        ap.error(f"--quant {args.quant} requires --quant-artifact: the scale/residual "
                 f"search is produced by 1.2/1.3 and this entry refuses to run RTN "
                 f"under an {args.quant.upper()} label")
    if args.quant not in ARTIFACT_QUANTS and args.quant_artifact is not None:
        ap.error(f"--quant {args.quant} takes no artifact")


    repo = args.repo_root or EMB_ROOT / {"openvla": "OpenVLA", "vla_adapter": "VLA-Adapter", "fastwam": "FastWAM"}[args.model]
    if not repo.is_dir():
        ap.error(f"model source directory does not exist: {repo}")
    # Both repositories export experiments/prismatic. A worker may load only
    # one repository, and must reject already-imported conflicting modules.
    for name in ("experiments", "prismatic"):
        module = sys.modules.get(name)
        paths = list(getattr(module, "__path__", [])) if module else []
        if paths and any(not Path(p).resolve().is_relative_to(repo.resolve()) for p in paths):
            ap.error(f"{name} already imported from another repo; start a fresh worker")
    sys.path.insert(0, str(repo.resolve()))
    if args.model == "openvla":
        from embodiq.backends.openvla.policy import OpenVLAPolicy as PolicyClass
    elif args.model == "fastwam":
        if args.quant in ("rtn-llm", "rtn-all"):
            ap.error("rtn-llm/rtn-all are Prismatic-scope presets; FastWAM uses "
                     "--quant rtn with its own video_dit/action_dit range")
        if args.dataset_stats is None or not args.dataset_stats.is_file():
            ap.error("FastWAM requires an existing --dataset-stats JSON")
        from embodiq.backends.fastwam.policy import FastWAMPolicy as PolicyClass
    else:
        from embodiq.backends.adapter.policy import VLAAdapterPolicy as PolicyClass
    from embodiq.backends.prismatic_env import PrismaticEnvFactory
    from embodiq.core.protocol import PROTOCOLS
    from embodiq.core.result import TaskResult
    from embodiq.core.rollout import run_episode
    from embodiq.core.timing import LatencyAccumulator

    proto = dataclasses.replace(
        PROTOCOLS[args.protocol or f"official.{args.model}"], num_trials_per_task=args.trials,
        save_videos=args.save_videos,
    )
    if args.seed is not None:
        proto = dataclasses.replace(proto, seed_base=args.seed)
    if args.model in ("openvla", "fastwam") and args.checkpoint is None:
        ap.error(f"{args.model} requires --checkpoint")
    ckpt = str(args.checkpoint or args.checkpoint_root / SUITE_TO_CKPT[args.suite])
    if not (Path(ckpt).is_file() if args.model == "fastwam" else Path(ckpt).is_dir()):
        ap.error(f"checkpoint directory does not exist: {ckpt}; set --checkpoint-root")

    quant_apply = None
    quant_meta = {"method": "fp16", "mode": "none", "scope": []}
    preset = QUANT_PRESETS[args.quant]
    if preset is not None:
        from embodiq.quant.api import make_quant_apply
        from embodiq.quant.spec import HSASpec, QuantSpec

        # Main-matrix methods take the model's shared range; the rtn-llm/rtn-all
        # supplementary presets carry their own.
        scope = preset.get("scopes") or MODEL_SCOPES[args.model]
        hsa = HSASpec(rho=args.hsa_rho) if preset["method"] == "hsa" else None
        spec = QuantSpec(method=preset["method"], hsa=hsa)
        quant_apply = make_quant_apply(spec, scopes=set(scope),
                                       artifact=args.quant_artifact, verbose=True)
        quant_meta = {"method": spec.method, "mode": "packed", "w_bits": spec.w_bit,
                      "group_size": spec.group_size, "scope": list(scope),
                      "kernel": "tinygemm", "spec_id": spec.id, "spec_hash": spec.spec_hash}
        if hsa is not None:
            quant_meta["hsa"] = dataclasses.asdict(hsa)

    # Load the model ONCE for the whole suite.
    policy_kwargs = {}
    if args.model == "fastwam":
        policy_kwargs = dict(repo_root=repo, dataset_stats=args.dataset_stats,
                             compile_infer=args.compile_action_infer, seed=proto.seed_base,
                             action_infer_mode=args.fastwam_mode)
    policy = PolicyClass(ckpt, args.suite, device_index=0, quant_apply=quant_apply, **policy_kwargs)
    t_load = time.time()
    policy.load()
    policy.spec = dataclasses.replace(policy.spec, quant_id=args.quant)
    print(f"[{args.suite}/{args.quant}] model loaded in {time.time()-t_load:.1f}s", flush=True)
    if getattr(policy, "quant_report", None) is not None:
        quant_meta.update(policy.quant_report.accounting())
        quant_meta.pop("layers", None)
        quant_meta["tensor_storage_bytes_before"] = policy.quant_report.tensor_storage_before
        quant_meta["tensor_storage_bytes_after"] = policy.quant_report.tensor_storage_after
        # Historical accounting covers Linear modules, not the whole model.
        quant_meta["compression_linear_modules_estimate"] = quant_meta.pop("compression_whole_model", None)
        quant_meta["tensor_storage_compression"] = policy.quant_report.tensor_storage_before / policy.quant_report.tensor_storage_after
        artifact_meta = getattr(policy.quant_report, "artifact", None)
        if artifact_meta is not None:
            quant_meta["artifact"] = artifact_meta
            quant_meta["implementation"] = artifact_meta["implementation"]
            quant_meta["implementation_version"] = artifact_meta["implementation_version"]
        report_dir = args.out / "quant_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{args.suite}_{args.quant}.json").write_text(
            json.dumps(policy.quant_report.to_dict(), indent=2))
    # Validate source identity even with uncommitted local implementation work.
    import hashlib
    source_hash = hashlib.sha256()
    # Keyed on the path WITHIN the package, so the digest does not change when
    # the checkout moves. Recorded alongside the digest as source_hash_basis so
    # a value from a different basis is never silently compared against one of
    # these.
    for path in sorted(PKG_ROOT.rglob("*.py")):
        source_hash.update(str(path.relative_to(PKG_ROOT)).encode())
        source_hash.update(path.read_bytes())

    if args.model == "fastwam":
        from embodiq.backends.fastwam.env import FastWAMEnvFactory
        factory = FastWAMEnvFactory()
    else:
        factory = PrismaticEnvFactory()
    task_ids = ([int(t) for t in args.tasks.split(",") if t != ""]
                or list(range(factory.n_tasks(args.suite))))
    args.out.mkdir(parents=True, exist_ok=True)

    git = {"embodiq": _git_sha(EMB_ROOT / "embodiq"),
           "model_repo": _git_sha(repo), "embodiq_source_sha256": source_hash.hexdigest(),
           # Results written before 2026-10-03 keyed the digest on the path
           # relative to the parent of the checkout, so their value is not
           # comparable with this one. Stating the basis makes that checkable
           # instead of a silent mismatch.
           "embodiq_source_hash_basis": "paths relative to src/embodiq"}
    versions = _versions()
    if proto.name in ("unified.v2", "unified.v3"):
        if versions.get("mujoco") != "3.3.2" or versions.get("robosuite") != "1.4.1":
            policy.unload()
            raise RuntimeError("Unified baselines require mujoco=3.3.2 and robosuite=1.4.1")

    contention = _detect_contention(0)
    if contention.get("contended"):
        print(f"[{args.suite}] WARNING: GPU shared with {contention['n_other_procs']} "
              f"other process(es) using {contention['other_mem_gib']} GiB -- "
              f"resource measurements are not headline results; contended=true.",
              flush=True)

    suite_ok = suite_n = suite_errors = 0
    for task_id in task_ids:
        result_file = args.out / args.suite / f"gpu{args.gpu}_task{task_id}_results.json"
        if result_file.exists():
            from embodiq.core.resume import validate_resume
            saved = json.loads(result_file.read_text())
            validate_resume(saved, spec=policy.spec, protocol=proto, quant=quant_meta,
                            git=git, versions=versions, suite=args.suite, task_id=task_id)
            suite_ok += saved["successes"]
            suite_n += saved["total_episodes"]
            print(f"[{args.suite}] task {task_id}: already done, skipping", flush=True)
            continue

        desc = factory.task_description(args.suite, task_id)
        states = factory.initial_states(args.suite, task_id, proto.num_trials_per_task)
        state_file_sha256 = hashlib.sha256(factory.initial_state_path(args.suite, task_id).read_bytes()).hexdigest()
        env, _ = factory.make(args.suite, task_id, resolution=proto.env_resolution,
                              seed=proto.seed_base)

        tr = TaskResult(
            task_suite=args.suite, task_id=task_id, task_description=desc,
            spec=policy.spec, protocol_name=proto.name,
            protocol_fields={k: v for k, v in dataclasses.asdict(proto).items() if k != "name"},
            gpu_id=args.gpu, quant=quant_meta, env_versions=versions, git=git,
            contended=bool(contention.get("contended")),
        )
        tr.env_versions = {**versions, "gpu_contention": contention}
        acc = LatencyAccumulator()

        # An in-flight checkpoint turns a stop into the loss of one episode
        # instead of a whole task. It is only usable when the config, the code
        # and the per-episode initial states all still match; anything else and
        # the task restarts from zero rather than being partially trusted.
        start_episode = 0
        partial_path = tr.partial_path(args.out)
        if args.checkpoint_every and partial_path.is_file():
            from embodiq.core.resume import rebuild_accumulator, validate_partial
            from embodiq.core.types import EpisodeRecord

            import numpy as np_mod

            def state_digest(index: int) -> str:
                return hashlib.sha256(
                    np_mod.ascontiguousarray(states[index]).tobytes()).hexdigest()

            saved_partial = json.loads(partial_path.read_text())
            start_episode, reason = validate_partial(
                saved_partial, spec=policy.spec, protocol=proto, quant=quant_meta,
                git=git, versions=versions, suite=args.suite, task_id=task_id,
                initial_state_sha256=state_digest)
            if start_episode:
                tr.episodes = [EpisodeRecord.from_dict(e)
                               for e in saved_partial["episodes"]]
                n_queries = rebuild_accumulator(saved_partial, acc)
                tr.aux_resumed = {"resumed_from_episode": start_episode,
                                  "restored_queries": n_queries,
                                  "memory_covers": "post_resume_only"}
                print(f"[{args.suite}] task {task_id}: resuming at episode "
                      f"{start_episode + 1}/{proto.num_trials_per_task} "
                      f"({n_queries} query timings restored)", flush=True)
            else:
                print(f"[{args.suite}] task {task_id}: discarding checkpoint "
                      f"({reason}); restarting the task", flush=True)
                partial_path.unlink(missing_ok=True)

        # Warmup is per-process, not per-task: only pay it on the first task.
        if suite_n == 0 and start_episode == 0:
            from embodiq.core.rollout import build_observation

            env.reset()
            o = build_observation(env.set_init_state(states[0]), 0, desc)
            warm = policy.warmup(o, n=5)
            acc.compile_overhead_s = sum(warm)
            acc.warmup_queries_excluded = len(warm)
            policy._probe.reset_peak()

        t0 = time.time()
        for ep in range(start_episode, proto.num_trials_per_task):
            rec = run_episode(env, policy, proto, suite=args.suite, task_id=task_id,
                              episode_idx=ep, initial_state=states[ep],
                              task_description=desc, dummy_action=factory.dummy_action(),
                              accumulator=acc)
            tr.episodes.append(rec)
            import numpy as np
            state_array = np.ascontiguousarray(states[ep])
            rec.aux.update(initial_state_id=ep, initial_state_file_sha256=state_file_sha256,
                           initial_state_sha256=hashlib.sha256(state_array.tobytes()).hexdigest(),
                           initial_state_dtype=str(state_array.dtype))
            print(f"[{args.suite}] task {task_id} episode {ep + 1}/{proto.num_trials_per_task} "
                  f"status={rec.status.value} successes={tr.n_success} errors={tr.n_error}", flush=True)
            # Checkpoint AFTER aux is filled in, so the saved episode carries the
            # initial-state hash a later resume validates against.
            if args.checkpoint_every and (ep + 1) % args.checkpoint_every == 0:
                tr.latency = acc
                tr.memory = policy.memory_footprint()
                tr.duration = time.time() - t0
                tr.write_partial(args.out)
        env.close()

        tr.latency = acc
        tr.memory = policy.memory_footprint()
        tr.duration = time.time() - t0
        tr.write(args.out)

        suite_ok += tr.n_success
        suite_n += tr.n_scored
        suite_errors += tr.n_error
        print(f"[{args.suite}] task {task_id}: {tr.n_success}/{tr.n_scored} "
              f"({100*tr.n_success/max(1,tr.n_scored):.1f}%)  err={tr.n_error}  "
              f"{tr.duration/60:.1f} min  | running {100*suite_ok/max(1,suite_n):.1f}%", flush=True)

    policy.unload()
    print(f"[{args.suite}/{args.quant}] SUITE DONE: {suite_ok}/{suite_n} = "
          f"{100*suite_ok/max(1,suite_n):.2f}%", flush=True)
    return 1 if suite_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
