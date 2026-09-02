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
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

EMB_ROOT = Path("/data/jyc/space/Embodied AI")
CKPT_ROOT = EMB_ROOT / "checkpoints" / "vla-adapter"
SUITE_TO_CKPT = {
    "libero_spatial": "LIBERO-Spatial-Pro",
    "libero_object": "LIBERO-Object-Pro",
    "libero_goal": "LIBERO-Goal-Pro",
    "libero_10": "LIBERO-Long-Pro",
}

QUANT_PRESETS = {
    "fp16": None,
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
    ap.add_argument("--suite", required=True, choices=sorted(SUITE_TO_CKPT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quant", default="fp16", choices=sorted(QUANT_PRESETS))
    ap.add_argument("--protocol", default="official.vla_adapter")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args(argv)

    from embodiq.backends.adapter.policy import VLAAdapterPolicy
    from embodiq.backends.prismatic_env import PrismaticEnvFactory
    from embodiq.core.protocol import PROTOCOLS
    from embodiq.core.result import TaskResult
    from embodiq.core.rollout import run_episode
    from embodiq.core.timing import LatencyAccumulator

    proto = dataclasses.replace(
        PROTOCOLS[args.protocol], num_trials_per_task=args.trials,
        save_videos=args.save_videos,
    )
    ckpt = str(CKPT_ROOT / SUITE_TO_CKPT[args.suite])

    quant_apply = None
    quant_meta = {"method": "fp16", "mode": "none", "scope": []}
    preset = QUANT_PRESETS[args.quant]
    if preset is not None:
        from embodiq.quant.api import make_quant_apply
        from embodiq.quant.spec import QuantSpec

        spec = QuantSpec(method=preset["method"])
        quant_apply = make_quant_apply(spec, scopes=set(preset["scopes"]), verbose=True)
        quant_meta = {"method": spec.method, "mode": "packed", "w_bits": spec.w_bit,
                      "group_size": spec.group_size, "scope": preset["scopes"],
                      "kernel": "tinygemm", "spec_id": spec.id, "spec_hash": spec.spec_hash}

    # Load the model ONCE for the whole suite.
    policy = VLAAdapterPolicy(ckpt, args.suite, device_index=0, quant_apply=quant_apply)
    t_load = time.time()
    policy.load()
    print(f"[{args.suite}/{args.quant}] model loaded in {time.time()-t_load:.1f}s", flush=True)
    if getattr(policy, "quant_report", None) is not None:
        quant_meta.update(policy.quant_report.accounting())
        quant_meta.pop("layers", None)

    factory = PrismaticEnvFactory()
    task_ids = ([int(t) for t in args.tasks.split(",") if t != ""]
                or list(range(factory.n_tasks(args.suite))))
    args.out.mkdir(parents=True, exist_ok=True)

    git = {"embodiq": _git_sha(EMB_ROOT / "embodiq"),
           "model_repo": _git_sha(EMB_ROOT / "VLA-Adapter")}
    versions = _versions()

    suite_ok = suite_n = 0
    for task_id in task_ids:
        result_file = args.out / args.suite / f"gpu{args.gpu}_task{task_id}_results.json"
        if result_file.exists():
            print(f"[{args.suite}] task {task_id}: already done, skipping", flush=True)
            continue

        desc = factory.task_description(args.suite, task_id)
        states = factory.initial_states(args.suite, task_id, proto.num_trials_per_task)
        env, _ = factory.make(args.suite, task_id, resolution=proto.env_resolution,
                              seed=proto.seed_base)

        tr = TaskResult(
            task_suite=args.suite, task_id=task_id, task_description=desc,
            spec=policy.spec, protocol_name=proto.name,
            protocol_fields={k: v for k, v in dataclasses.asdict(proto).items() if k != "name"},
            gpu_id=args.gpu, quant=quant_meta, env_versions=versions, git=git,
        )
        acc = LatencyAccumulator()
        # Warmup is per-process, not per-task: only pay it on the first task.
        if suite_n == 0:
            from embodiq.core.rollout import build_observation

            env.reset()
            o = build_observation(env.set_init_state(states[0]), 0, desc)
            warm = policy.warmup(o, n=5)
            acc.compile_overhead_s = sum(warm)
            acc.warmup_queries_excluded = len(warm)
            policy._probe.reset_peak()

        t0 = time.time()
        for ep in range(proto.num_trials_per_task):
            rec = run_episode(env, policy, proto, suite=args.suite, task_id=task_id,
                              episode_idx=ep, initial_state=states[ep],
                              task_description=desc, dummy_action=factory.dummy_action(),
                              accumulator=acc)
            tr.episodes.append(rec)
        env.close()

        tr.latency = acc
        tr.memory = policy.memory_footprint()
        tr.duration = time.time() - t0
        tr.write(args.out)

        suite_ok += tr.n_success
        suite_n += tr.n_scored
        print(f"[{args.suite}] task {task_id}: {tr.n_success}/{tr.n_scored} "
              f"({100*tr.n_success/max(1,tr.n_scored):.1f}%)  err={tr.n_error}  "
              f"{tr.duration/60:.1f} min  | running {100*suite_ok/max(1,suite_n):.1f}%", flush=True)

    policy.unload()
    print(f"[{args.suite}/{args.quant}] SUITE DONE: {suite_ok}/{suite_n} = "
          f"{100*suite_ok/max(1,suite_n):.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
