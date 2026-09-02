"""V1: prove the Policy reproduces the repo's native action path bit-for-bit.

Method: drive the REPO's own episode loop (run_libero_eval.run_episode) with
get_action monkeypatched to record every (raw_obs, task_description, action)
triple, then replay those raw observations through our Policy and require
np.array_equal -- exact equality, not allclose.

This is stronger than calling get_action twice, because the recording runs
through the repo's own cfg construction, its own model init, its own
prepare_observation and its own process_action. Anything our adapter reorders,
drops (the 0.9 center crop), or substitutes (the Qwen chat template vs the plain
prompt) shows up as a mismatch.

    python tools/capture_parity_fixture.py --suite libero_spatial --task-id 0 \
        --checkpoint .../LIBERO-Spatial-Pro --episodes 1 --max-records 20
"""

from __future__ import annotations

# before anything pulls in tensorflow
from embodiq.backends.compat import tf_cpu_guard

tf_cpu_guard.apply()

import argparse  # noqa: E402
import pickle  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/adapter"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from embodiq.backends.compat import adapter_freeze
    from experiments.robot import openvla_utils, robot_utils
    from experiments.robot.libero import run_libero_eval as rle

    adapter_freeze.apply(openvla_utils)

    from embodiq.backends.adapter.policy import make_cfg

    cfg = make_cfg(args.checkpoint, args.suite)
    cfg.num_trials_per_task = args.episodes
    cfg.num_steps_wait = 10
    cfg.initial_states_path = "DEFAULT"

    model, action_head, proprio_projector, noisy, processor = rle.initialize_model(cfg)

    records: list[dict] = []
    native_get_action = robot_utils.get_action
    native_prepare = rle.prepare_observation

    # prepare_observation runs EVERY step (run_libero_eval.py:326) but get_action
    # only when the queue drains (~every 8 steps), so two independent recorders
    # would be misaligned by up to 7 steps and every query after the first would
    # be paired with the wrong observation. Instead: prepare_observation parks
    # the raw obs, and get_action -- which the repo calls immediately after --
    # consumes it.
    pending: dict = {}

    def recording_prepare(obs, resize_size):
        pending["raw"] = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in obs.items()}
        return native_prepare(obs, resize_size)

    def recording_get_action(cfg_, model_, obs_, task_label_, **kw):
        out = native_get_action(cfg_, model_, obs_, task_label_, **kw)
        if len(records) < args.max_records and "raw" in pending:
            records.append({
                "raw": pending["raw"],
                "task_label": task_label_,
                "actions": [np.array(a, copy=True) for a in
                            (out if isinstance(out, list) else [out])],
            })
        return out

    # And capture what actually reaches the simulator. The repo applies
    # process_action per POPPED action, after the queue (run_libero_eval.py:353),
    # so get_action's return value is pre-gripper-transform. Recording env.step's
    # argument gives the true end of the pipeline, which is what our Policy
    # returns and therefore what parity must be measured against.
    stepped: list[np.ndarray] = []
    native_process = rle.process_action

    def recording_process(action, model_family):
        out = native_process(action, model_family)
        stepped.append(np.array(out, copy=True))
        return out

    robot_utils.get_action = recording_get_action
    rle.get_action = recording_get_action
    rle.prepare_observation = recording_prepare
    rle.process_action = recording_process

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    from experiments.robot.libero.libero_utils import get_libero_env

    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.seed(0)

    for ep in range(args.episodes):
        if len(records) >= args.max_records:
            break
        rle.run_episode(
            cfg, env, task_description=desc, model=model, resize_size=224,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector, noisy_action_projector=noisy,
            initial_state=states[ep], log_file=None,
        )
    env.close()

    robot_utils.get_action = native_get_action
    rle.prepare_observation = native_prepare
    rle.process_action = native_process

    # Query i produced actions that were executed as stepped[i*8 : (i+1)*8].
    # The final chunk may be truncated if the episode ended mid-chunk.
    chunk = int(cfg.num_open_loop_steps)
    fixture = []
    for i, rec in enumerate(records):
        exec_slice = stepped[i * chunk : (i + 1) * chunk]
        if not exec_slice:
            break
        fixture.append({
            "raw": rec["raw"],
            "task_description": rec["task_label"],
            "actions_raw": rec["actions"],   # pre process_action
            "actions": exec_slice,           # what env.step actually received
        })
    n = len(fixture)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.suite}_task{args.task_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(fixture, f, protocol=4)
    print(f"captured {n} query records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
