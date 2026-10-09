"""The single episode loop, shared by all three backends.

It touches only ``env.reset()``, ``env.set_init_state()``, ``env.step()`` and
``env.close()`` -- a robosuite 1.4.1 surface that is identical under mujoco
3.3.0 and 3.3.2 -- so one implementation is vendored into all three prefixes.
Everything version-sensitive lives behind ``Policy`` and ``EnvFactory``.

Three things this loop does that none of the three repo loops do:

1. **Four-state termination.** Both Prismatic loops
   (``openvla/.../run_libero_eval.py:235-238``, ``VLA-Adapter/...:365``) wrap the
   step body in a bare ``except Exception`` and let the episode fall through as
   ``success=False``. An OOM, a shape bug, or a broken quantized kernel then
   looks exactly like a policy failure, and "RTN loses 4 points" might be four
   OOMs. Model errors are typed and count as failures; environment errors are
   invalid trials, and OOM/interrupts propagate to stop the worker.
2. **Executed prefix != chunk length.** FastWAM predicts 32 actions and executes
   10. A loop that assumes they are equal works on two of three models.
3. **Video is opt-in and frames are not even accumulated when off.** Both repos
   append unconditionally and write an MP4 per episode; at 1-3 s per episode
   that is 10-30% of a VLA-Adapter episode and it would land in the throughput
   numbers.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Optional, Protocol

import numpy as np

from embodiq.core.protocol import RolloutProtocol
from embodiq.core.timing import LatencyAccumulator
from embodiq.core.types import (
    CAM_AGENTVIEW,
    CAM_WRIST,
    EpisodeRecord,
    EpisodeStatus,
    Observation,
    Policy,
)


class EnvFactory(Protocol):
    def make(self, suite: str, task_id: int, *, resolution: int, seed: int) -> tuple[Any, str]: ...
    def dummy_action(self) -> list[float]: ...
    def initial_states(self, suite: str, task_id: int, n: int) -> np.ndarray: ...


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Identical to the copy in all three repos (they each vendor it)."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def build_observation(raw: dict, t: int, task_description: str) -> Observation:
    """Raw robosuite obs -> Observation. No rotation, no resize, no normalization."""
    images = {CAM_AGENTVIEW: raw["agentview_image"]}
    if "robot0_eye_in_hand_image" in raw:
        images[CAM_WRIST] = raw["robot0_eye_in_hand_image"]
    proprio = np.concatenate(
        (raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
    ).astype(np.float32)
    return Observation(images=images, proprio=proprio, raw=raw, t=t, task_description=task_description)


def run_episode(
    env: Any,
    policy: Policy,
    protocol: RolloutProtocol,
    *,
    suite: str,
    task_id: int,
    episode_idx: int,
    initial_state: np.ndarray,
    task_description: str,
    dummy_action: list[float],
    accumulator: Optional[LatencyAccumulator] = None,
    obs_sink: Optional[Callable[[Observation], None]] = None,
    frame_sink: Optional[Callable[[np.ndarray], None]] = None,
) -> EpisodeRecord:
    seed = protocol.episode_seed(task_id, episode_idx)
    replan = protocol.replan_steps or policy.spec.default_replan_steps
    max_steps = protocol.steps_for(suite)

    rec = EpisodeRecord(
        episode_idx=episode_idx, status=EpisodeStatus.TIMEOUT, env_steps=0, n_queries=0, seed=seed
    )

    def fail(status: EpisodeStatus, exc: BaseException) -> EpisodeRecord:
        rec.status = status
        rec.error = {
            "type": type(exc).__name__,
            "msg": str(exc)[:512],
            "traceback": "".join(traceback.format_exception(exc))[-4096:],
        }
        return rec

    try:
        if protocol.reseed_each_episode:
            import random
            random.seed(seed)
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            env.seed(seed)
        env.reset()
        obs = env.set_init_state(initial_state)
    except BaseException as exc:
        _reraise_fatal(exc)
        return fail(EpisodeStatus.ENV_ERROR, exc)
    try:
        policy.reset(
            task_id=task_id, episode_idx=episode_idx,
            task_description=task_description, seed=seed,
        )
    except BaseException as exc:  # noqa: BLE001
        _reraise_fatal(exc)
        return fail(EpisodeStatus.POLICY_ERROR, exc)

    # Settle period. Excluded from step counts, from latency denominators and
    # from video -- otherwise the wait steps inflate n_env_steps and deflate
    # per_env_step_s by ~5%.
    try:
        for _ in range(protocol.num_steps_wait):
            obs, _, _, _ = env.step(dummy_action)
    except BaseException as exc:  # noqa: BLE001
        _reraise_fatal(exc)
        return fail(EpisodeStatus.ENV_ERROR, exc)

    pending: list[np.ndarray] = []
    env_time = 0.0
    t = 0
    while t < max_steps:
        if not pending:
            try:
                observation = build_observation(obs, t, task_description)
                if obs_sink is not None:
                    obs_sink(observation)
                chunk = policy(observation)
            except BaseException as exc:  # noqa: BLE001
                _reraise_fatal(exc)
                rec.env_steps, rec.env_step_s_total = t, env_time
                return fail(EpisodeStatus.POLICY_ERROR, exc)

            n = min(chunk.num_valid, replan, chunk.actions.shape[0])
            pending = [chunk.actions[i] for i in range(n)]
            rec.n_queries += 1
            rec.query_latency_s.append(chunk.query_latency_s)
            if accumulator is not None:
                accumulator.add_query(chunk.query_latency_s)

        action = pending.pop(0)
        try:
            t0 = time.perf_counter()
            obs, _reward, done, _info = env.step(action.tolist())
            env_time += time.perf_counter() - t0
        except BaseException as exc:  # noqa: BLE001
            _reraise_fatal(exc)
            rec.env_steps, rec.env_step_s_total = t, env_time
            return fail(EpisodeStatus.ENV_ERROR, exc)

        t += 1
        if frame_sink is not None:
            frame_sink(obs["agentview_image"])
        if done:
            rec.status = EpisodeStatus.SUCCESS
            break

    rec.env_steps = t
    rec.env_step_s_total = env_time
    if accumulator is not None:
        accumulator.add_env_steps(t, env_time)
    return rec


def _reraise_fatal(exc: BaseException) -> None:
    """Never swallow these. An OOM recorded as a failed episode is the exact bug
    this module exists to prevent, and a KeyboardInterrupt must reach the user."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    if type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc):
        raise exc
