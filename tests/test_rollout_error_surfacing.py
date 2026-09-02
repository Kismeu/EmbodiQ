"""V7: an error must never be scored as a policy failure.

Both Prismatic eval loops turn an exception into success=False. If we inherited
that, "RTN loses 4 points" could be four OOMs and no quantization delta in this
project would be interpretable. These tests pin the corrected behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodiq.core.protocol import PROTOCOLS
from embodiq.core.rollout import run_episode
from embodiq.core.timing import LatencyAccumulator
from embodiq.core.types import (
    ActionChunk, EpisodeStatus, GripperConvention, Observation, Policy, PolicySpec, MemoryReport,
)


def _spec(chunk: int, replan: int) -> PolicySpec:
    return PolicySpec(
        model_id="fake", family="fake", checkpoint="none", cameras=("agentview",),
        env_resolution=256, uses_proprio=False, chunk_size=chunk,
        default_replan_steps=replan, gripper=GripperConvention.ENV_NATIVE,
        torch_dtype="float32",
    )


class FakePolicy(Policy):
    """Optionally raises on the Nth query."""

    def __init__(self, chunk=1, replan=1, raise_on=None, exc=RuntimeError("boom")):
        self.spec = _spec(chunk, replan)
        self.n = 0
        self.raise_on = raise_on
        self.exc = exc

    def load(self): ...
    def reset(self, **kw): self.n = 0
    def memory_footprint(self): return MemoryReport()

    def act(self, obs: Observation) -> ActionChunk:
        self.n += 1
        if self.raise_on is not None and self.n == self.raise_on:
            raise self.exc
        a = np.zeros((self.spec.chunk_size, 7), dtype=np.float32)
        return ActionChunk(actions=a, num_valid=self.spec.chunk_size)


class FakeEnv:
    """Succeeds at `succeed_at` env steps, else runs to the horizon."""

    def __init__(self, succeed_at=None):
        self.succeed_at = succeed_at
        self.t = 0

    def _obs(self):
        return {
            "agentview_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def reset(self): self.t = 0; return self._obs()
    def set_init_state(self, s): return self._obs()

    def step(self, a):
        self.t += 1
        done = self.succeed_at is not None and self.t >= self.succeed_at
        return self._obs(), 0.0, done, {}


PROTO = PROTOCOLS["official.openvla"]
KW = dict(
    protocol=PROTO, suite="libero_spatial", task_id=0, episode_idx=0,
    initial_state=np.zeros(1), task_description="t",
    dummy_action=[0, 0, 0, 0, 0, 0, -1],
)


def test_success_and_timeout_are_scored():
    r = run_episode(FakeEnv(succeed_at=20), FakePolicy(), **KW)
    assert r.status is EpisodeStatus.SUCCESS and r.counts_toward_success_rate
    r = run_episode(FakeEnv(succeed_at=None), FakePolicy(), **KW)
    assert r.status is EpisodeStatus.TIMEOUT and r.counts_toward_success_rate
    assert r.env_steps == PROTO.steps_for("libero_spatial")


def test_policy_error_is_typed_and_not_scored():
    r = run_episode(FakeEnv(succeed_at=100), FakePolicy(raise_on=5), **KW)
    assert r.status is EpisodeStatus.POLICY_ERROR
    assert not r.counts_toward_success_rate, "an error must not count as a failure"
    assert r.error["type"] == "RuntimeError"
    assert r.error["traceback"], "traceback must be captured, not discarded"
    assert r.env_steps == 4, "partial progress is preserved"


def test_oom_is_reraised_never_swallowed():
    class OutOfMemoryError(RuntimeError): ...
    with pytest.raises(OutOfMemoryError):
        run_episode(FakeEnv(), FakePolicy(raise_on=1, exc=OutOfMemoryError("CUDA out of memory")), **KW)


def test_keyboard_interrupt_reaches_the_user():
    with pytest.raises(KeyboardInterrupt):
        run_episode(FakeEnv(), FakePolicy(raise_on=1, exc=KeyboardInterrupt()), **KW)


def test_executed_prefix_may_be_shorter_than_chunk():
    """FastWAM predicts 32 and executes 10. Queries must be ceil(steps/10), not
    ceil(steps/32), or the per-query latency denominator is wrong by 3x."""
    import dataclasses

    proto = dataclasses.replace(PROTO, replan_steps=10)
    kw = {**KW, "protocol": proto}
    acc = LatencyAccumulator()
    # FakeEnv counts every step it is given, including the 10 settle steps, so
    # succeed_at=95 means 10 wait + 85 scored. The record must report 85.
    r = run_episode(FakeEnv(succeed_at=95), FakePolicy(chunk=32, replan=10), accumulator=acc, **kw)
    assert r.status is EpisodeStatus.SUCCESS
    assert r.env_steps == 85, "settle steps must not appear in the scored step count"
    assert r.n_queries == 9, f"expected ceil(85/10)=9 queries, got {r.n_queries}"
    assert acc.n_env_steps == 85
    # The point of the test: the policy returns 32 actions but only 10 are
    # executed per query, so queries scale with replan, not with chunk_size.
    assert r.n_queries == -(-r.env_steps // 10) != -(-r.env_steps // 32)


def test_wait_steps_excluded_from_counts():
    """30 settle steps must not appear in env_steps or in the latency denominator."""
    import dataclasses

    proto = dataclasses.replace(PROTO, num_steps_wait=30)
    acc = LatencyAccumulator()
    r = run_episode(FakeEnv(succeed_at=1000), FakePolicy(), accumulator=acc,
                    **{**KW, "protocol": proto})
    assert r.env_steps == proto.steps_for("libero_spatial")
    assert acc.n_env_steps == r.env_steps


def test_error_excluded_from_success_rate_denominator():
    from embodiq.core.result import TaskResult
    import dataclasses

    tr = TaskResult(
        task_suite="libero_spatial", task_id=0, task_description="t", spec=_spec(1, 1),
        protocol_name=PROTO.name,
        protocol_fields={k: v for k, v in dataclasses.asdict(PROTO).items() if k != "name"},
        gpu_id=0,
    )
    for i in range(10):
        env = FakeEnv(succeed_at=20 if i < 8 else None)
        pol = FakePolicy(raise_on=1) if i == 9 else FakePolicy()
        tr.episodes.append(run_episode(env, pol, **{**KW, "episode_idx": i}))
    d = tr.to_dict()
    assert d["metrics"]["n_success"] == 8
    assert d["metrics"]["n_error"] == 1
    # 8/9, not 8/10: the error is removed from the denominator entirely.
    assert d["metrics"]["success_rate"] == pytest.approx(8 / 9)
    assert d["run"]["status"] == "degraded"
    assert len(d["error_episodes"]) == 1
