"""Contract and runner tests that do not require a model checkpoint or GPU."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from embodiq.core.protocol import PROTOCOLS
from embodiq.core.rollout import run_episode
from embodiq.core.types import (
    ACTION_DIM,
    MODEL_CONTRACTS,
    ActionChunk,
    EpisodeStatus,
    GripperConvention,
    MemoryReport,
    Observation,
    Policy,
    PolicySpec,
    validate_policy_spec,
)


def _raw_obs() -> dict:
    return {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def _spec(family: str, *, replan: int | None = None) -> PolicySpec:
    contract = MODEL_CONTRACTS[family]
    return PolicySpec(
        model_id=f"mock-{family}",
        family=family,
        checkpoint="mock",
        cameras=contract.cameras,
        env_resolution=contract.env_resolution,
        uses_proprio=contract.uses_proprio,
        chunk_size=contract.chunk_size,
        default_replan_steps=(
            contract.native_replan_steps if replan is None else replan
        ),
        gripper=GripperConvention.ENV_NATIVE,
        torch_dtype="float32",
    )


class MockChunkPolicy(Policy):
    """A deterministic backend with FastWAM's 32-predicted/10-executed shape."""

    def __init__(self, family: str = "fastwam") -> None:
        self.spec = _spec(family)
        self.query_count = 0

    def load(self) -> None:
        return None

    def reset(self, **_kwargs) -> None:
        self.query_count = 0

    def memory_footprint(self) -> MemoryReport:
        return MemoryReport()

    def act(self, _obs: Observation) -> ActionChunk:
        base = self.query_count * 100
        self.query_count += 1
        actions = np.stack(
            [
                np.full((ACTION_DIM,), base + i, dtype=np.float32)
                for i in range(self.spec.chunk_size)
            ]
        )
        return ActionChunk(
            actions=actions,
            num_valid=self.spec.default_replan_steps,
        )


class TrackingEnv:
    def __init__(self, done_at: int | None = None) -> None:
        self.done_at = done_at
        self.t = 0
        self.actions: list[np.ndarray] = []

    def reset(self):
        self.t = 0
        self.actions.clear()
        return _raw_obs()

    def set_init_state(self, _state):
        return _raw_obs()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.t += 1
        done = self.done_at is not None and self.t >= self.done_at
        return _raw_obs(), 0.0, done, {}


def test_native_model_contracts_are_explicit():
    assert MODEL_CONTRACTS["openvla"].chunk_size == 1
    assert MODEL_CONTRACTS["vla_adapter"].chunk_size == 8
    assert MODEL_CONTRACTS["fastwam"].chunk_size == 32
    assert MODEL_CONTRACTS["fastwam"].native_replan_steps == 10

    for family in MODEL_CONTRACTS:
        validate_policy_spec(_spec(family), allow_replan_override=False)


def test_replan_override_keeps_chunk_contract():
    # A cadence isolation row may execute fewer actions than the native
    # cadence, but it may never claim a chunk wider than the model returns.
    spec = _spec("fastwam", replan=4)
    validate_policy_spec(spec)
    with pytest.raises(ValueError, match="chunk_size"):
        PolicySpec(
            model_id="bad",
            family="fastwam",
            checkpoint="mock",
            cameras=("agentview", "wrist"),
            env_resolution=256,
            uses_proprio=True,
            chunk_size=8,
            default_replan_steps=8,
            gripper=GripperConvention.ENV_NATIVE,
            torch_dtype="float32",
        )


def test_observation_and_action_contract_reject_bad_values():
    obs = Observation(
        images={"agentview": _raw_obs()["agentview_image"]},
        proprio=np.zeros(8, dtype=np.float32),
        raw={},
        t=0,
        task_description="pick the block",
    )
    assert obs.proprio.shape == (8,)

    with pytest.raises(ValueError, match="3 channels"):
        Observation(
            images={"agentview": np.zeros((256, 256), dtype=np.uint8)},
            proprio=np.zeros(8, dtype=np.float32),
            raw={},
            t=0,
            task_description="task",
        )

    with pytest.raises(ValueError, match="NaN"):
        ActionChunk(
            actions=np.full((1, ACTION_DIM), np.nan, dtype=np.float32),
            num_valid=1,
        )

    with pytest.raises(ValueError, match="num_valid"):
        ActionChunk(
            actions=np.zeros((1, ACTION_DIM), dtype=np.float32),
            num_valid=2,
        )


def test_unified_runner_executes_fastwam_valid_prefix_only():
    """One query predicts 32 actions, but the runner executes only 10."""
    proto = dataclasses.replace(
        PROTOCOLS["official.fastwam"],
        max_steps={
            "libero_spatial": 25,
            "libero_object": 25,
            "libero_goal": 25,
            "libero_10": 25,
        },
        num_steps_wait=0,
        replan_steps=None,
    )
    env = TrackingEnv(done_at=25)
    policy = MockChunkPolicy("fastwam")
    rec = run_episode(
        env,
        policy,
        proto,
        suite="libero_spatial",
        task_id=0,
        episode_idx=0,
        initial_state=np.zeros(1),
        task_description="pick the block",
        dummy_action=[0.0] * ACTION_DIM,
    )

    assert rec.status is EpisodeStatus.SUCCESS
    assert rec.env_steps == 25
    assert rec.n_queries == 3
    assert len(env.actions) == 25
    assert np.array_equal(
        np.asarray([a[0] for a in env.actions[:10]]),
        np.arange(10, dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray([a[0] for a in env.actions[10:20]]),
        np.arange(100, 110, dtype=np.float32),
    )
    assert np.array_equal(
        np.asarray([a[0] for a in env.actions[20:25]]),
        np.arange(200, 205, dtype=np.float32),
    )
