import dataclasses
import random
import numpy as np
import pytest

from embodiq.core.protocol import PROTOCOLS
from embodiq.core.rollout import run_episode
from embodiq.core.types import EpisodeStatus
from test_rollout_error_surfacing import FakeEnv, FakePolicy, KW


@pytest.mark.parametrize("protocol_name,seed", [("unified.v2", 0), ("unified.v3", 42)])
def test_unified_episode_resets_rng_and_environment_seed(protocol_name, seed):
    class SeededEnv(FakeEnv):
        def seed(self, value): self.seen_seed = value
    proto = PROTOCOLS[protocol_name]
    samples = []
    class SamplingPolicy(FakePolicy):
        def act(self, obs):
            samples.append((random.random(), np.random.random()))
            return super().act(obs)
    for _ in range(2):
        env = SeededEnv(succeed_at=31)
        r = run_episode(env, SamplingPolicy(), **{**KW, "protocol": proto})
        assert r.status is EpisodeStatus.SUCCESS
        assert env.seen_seed == seed
    assert samples[0] == samples[1]


def test_environment_failure_stays_invalid():
    class BrokenEnv(FakeEnv):
        def reset(self): raise RuntimeError("environment unavailable")
    r = run_episode(BrokenEnv(), FakePolicy(), **KW)
    assert r.status is EpisodeStatus.ENV_ERROR
    assert not r.counts_toward_success_rate


def test_policy_reset_failure_is_model_failure():
    class BrokenPolicy(FakePolicy):
        def reset(self, **kw): raise ValueError("invalid policy state")
    r = run_episode(FakeEnv(), BrokenPolicy(), **KW)
    assert r.status is EpisodeStatus.POLICY_ERROR
    assert r.counts_toward_success_rate
