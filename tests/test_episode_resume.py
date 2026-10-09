"""Episode-level checkpointing: stopping a run must cost one episode, not one task.

The reason this needs careful tests rather than just a happy path: a checkpoint
that is resumed when it should not be produces a task that is 50 episodes long
and looks finished, but is not the 50 episodes the protocol specifies. That is
invisible in every downstream table. So every way a checkpoint can be wrong gets
a test, and each of them must make the task restart from zero rather than be
partially trusted.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from embodiq.core.protocol import PROTOCOLS
from embodiq.core.resume import rebuild_accumulator, validate_partial
from embodiq.core.result import TaskResult
from embodiq.core.timing import LatencyAccumulator
from embodiq.core.types import EpisodeRecord, EpisodeStatus, GripperConvention, PolicySpec

SUITE = "libero_spatial"
TASK_ID = 3
QUANT = {"method": "rtn", "mode": "packed", "scope": ["llm"]}
GIT = {"embodiq": "abc1234", "model_repo": "def5678", "embodiq_source_sha256": "f" * 64}
VERSIONS = {"python": "3.10.21", "torch": "2.2.0+cu121", "mujoco": "3.3.2"}


def spec() -> PolicySpec:
    return PolicySpec(
        model_id="openvla-test", family="openvla", checkpoint="/dev/null",
        cameras=("agentview",), env_resolution=256, uses_proprio=False,
        chunk_size=1, default_replan_steps=1,
        gripper=GripperConvention.ENV_NATIVE_BINARIZED, torch_dtype="bfloat16",
        extra={},
    )


def protocol(trials=50):
    return dataclasses.replace(PROTOCOLS["unified.v3"], num_trials_per_task=trials)


def digest(index: int) -> str:
    """Stand-in for the per-episode initial-state hash of this run."""
    return f"state-{index:04d}"


def episode(index: int, *, status=EpisodeStatus.SUCCESS, state=None, queries=2):
    return EpisodeRecord(
        episode_idx=index, status=status, env_steps=80, n_queries=queries, seed=42,
        query_latency_s=[0.2 + 0.01 * i for i in range(queries)],
        env_step_s_total=1.5,
        aux={"initial_state_id": index,
             "initial_state_sha256": digest(index) if state is None else state},
    )


def task(n_episodes=5, trials=50, **over) -> TaskResult:
    tr = TaskResult(
        task_suite=over.get("suite", SUITE), task_id=over.get("task_id", TASK_ID),
        task_description="pick up the bowl", spec=over.get("spec", spec()),
        protocol_name="unified.v3",
        protocol_fields={k: v for k, v in dataclasses.asdict(protocol(trials)).items()
                         if k != "name"},
        gpu_id=0, quant=over.get("quant", QUANT), env_versions=dict(VERSIONS),
        git=over.get("git", GIT),
    )
    tr.episodes = [episode(i) for i in range(n_episodes)]
    tr.latency = LatencyAccumulator()
    return tr


def saved(tmp_path, tr, trials=50):
    tr.write_partial(tmp_path)
    return json.loads(tr.partial_path(tmp_path).read_text())


def check(payload, *, trials=50, **over):
    kwargs = dict(spec=spec(), protocol=protocol(trials), quant=QUANT, git=GIT,
                  versions=VERSIONS, suite=SUITE, task_id=TASK_ID,
                  initial_state_sha256=digest)
    kwargs.update(over)
    return validate_partial(payload, **kwargs)


# -- the happy path --------------------------------------------------------

def test_a_matching_checkpoint_resumes_at_the_right_episode(tmp_path):
    n, reason = check(saved(tmp_path, task(n_episodes=7)))
    assert (n, reason) == (7, "ok")


def test_partial_is_hidden_from_the_results_glob(tmp_path):
    """Every tool finds finished tasks with *_results.json; a checkpoint must
    never be picked up as one."""
    tr = task(n_episodes=3)
    tr.write_partial(tmp_path)
    assert list((tmp_path / SUITE).glob("*_results.json")) == []
    assert tr.partial_path(tmp_path).name.startswith(".")


def test_finalizing_removes_the_checkpoint(tmp_path):
    tr = task(n_episodes=50)
    tr.write_partial(tmp_path)
    assert tr.partial_path(tmp_path).is_file()
    out = tr.write(tmp_path)
    assert out.is_file() and not tr.partial_path(tmp_path).exists()


def test_restored_latency_covers_the_resumed_episodes(tmp_path):
    """Otherwise the task's latency summary would describe only the episodes that
    ran after the restart, under the same field name."""
    tr = task(n_episodes=4)
    payload = saved(tmp_path, tr)
    acc = LatencyAccumulator()
    restored = rebuild_accumulator(payload, acc)
    assert restored == sum(len(e.query_latency_s) for e in tr.episodes)
    assert len(acc.queries) == restored
    assert acc.n_env_steps == sum(e.env_steps for e in tr.episodes)


# -- every way a checkpoint can be wrong -----------------------------------

def test_a_non_partial_file_is_refused(tmp_path):
    tr = task(n_episodes=5)
    payload = json.loads(tr.partial_path(tmp_path).parent.as_posix() and
                         json.dumps(tr.to_dict()))
    n, reason = check(payload)
    assert n == 0 and "not a partial" in reason


def test_a_different_quantization_is_refused(tmp_path):
    payload = saved(tmp_path, task(n_episodes=5, quant={"method": "fp16", "scope": []}))
    n, reason = check(payload)
    assert n == 0 and reason == "quant differs"


def test_a_different_policy_is_refused(tmp_path):
    other = dataclasses.replace(spec(), model_id="openvla-other")
    payload = saved(tmp_path, task(n_episodes=5, spec=other))
    n, reason = check(payload)
    assert n == 0 and reason == "policy differs"


def test_a_different_code_version_is_refused(tmp_path):
    payload = saved(tmp_path, task(n_episodes=5,
                                   git={**GIT, "embodiq_source_sha256": "0" * 64}))
    n, reason = check(payload)
    assert n == 0 and reason == "git differs"


def test_a_different_environment_is_refused(tmp_path):
    payload = saved(tmp_path, task(n_episodes=5))
    n, reason = check(payload, versions={**VERSIONS, "mujoco": "3.1.6"})
    assert n == 0 and reason == "versions differ"


def test_a_different_task_is_refused(tmp_path):
    payload = saved(tmp_path, task(n_episodes=5, task_id=9))
    n, reason = check(payload)
    assert n == 0 and reason == "task_id differs"


def test_a_different_initial_state_list_is_refused(tmp_path):
    """The protocol fixes a task's 50 episodes to its 50 initial states in order.
    A checkpoint written against a different list would yield a task that is the
    right length and the wrong experiment."""
    tr = task(n_episodes=5)
    tr.episodes[3] = episode(3, state="state-from-another-run")
    n, reason = check(saved(tmp_path, tr))
    assert n == 0 and "different initial state" in reason


def test_out_of_order_episodes_are_refused(tmp_path):
    tr = task(n_episodes=5)
    tr.episodes[2], tr.episodes[3] = tr.episodes[3], tr.episodes[2]
    n, reason = check(saved(tmp_path, tr))
    assert n == 0 and "out of order" in reason


def test_an_env_error_episode_is_refused(tmp_path):
    """ENV_ERROR is the protocol's INVALID: the trial is not valid and must be
    re-run, so a checkpoint carrying one is discarded rather than carried on."""
    tr = task(n_episodes=5)
    tr.episodes[1] = episode(1, status=EpisodeStatus.ENV_ERROR)
    n, reason = check(saved(tmp_path, tr))
    assert n == 0 and "INVALID" in reason


def test_a_policy_error_episode_still_resumes(tmp_path):
    """A model failure is a scored failure, not an invalid trial -- it counts
    toward the success rate, so it must not block a resume."""
    tr = task(n_episodes=5)
    tr.episodes[1] = episode(1, status=EpisodeStatus.POLICY_ERROR)
    n, reason = check(saved(tmp_path, tr))
    assert (n, reason) == (5, "ok")


def test_an_empty_checkpoint_is_refused(tmp_path):
    payload = saved(tmp_path, task(n_episodes=0))
    n, reason = check(payload)
    assert n == 0 and "no episodes" in reason


def test_a_complete_but_unfinalized_checkpoint_is_refused(tmp_path):
    """50 of 50 episodes in a partial means the finalize step never ran. Resuming
    at 50 would write a result that skipped the completion path."""
    payload = saved(tmp_path, task(n_episodes=50))
    n, reason = check(payload)
    assert n == 0 and "already complete" in reason


def test_a_checkpoint_from_a_different_episode_budget_is_refused(tmp_path):
    """A 10-trial screening checkpoint must not seed a 50-trial formal run."""
    payload = saved(tmp_path, task(n_episodes=5, trials=10))
    n, reason = check(payload, trials=50)
    assert n == 0 and reason == "protocol differs"


# -- EpisodeRecord round-trip ---------------------------------------------

def test_episode_record_round_trips(tmp_path):
    original = episode(2, queries=3)
    payload = json.loads(json.dumps(dataclasses.asdict(original), default=str))
    payload["status"] = original.status.value
    assert EpisodeRecord.from_dict(payload) == original


def test_episode_record_rejects_an_unknown_field():
    """An extra key means a different schema wrote the file; resuming from it
    would silently drop whatever that key carried."""
    payload = json.loads(json.dumps(dataclasses.asdict(episode(0)), default=str))
    payload["status"] = "success"
    payload["future_field"] = 1
    with pytest.raises(ValueError, match="has no field"):
        EpisodeRecord.from_dict(payload)
