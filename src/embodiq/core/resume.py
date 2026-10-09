"""Reject stale results rather than silently reusing another model's run."""
from dataclasses import asdict
import json

from embodiq.core.types import EpisodeStatus


def validate_resume(saved, *, spec, protocol, quant, git, versions, suite, task_id):
    expected = {
        "task_suite": suite, "task_id": task_id,
        "policy": asdict(spec), "quant": quant,
        "protocol": asdict(protocol),
    }
    expected = json.loads(json.dumps(expected))
    mismatches = [key for key, value in expected.items() if saved.get(key) != value]
    run = saved.get("run", {})
    if run.get("git") != git:
        mismatches.append("git")
    recorded_versions = {k: v for k, v in run.get("versions", {}).items() if k != "gpu_contention"}
    if recorded_versions != versions:
        mismatches.append("versions")
    if saved.get("total_episodes") != protocol.num_trials_per_task or saved.get("error_episodes"):
        mismatches.append("incomplete_or_error_episodes")
    if run.get("status") != "ok":
        mismatches.append("run.status")
    if mismatches:
        raise ValueError("Existing result is incompatible; use a new output directory: " + ", ".join(mismatches))


def validate_partial(saved, *, spec, protocol, quant, git, versions, suite, task_id,
                     initial_state_sha256):
    """Decide whether an in-flight checkpoint may be resumed from.

    Stricter than validate_resume in one way that matters: besides the config
    having to match, every episode already recorded must sit at its own index and
    carry the initial-state hash this run would have used for that index. The
    protocol requires a task's 50 episodes to be its 50 distinct initial states
    in order; a checkpoint that was written against a different state list would
    silently produce a task that is 50 episodes long but not the right 50.

    Returns the number of episodes to skip. A checkpoint that fails any check is
    discarded and the task restarts from zero -- never partially trusted.

    Args:
        initial_state_sha256: callable (episode_idx) -> hex digest for this run.
    """
    from dataclasses import asdict
    import json as _json

    if not saved.get("partial"):
        return 0, "not a partial checkpoint"
    expected = {
        "task_suite": suite, "task_id": task_id,
        "policy": asdict(spec), "quant": quant, "protocol": asdict(protocol),
    }
    expected = _json.loads(_json.dumps(expected))
    for key, value in expected.items():
        if saved.get(key) != value:
            return 0, f"{key} differs"
    run = saved.get("run", {})
    if run.get("git") != git:
        return 0, "git differs"
    recorded = {k: v for k, v in run.get("versions", {}).items() if k != "gpu_contention"}
    if recorded != versions:
        return 0, "versions differ"

    episodes = saved.get("episodes") or []
    if not episodes:
        return 0, "no episodes recorded"
    if len(episodes) >= protocol.num_trials_per_task:
        return 0, "checkpoint is already complete but was never finalized"
    for position, episode in enumerate(episodes):
        if episode.get("episode_idx") != position:
            return 0, f"episode {position} recorded out of order"
        aux = episode.get("aux") or {}
        if aux.get("initial_state_sha256") != initial_state_sha256(position):
            return 0, f"episode {position} used a different initial state"
        # ENV_ERROR is the protocol's INVALID: the episode did not produce a
        # valid trial and must be re-run rather than carried forward, so a
        # checkpoint containing one is discarded. POLICY_ERROR is a legitimate
        # scored outcome (a model failure is a failure) and resumes normally.
        if episode.get("status") == EpisodeStatus.ENV_ERROR.value:
            return 0, f"episode {position} is an env_error (INVALID) episode"
    return len(episodes), "ok"


def rebuild_accumulator(saved, accumulator):
    """Restore the query latencies of resumed episodes.

    Without this a resumed task's latency summary would describe only the
    episodes that ran after the restart, which is a different measurement wearing
    the same field name.
    """
    n = 0
    for episode in saved.get("episodes") or []:
        for seconds in episode.get("query_latency_s") or []:
            accumulator.add_query(float(seconds))
            n += 1
        steps = episode.get("env_steps")
        total = episode.get("env_step_s_total")
        if steps and total is not None:
            accumulator.add_env_steps(int(steps), float(total))
    return n
