"""Controller durability: the faults that cost two suite cells on 2026-09-20.

A long pipeline supervises subprocesses whose results are already durable on
disk. The controller's own bookkeeping must therefore never be able to kill it,
must never leave a lie behind when it does die, and must let a later invocation
pick up exactly the cells that are missing. Each test below pins one of those.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import run_unified_baselines as base  # noqa: E402


def write_task(directory: Path, task_id: int, *, episodes=50, errors=(), successes=40):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"gpu0_task{task_id}_results.json").write_text(json.dumps({
        "task_id": task_id,
        "total_episodes": episodes,
        "successes": successes,
        "error_episodes": list(errors),
    }))


# -- atomic_json -----------------------------------------------------------

def test_atomic_json_creates_missing_parents(tmp_path):
    """The original raised FileNotFoundError here and killed the controller."""
    target = tmp_path / "deep" / "nested" / "status.json"
    base.atomic_json(target, {"phase": "running"})
    assert json.loads(target.read_text())["phase"] == "running"


def test_atomic_json_leaves_no_temp_behind(tmp_path):
    target = tmp_path / "status.json"
    base.atomic_json(target, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_atomic_json_does_not_mangle_multi_dot_names(tmp_path):
    """with_suffix() rewrites the last dotted component, so a.b.json -> a.tmp."""
    target = tmp_path / "status.v3.json"
    base.atomic_json(target, {"a": 1})
    assert target.exists()
    assert not (tmp_path / "status.v3.tmp").exists()
    assert not (tmp_path / "status.tmp").exists()


def test_atomic_json_replaces_in_place(tmp_path):
    target = tmp_path / "status.json"
    base.atomic_json(target, {"n": 1})
    base.atomic_json(target, {"n": 2})
    assert json.loads(target.read_text())["n"] == 2


def test_atomic_json_keeps_non_ascii_readable(tmp_path):
    target = tmp_path / "status.json"
    base.atomic_json(target, {"phase": "已完成"})
    assert "已完成" in target.read_text(encoding="utf-8")


# -- status_writer ---------------------------------------------------------

def test_status_writer_survives_an_unwritable_path(tmp_path):
    """The actual 2026-09-20 failure: a status write raised and took nine
    finished jobs' supervisor down with it."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    state = {"phase": "running"}
    logged: list[str] = []
    save = base.status_writer(blocker / "status.json", state, log=lambda *a, **k: logged.append(a[0]))

    save()          # must not raise
    save()

    assert state["status_write_failures"] == 1, "the failure count must be visible in the state"
    assert "status_write_last_error" in state
    assert logged, "a failed status write must be logged, not swallowed silently"


def test_status_writer_records_a_timestamp_and_writes(tmp_path):
    target = tmp_path / "status.json"
    state = {"phase": "running"}
    base.status_writer(target, state)()
    on_disk = json.loads(target.read_text())
    assert on_disk["phase"] == "running" and on_disk["updated"]
    assert "status_write_failures" not in on_disk


# -- mark_crashed ----------------------------------------------------------

def test_mark_crashed_replaces_running_with_crashed(tmp_path):
    """A frozen "running" with no live PID is indistinguishable from a slow job;
    that is why OpenVLA Goal sat in "running" for two weeks."""
    target = tmp_path / "status.json"
    state = {"phase": "running", "jobs": []}
    base.atomic_json(target, state)
    try:
        raise RuntimeError("source changed after freeze")
    except RuntimeError as exc:
        base.mark_crashed(target, state, exc)

    on_disk = json.loads(target.read_text())
    assert on_disk["phase"] == "crashed"
    assert "source changed after freeze" in on_disk["error"]
    assert on_disk["traceback"] and on_disk["crashed_at"]


def test_mark_crashed_does_not_raise_when_it_cannot_write(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    base.mark_crashed(blocker / "status.json", {"phase": "running"},
                      RuntimeError("x"), log=lambda *a, **k: None)


# -- cell_tasks ------------------------------------------------------------

def test_cell_tasks_on_a_complete_cell(tmp_path):
    for i in range(10):
        write_task(tmp_path, i)
    present, missing, complete = base.cell_tasks(tmp_path)
    assert present == list(range(10)) and missing == [] and complete


def test_cell_tasks_identifies_the_one_missing_task(tmp_path):
    """OpenVLA RTN libero_goal: 9 of 10 tasks, 450 of 500 episodes."""
    for i in range(9):
        write_task(tmp_path, i)
    present, missing, complete = base.cell_tasks(tmp_path)
    assert present == list(range(9))
    assert missing == [9]
    assert not complete


def test_cell_tasks_on_an_empty_cell(tmp_path):
    present, missing, complete = base.cell_tasks(tmp_path)
    assert present == [] and missing == list(range(10)) and not complete


def test_cell_tasks_rejects_a_short_task(tmp_path):
    """A task file with 30 episodes is not a finished task."""
    for i in range(9):
        write_task(tmp_path, i)
    write_task(tmp_path, 9, episodes=30)
    present, missing, complete = base.cell_tasks(tmp_path)
    assert 9 not in present and missing == [9] and not complete


def test_cell_tasks_rejects_a_task_with_error_episodes(tmp_path):
    """An error episode makes the cell incomplete; it is never counted as a
    failure and never dropped to make the average look whole."""
    for i in range(10):
        write_task(tmp_path, i)
    write_task(tmp_path, 7, errors=[3])
    present, missing, complete = base.cell_tasks(tmp_path)
    assert 7 not in present and not complete


def test_cell_tasks_ignores_unreadable_files(tmp_path):
    for i in range(10):
        write_task(tmp_path, i)
    (tmp_path / "gpu0_task99_results.json").write_text("{ truncated")
    present, _, complete = base.cell_tasks(tmp_path)
    assert present == list(range(10)) and complete


def test_cell_tasks_is_configurable_for_other_protocols(tmp_path):
    write_task(tmp_path, 0, episodes=10)
    write_task(tmp_path, 1, episodes=10)
    present, missing, complete = base.cell_tasks(tmp_path, expect=2, episodes=10)
    assert present == [0, 1] and missing == [] and complete


# -- clean stop ------------------------------------------------------------

def load_rtn_controller():
    """Import the RTN controller without running it."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools/run_rtn_validation.py"
    spec = importlib.util.spec_from_file_location("rtn_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeChild:
    """Stands in for a Popen: records terminate/kill instead of signalling."""

    def __init__(self, pid, exits_on_terminate=True):
        self.pid = pid
        self.terminated = self.killed = False
        self._exits = exits_on_terminate
        self._done = False

    def poll(self):
        return 0 if self._done else None

    def terminate(self):
        self.terminated = True
        if self._exits:
            self._done = True

    def wait(self, timeout=None):
        if not self._done:
            raise TimeoutError("still running")
        return 0

    def kill(self):
        self.killed = True
        self._done = True


def capture_handler(module, monkeypatch, state, save):
    """install_stop_handlers registers with signal.signal; grab what it registers."""
    import signal as signal_module

    registered = {}
    monkeypatch.setattr(signal_module, "signal",
                        lambda num, fn: registered.setdefault(num, fn))
    module.install_stop_handlers(Path("unused"), state, save)
    assert signal_module.SIGTERM in registered and signal_module.SIGINT in registered
    return registered[signal_module.SIGTERM]


def test_sigterm_records_the_stop_instead_of_freezing_at_running(monkeypatch):
    """A controller killed without a handler leaves phase="running" forever --
    the state that hid a dead pipeline for two weeks."""
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    state = {"phase": "running", "jobs": [{"model": "openvla", "suite": "libero_10",
                                           "status": "running"}]}
    saves = []
    handler = capture_handler(module, monkeypatch, state, lambda: saves.append(dict(state)))

    with pytest.raises(SystemExit) as exit_info:
        handler(signal_module.SIGTERM, None)

    assert exit_info.value.code == 130
    assert state["phase"] == "stopped"
    assert state["stopped_by"] == "SIGTERM"
    assert state["stopped_at"]
    # The job in flight is reported as interrupted, not left claiming to run.
    assert state["jobs"][0]["status"] == "stopped_while_running"
    assert saves, "the stop must be persisted"


def test_sigterm_terminates_live_workers(monkeypatch):
    """Otherwise a 17 GiB evaluator keeps holding the GPU after the stop."""
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    child = FakeChild(4242)
    module.LIVE[child.pid] = child
    handler = capture_handler(module, monkeypatch, {"phase": "running"}, lambda: None)

    with pytest.raises(SystemExit):
        handler(signal_module.SIGTERM, None)

    assert child.terminated and not child.killed


def test_a_worker_that_ignores_terminate_is_killed(monkeypatch):
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    child = FakeChild(99, exits_on_terminate=False)
    module.LIVE[child.pid] = child
    handler = capture_handler(module, monkeypatch, {"phase": "running"}, lambda: None)

    with pytest.raises(SystemExit):
        handler(signal_module.SIGTERM, None)

    assert child.terminated and child.killed


def test_a_second_signal_is_ignored(monkeypatch):
    """Two Ctrl-C presses must not restart the teardown mid-flight."""
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    handler = capture_handler(module, monkeypatch, {"phase": "running"}, lambda: None)
    module.STOPPING.set()

    assert handler(signal_module.SIGINT, None) is None


def test_stop_note_states_the_resume_granularity(monkeypatch):
    """The user needs to know what a stop costs: one task, not one episode."""
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    state = {"phase": "running"}
    handler = capture_handler(module, monkeypatch, state, lambda: None)
    with pytest.raises(SystemExit):
        handler(signal_module.SIGTERM, None)

    note = state["stop_note"]
    assert "per task" in note and "not per" in note


def test_a_deliberate_stop_is_not_reported_as_a_crash(tmp_path, monkeypatch):
    """The handler records phase="stopped" then raises SystemExit to unwind. If
    the outer handler treats that as a crash, an intentional stop reads as a
    failure -- which is what the first real --stop produced."""
    import signal as signal_module

    module = load_rtn_controller()
    module.STOPPING.clear()
    module.LIVE.clear()
    target = tmp_path / "status.json"
    state = {"phase": "running", "jobs": []}
    save = base.status_writer(target, state)
    handler = capture_handler(module, monkeypatch, state, save)

    with pytest.raises(SystemExit):
        handler(signal_module.SIGTERM, None)
    assert json.loads(target.read_text())["phase"] == "stopped"

    # The outer guard must leave that alone while a stop is in progress.
    assert module.STOPPING.is_set()
    if not module.STOPPING.is_set():
        base.mark_crashed(target, state, SystemExit(130))
    assert json.loads(target.read_text())["phase"] == "stopped"
