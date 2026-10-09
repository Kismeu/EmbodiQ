"""Result records.

Schema authorship is deliberately inverted: FastWAM is the only one of the three
repos that already emits JSON results, so we adopt ITS field names as the base
and make the two Prismatic models conform, rather than inventing a schema
against VLA-Adapter and bending it later.

The record is a strict SUPERSET of FastWAM's per-task JSON, which means
``FastWAM/experiments/libero/summarize_results.py`` reads our files unmodified.
That is a free cross-check on every aggregation we write.

Schema v2 counts model errors as failures while retaining diagnostic details.
Environment errors are invalid trials; incomplete/degraded runs must not enter
the final aggregate. OOM remains a fatal infrastructure error in the runner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from embodiq.core.protocol import mcnemar_delta, wilson_ci95
from embodiq.core.timing import LatencyAccumulator
from embodiq.core.types import EpisodeRecord, EpisodeStatus, MemoryReport, PolicySpec

# /3 adds the load-phase memory attribution (load_peak_*, cache_released_bytes)
# and the AWQ/HSA artifact identity under quant.artifact. schema/2 files are
# left as they are; tools must branch on this field, not assume.
SCHEMA_VERSION = "embodiq/run_result/3"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, EpisodeStatus):
            return o.value
        return super().default(o)


@dataclass
class TaskResult:
    """One work unit: (model, quant, protocol, suite, task_id)."""

    task_suite: str
    task_id: int
    task_description: str
    spec: PolicySpec
    protocol_name: str
    protocol_fields: dict
    gpu_id: int
    episodes: list[EpisodeRecord] = field(default_factory=list)
    latency: LatencyAccumulator = field(default_factory=LatencyAccumulator)
    memory: MemoryReport = field(default_factory=MemoryReport)
    quant: dict = field(default_factory=dict)
    env_versions: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    duration: float = 0.0
    worker_id: Optional[str] = None
    contended: bool = False

    # -- derived ---------------------------------------------------------
    @property
    def n_error(self) -> int:
        return sum(1 for e in self.episodes if e.status in (EpisodeStatus.POLICY_ERROR, EpisodeStatus.ENV_ERROR))

    @property
    def n_success(self) -> int:
        return sum(1 for e in self.episodes if e.status is EpisodeStatus.SUCCESS)

    @property
    def n_scored(self) -> int:
        return sum(1 for e in self.episodes if e.counts_toward_success_rate)

    @property
    def status(self) -> str:
        if not self.episodes:
            return "failed"
        return "degraded" if self.n_error else "ok"

    def to_dict(self) -> dict:
        lo, hi = wilson_ci95(self.n_success, self.n_scored)
        replan = self.protocol_fields.get("replan_steps") or self.spec.default_replan_steps
        return {
            "schema_version": SCHEMA_VERSION,
            # --- FastWAM-compatible core (summarize_results.py reads these) ---
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "successes": self.n_success,
            "total_episodes": len(self.episodes),
            "gpu_id": self.gpu_id,
            "success_episodes": [e.episode_idx for e in self.episodes if e.status is EpisodeStatus.SUCCESS],
            "failure_episodes": [e.episode_idx for e in self.episodes if e.status is EpisodeStatus.TIMEOUT],
            "start_time": self.start_time,
            "duration": self.duration,
            "worker_id": self.worker_id,
            # --- embodiq extensions ---
            # Diagnostics remain separate even when model errors are scored.
            "error_episodes": [
                {"episode_idx": e.episode_idx, "status": e.status.value, "error": e.error}
                for e in self.episodes
                if e.status in (EpisodeStatus.POLICY_ERROR, EpisodeStatus.ENV_ERROR)
            ],
            "run": {
                "status": self.status,
                "gpu_id": self.gpu_id,
                "contended": self.contended,
                "git": self.git,
                "versions": self.env_versions,
            },
            "policy": {
                **{k: v for k, v in asdict(self.spec).items() if k != "gripper"},
                "gripper": self.spec.gripper.value,
            },
            "quant": self.quant,
            "protocol": {"name": self.protocol_name, **self.protocol_fields},
            "episodes": [
                {
                    "episode_idx": e.episode_idx,
                    "status": e.status.value,
                    "env_steps": e.env_steps,
                    "n_queries": e.n_queries,
                    "seed": e.seed,
                    "query_latency_s": e.query_latency_s,
                    "env_step_s_total": e.env_step_s_total,
                    "error": e.error,
                    "aux": e.aux,
                }
                for e in self.episodes
            ],
            "metrics": {
                "n_episodes": len(self.episodes),
                "n_success": self.n_success,
                "n_timeout": sum(1 for e in self.episodes if e.status is EpisodeStatus.TIMEOUT),
                "n_error": self.n_error,
                "n_policy_error": sum(e.status is EpisodeStatus.POLICY_ERROR for e in self.episodes),
                "n_invalid": sum(e.status is EpisodeStatus.ENV_ERROR for e in self.episodes),
                "success_rate": (self.n_success / self.n_scored) if self.n_scored else None,
                "success_ci95": [lo, hi],
                "latency": self.latency.summary(replan),
                "memory": {
                    **asdict(self.memory),
                    "activation_peak_bytes": self.memory.activation_peak_bytes,
                    "runtime_overhead_bytes": self.memory.runtime_overhead_bytes,
                },
            },
        }

    def result_path(self, output_root: Path) -> Path:
        return (Path(output_root) / self.task_suite
                / f"gpu{self.gpu_id}_task{self.task_id}_results.json")

    def partial_path(self, output_root: Path) -> Path:
        """Where the in-flight checkpoint lives. Leading dot so a stray partial
        can never be mistaken for a finished task by the ``*_results.json`` globs
        every tool uses."""
        return (Path(output_root) / self.task_suite
                / f".gpu{self.gpu_id}_task{self.task_id}_partial.json")

    def write(self, output_root: Path) -> Path:
        """Atomic write, mirroring FastWAM's tmp+os.replace and filename layout."""
        out = self.result_path(output_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / f".{out.name}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=4, cls=_NumpyEncoder), encoding="utf-8")
        os.replace(tmp, out)
        # The task is finished; its checkpoint is now noise that could only
        # confuse a later resume.
        self.partial_path(output_root).unlink(missing_ok=True)
        return out

    def write_partial(self, output_root: Path) -> Path:
        """Checkpoint after an episode, so a stop costs one episode not one task.

        Carries the same identity fields a finished result carries, because a
        partial is only safe to resume from when the policy, quantization,
        protocol, code and environment all still match -- see
        ``core.resume.validate_partial``. The per-episode query latencies ride
        along inside each EpisodeRecord, so the accumulator can be rebuilt and
        the task's latency summary still covers all of its episodes.
        """
        out = self.partial_path(output_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["partial"] = True
        payload["episodes_completed"] = len(self.episodes)
        tmp = out.parent / f"{out.name}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, indent=1, cls=_NumpyEncoder), encoding="utf-8")
        os.replace(tmp, out)
        return out


def paired_success(baseline: dict, variant: dict) -> dict:
    """Compare two result dicts episode-by-episode on identical initial states.

    Only episodes that scored in BOTH arms are paired; an error on either side
    drops the pair rather than silently counting as a failure.
    """
    def outcomes(d: dict) -> dict[int, bool]:
        return {
            e["episode_idx"]: e["status"] == "success"
            for e in d["episodes"]
            if e["status"] in ("success", "timeout")
        }

    b, v = outcomes(baseline), outcomes(variant)
    shared = sorted(set(b) & set(v))
    res = mcnemar_delta([(b[i], v[i]) for i in shared])
    res["n_dropped_unpaired"] = len(set(b) ^ set(v))
    return res
