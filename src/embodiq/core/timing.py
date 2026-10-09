"""Timing primitives.

torch is imported lazily inside the functions so that ``embodiq.core`` stays
importable in a torch-free interpreter (the ctl launcher lives in base py3.12).
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class _Elapsed:
    elapsed_s: float = 0.0


@contextlib.contextmanager
def cuda_sync_timer(device_index: int = 0) -> Iterator[_Elapsed]:
    """Wall time with a CUDA sync on both sides.

    Without the leading sync the measurement absorbs whatever work was still
    queued from the previous call; without the trailing one it reports launch
    time instead of execution time.
    """
    try:
        import torch

        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    except Exception:
        sync = None

    if sync is not None:
        sync(device_index)
    t0 = time.perf_counter()
    out = _Elapsed()
    try:
        yield out
    finally:
        if sync is not None:
            sync(device_index)
        out.elapsed_s = time.perf_counter() - t0


@dataclass
class LatencyAccumulator:
    """Collects per-query latencies and derives the four reported figures.

    The three models query at different densities (OpenVLA 1 per env step,
    VLA-Adapter 1 per 8, FastWAM 1 per 10), so per-query latency alone is not
    comparable across them. per_env_step_s is the only figure that is.
    """

    queries: list[float] = field(default_factory=list)
    n_env_steps: int = 0
    env_step_s_total: float = 0.0
    compile_overhead_s: float = 0.0
    warmup_queries_excluded: int = 0

    def add_query(self, seconds: float) -> None:
        self.queries.append(seconds)

    def add_env_steps(self, n: int, seconds: float) -> None:
        self.n_env_steps += n
        self.env_step_s_total += seconds

    def merge(self, other: "LatencyAccumulator") -> None:
        self.queries.extend(other.queries)
        self.n_env_steps += other.n_env_steps
        self.env_step_s_total += other.env_step_s_total

    def summary(self, replan_steps: int) -> dict:
        q = sorted(self.queries)
        if not q:
            return {"n": 0}

        def pct(p: float) -> float:
            if len(q) == 1:
                return q[0]
            idx = min(len(q) - 1, max(0, int(round(p * (len(q) - 1)))))
            return q[idx]

        per_env_step = sum(q) / self.n_env_steps if self.n_env_steps else float("nan")
        return {
            "query_s": {
                "mean": statistics.fmean(q),
                "std": statistics.pstdev(q) if len(q) > 1 else 0.0,
                "p50": pct(0.50),
                "p90": pct(0.90),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "min": q[0],
                "max": q[-1],
                "n": len(q),
            },
            "per_env_step_s": per_env_step,
            "per_action_s": pct(0.50) / max(1, replan_steps),
            "control_freq_hz": (1.0 / per_env_step) if per_env_step and per_env_step == per_env_step else None,
            "queries_per_100_steps": 100.0 * len(q) / self.n_env_steps if self.n_env_steps else None,
            "env_step_s_total": self.env_step_s_total,
            "compile_overhead_s": self.compile_overhead_s,
            "warmup_queries_excluded": self.warmup_queries_excluded,
        }


def raw_latencies(records) -> list[float]:
    """Flatten per-episode arrays.

    Aggregation MUST re-derive percentiles from the concatenated raw array.
    Averaging per-task percentiles is not a percentile of anything.
    """
    out: list[float] = []
    for r in records:
        out.extend(r.query_latency_s)
    return out
