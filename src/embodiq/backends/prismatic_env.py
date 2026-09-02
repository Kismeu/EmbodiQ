"""LIBERO EnvFactory for the two Prismatic-derived repos.

Both vendor an identical `experiments/robot/libero/libero_utils.py`, so one
factory serves either prefix -- whichever repo is importable is the one in use.

One behaviour is deliberately overridden: ``get_libero_env`` hardcodes
``env.seed(0)`` (libero_utils.py:24). We re-seed immediately after construction
so the harness, not the repo, owns determinism. That needs no patch.
"""

from __future__ import annotations

from embodiq.backends.compat import tf_cpu_guard

tf_cpu_guard.apply()

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402


class PrismaticEnvFactory:
    def __init__(self, model_family: str = "openvla") -> None:
        self.model_family = model_family
        self._suites: dict[str, Any] = {}

    def _suite(self, suite: str):
        if suite not in self._suites:
            from libero.libero import benchmark

            self._suites[suite] = benchmark.get_benchmark_dict()[suite]()
        return self._suites[suite]

    def make(self, suite: str, task_id: int, *, resolution: int = 256, seed: int = 0):
        from experiments.robot.libero.libero_utils import get_libero_env

        task = self._suite(suite).get_task(task_id)
        env, description = get_libero_env(task, self.model_family, resolution=resolution)
        env.seed(seed)  # override the repo's hardcoded env.seed(0)
        return env, description

    def dummy_action(self) -> list[float]:
        from experiments.robot.libero.libero_utils import get_libero_dummy_action

        return get_libero_dummy_action(self.model_family)

    def initial_states(self, suite: str, task_id: int, n: int) -> np.ndarray:
        """Deterministic starts.

        The benchmark API and FastWAM's ``torch.load(init_states_file)`` read the
        same ``.pruned_init`` tensor; ``embodiq.core.suites`` asserts that. Here
        we use the benchmark API and cycle if fewer states exist than trials,
        matching FastWAM's behaviour so the two protocols stay comparable.
        """
        states = self._suite(suite).get_task_init_states(task_id)
        if len(states) >= n:
            return states[:n]
        reps = int(np.ceil(n / len(states)))
        return np.concatenate([states] * reps, axis=0)[:n]

    def task_description(self, suite: str, task_id: int) -> str:
        return self._suite(suite).get_task(task_id).language

    def n_tasks(self, suite: str) -> int:
        return self._suite(suite).n_tasks
