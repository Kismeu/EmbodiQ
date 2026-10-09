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

        All model factories read the same trusted local .pruned_init asset.
        Refuse excess trials rather than silently repeating initial states.
        """
        import torch
        states = np.asarray(torch.load(self.initial_state_path(suite, task_id),
                                      map_location="cpu", weights_only=False))
        if states.ndim != 2 or len(states) == 0 or not np.isfinite(states).all() or n < 1:
            raise ValueError("Invalid LIBERO initial states or trial count")
        if n > len(states):
            raise ValueError("Requested trials exceed distinct initial states; do not silently repeat")
        return states[:n]

    def initial_state_path(self, suite, task_id):
        from pathlib import Path
        from libero.libero import get_libero_path
        task = self._suite(suite).get_task(task_id)
        root = Path(get_libero_path("init_states")).resolve()
        path = (root / task.problem_folder / task.init_states_file).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Initial states must be local LIBERO assets")
        return path

    def task_description(self, suite: str, task_id: int) -> str:
        return self._suite(suite).get_task(task_id).language

    def n_tasks(self, suite: str) -> int:
        return self._suite(suite).n_tasks
