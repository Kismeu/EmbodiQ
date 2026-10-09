"""FastWAM native LIBERO construction behind the common EnvFactory API."""
from embodiq.backends.prismatic_env import PrismaticEnvFactory


class FastWAMEnvFactory(PrismaticEnvFactory):
    def make(self, suite, task_id, *, resolution=256, seed=0):
        from experiments.libero.libero_utils import get_libero_env
        return get_libero_env(self._suite(suite).get_task(task_id), resolution, seed)

    def dummy_action(self):
        from experiments.libero.libero_utils import get_libero_dummy_action
        return get_libero_dummy_action()
