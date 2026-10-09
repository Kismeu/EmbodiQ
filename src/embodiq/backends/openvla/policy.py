"""OpenVLA backend using the repository's native preprocessing and decoding."""
from __future__ import annotations

from embodiq.backends.compat import tf_cpu_guard
tf_cpu_guard.apply()

import gc
from types import SimpleNamespace
import numpy as np

from embodiq.core.types import ActionChunk, GripperConvention, MemoryReport, Policy, PolicySpec


class OpenVLAPolicy(Policy):
    def __init__(self, checkpoint, suite, *, device_index=0, quant_apply=None):
        if device_index != 0:
            raise ValueError("Native OpenVLA uses cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES")
        self.cfg = SimpleNamespace(model_family="openvla", pretrained_checkpoint=str(checkpoint),
                                   load_in_8bit=False, load_in_4bit=False, center_crop=True,
                                   unnorm_key=suite)
        self.quant_apply = quant_apply
        self.model = self.processor = self._probe = None
        self.spec = PolicySpec(model_id="openvla-7b", family="openvla", checkpoint=str(checkpoint),
                               cameras=("agentview",), env_resolution=256, uses_proprio=False,
                               chunk_size=1, default_replan_steps=1,
                               gripper=GripperConvention.ENV_NATIVE_BINARIZED,
                               torch_dtype="bfloat16", extra={"device_index": device_index})

    def quantization_roots(self) -> dict[str, object]:
        """Expose the loaded OpenVLA module through the common quantizer API."""
        if self.model is None:
            raise RuntimeError("load the policy before requesting quantization roots")
        return {"model": self.model}

    def load(self):
        from embodiq.core.memory import MemoryProbe
        from experiments.robot.robot_utils import get_model
        from experiments.robot.openvla_utils import get_processor
        self._probe = MemoryProbe(0)
        self._probe.mark_before_load()
        self._probe.start()
        try:
            self.model = get_model(self.cfg)
            key = self.cfg.unnorm_key
            if key not in self.model.norm_stats:
                key += "_no_noops"
            if key not in self.model.norm_stats:
                raise ValueError(f"Missing action normalization statistics: {self.cfg.unnorm_key}")
            self.cfg.unnorm_key = key
            self.processor = get_processor(self.cfg)
            if self.quant_apply is not None:
                self.quant_report = self.quant_apply(self.quantization_roots(), "openvla")
            self.model.eval()
            self._probe.mark_after_load()
        except Exception:
            self.unload()
            raise

    def reset(self, *, task_id, episode_idx, task_description, seed):
        # RNG seeding is owned by the common rollout, as for VLA-Adapter.
        pass

    def act(self, obs):
        import torch
        from experiments.robot.libero.libero_utils import get_libero_image
        from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action
        observation = {"full_image": get_libero_image(obs.raw, 224), "state": obs.proprio}
        with torch.inference_mode():
            action = get_action(self.cfg, self.model, observation, obs.task_description,
                                processor=self.processor)
        action = invert_gripper_action(normalize_gripper_action(action, binarize=True))
        return ActionChunk(np.asarray(action, dtype=np.float32).reshape(1, 7), num_valid=1)

    def memory_footprint(self):
        return self._probe.report() if self._probe else MemoryReport()

    def unload(self):
        import torch
        if self._probe is not None:
            self._probe.stop()
        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
