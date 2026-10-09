"""VLA-Adapter backend.

Every preprocessing step is the repo's own function, called unmodified. That is
what makes the V1 bit-exactness test possible: reimplementing the 180-degree
rotate, the lanczos resize, the center crop, the Qwen chat template or the
gripper transform would silently change the policy, and the resulting quantized
-vs-FP16 deltas would be measuring our bug instead of quantization.

Two model-specific facts drive the shape of this class:
  * quantization has THREE roots -- the HF model plus the action head and the
    proprio projector, which are loose .pt modules outside it;
  * one query returns 8 actions and all 8 are executed (chunk == replan here,
    unlike FastWAM where 32 are predicted and 10 executed).
"""

from __future__ import annotations

# MUST precede any import that pulls in tensorflow: both libero_utils and
# openvla_utils import it at module scope for an image resize, and TF 2.15
# claims the whole GPU on first use.
from embodiq.backends.compat import tf_cpu_guard

tf_cpu_guard.apply()

from types import SimpleNamespace  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402

from embodiq.core.types import (  # noqa: E402
    ActionChunk, GripperConvention, MemoryReport, Observation, Policy, PolicySpec,
)

SUITE_TO_CKPT = {
    "libero_spatial": "LIBERO-Spatial-Pro",
    "libero_object": "LIBERO-Object-Pro",
    "libero_goal": "LIBERO-Goal-Pro",
    "libero_10": "LIBERO-Long-Pro",
}


def make_cfg(checkpoint: str, suite: str, **over: Any) -> SimpleNamespace:
    """A stand-in for the repo's draccus GenerateConfig.

    We never invoke draccus: it would parse sys.argv, and the harness owns the
    command line. Field names and defaults mirror
    VLA-Adapter/experiments/robot/libero/run_libero_eval.py:82-129.
    """
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_minivlm=True,
        num_diffusion_steps=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        unnorm_key="",
        load_in_8bit=False,
        load_in_4bit=False,
        task_suite_name=suite,
        env_img_res=256,
        seed=7,
        save_version="vla-adapter",
        use_pro_version=True,
        phase="Inference",
        lora_rank=32,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class VLAAdapterPolicy(Policy):
    def __init__(
        self,
        checkpoint: str,
        suite: str,
        *,
        device_index: int = 0,
        quant_apply=None,
        replan_steps: int = 8,
        freeze_checkpoint: bool = True,
        **cfg_over: Any,
    ) -> None:
        self.cfg = make_cfg(checkpoint, suite, **cfg_over)
        self.device_index = device_index
        self.quant_apply = quant_apply
        self.freeze_checkpoint = freeze_checkpoint
        self._probe = None
        self.spec = PolicySpec(
            model_id="vla-adapter-pro" if self.cfg.use_pro_version else "vla-adapter",
            family="vla_adapter",
            checkpoint=checkpoint,
            cameras=("agentview", "wrist"),
            env_resolution=256,
            uses_proprio=True,
            chunk_size=8,
            default_replan_steps=replan_steps,
            gripper=GripperConvention.ENV_NATIVE_BINARIZED,
            torch_dtype="bfloat16",
            extra={"use_minivlm": True, "num_open_loop_steps": 8,
                   "device_index": device_index},
        )
        self.model = self.action_head = self.proprio_projector = self.processor = None

    def quantization_roots(self) -> dict[str, object]:
        """Expose all roots that belong to this policy.

        The action head and proprio projector are loose modules outside the HF
        model.  Keeping this mapping on the adapter lets RTN/AWQ/HSA share one
        quantizer API without embedding model-specific root logic in the
        quantization backends.
        """
        if self.model is None:
            raise RuntimeError("load the policy before requesting quantization roots")
        roots: dict[str, object] = {"model": self.model}
        if self.action_head is not None:
            roots["action_head"] = self.action_head
        if self.proprio_projector is not None:
            roots["proprio_projector"] = self.proprio_projector
        return roots

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        from embodiq.backends.compat import adapter_freeze
        from embodiq.core.memory import MemoryProbe
        from experiments.robot.libero import run_libero_eval as rle
        from experiments.robot import openvla_utils

        # Neutralize the two functions that rewrite the checkpoint directory on
        # every load. With four workers on one checkpoint that is a write race.
        adapter_freeze.apply(openvla_utils)
        if self.freeze_checkpoint:
            try:
                adapter_freeze.verify_manifest(self.cfg.pretrained_checkpoint)
            except FileNotFoundError:
                adapter_freeze.write_manifest(self.cfg.pretrained_checkpoint)

        self._probe = MemoryProbe(self.device_index)
        self._probe.mark_before_load()
        self._probe.start()

        (self.model, self.action_head, self.proprio_projector,
         _noisy, self.processor) = rle.initialize_model(self.cfg)

        # Checkpoint FIRST, quantization SECOND -- always. Swapping modules before
        # a load renames weight -> packed/scales_zeros, and a strict=False load
        # then silently drops every quantized tensor and leaves random init.
        if self.quant_apply is not None:
            self.quant_report = self.quant_apply(self.quantization_roots(), "vla_adapter")

        self._probe.mark_after_load()

    def reset(self, *, task_id: int, episode_idx: int, task_description: str, seed: int) -> None:
        # Stateless between episodes: the action queue lives in the runner, and
        # the repo does pure open-loop chunk replay with no temporal ensembling.
        return None

    def memory_footprint(self) -> MemoryReport:
        rep = self._probe.report() if self._probe else MemoryReport()
        rep.device_index = self.device_index
        return rep

    def unload(self) -> None:
        """Free the model and empty the allocator cache.

        A worker drains many (model, quant) pairs in one process. If the previous
        model is still resident when the next one calls mark_before_load(), the
        baseline is polluted and weights_bytes comes out wrong -- it can even go
        negative if the old model happens to be collected mid-load. That single
        number is the evidence that INT4 packing worked, so it has to be clean.
        """
        import gc

        import torch

        if self._probe is not None:
            self._probe.stop()
        self.model = self.action_head = self.proprio_projector = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device_index)

    # -- inference -------------------------------------------------------
    def act(self, obs: Observation) -> ActionChunk:
        import torch
        from experiments.robot.libero import run_libero_eval as rle
        from experiments.robot.robot_utils import get_action

        # 180-degree rotate + 256->224 resize for both cameras, plus the 8-dim
        # proprio vector -- all from the repo.
        observation, _img = rle.prepare_observation(obs.raw, 224)

        with torch.no_grad():
            actions = get_action(
                self.cfg, self.model, observation, obs.task_description,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=None,
                use_film=self.cfg.use_film,
                use_minivlm=self.cfg.use_minivlm,
            )
        if isinstance(actions, np.ndarray) and actions.ndim == 1:
            actions = [actions]
        # normalize_gripper_action(binarize=True) then invert_gripper_action
        processed = [rle.process_action(a, self.cfg.model_family) for a in actions]
        arr = np.stack(processed).astype(np.float32)
        return ActionChunk(actions=arr, num_valid=min(len(processed), self.spec.default_replan_steps))
