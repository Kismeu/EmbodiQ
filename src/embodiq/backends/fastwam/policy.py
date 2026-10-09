"""FastWAM policy over the verified native query implementation.

Temporal prediction is one of the five interfaces task 1.4 has to unify, and it
is the only one FastWAM actually has: the OptionalIDM checkpoint can denoise
future video frames with the 4.9 B video DiT before reading an action out of
them (``action_infer_mode="idm"``), or skip that and read the action from the
first frame alone (``"first_frame"``). Same weights, runtime switch.

The native dispatch defaults to ``"idm"`` and passes the kwarg only when the
loaded class's ``infer_action`` happens to declare it, so the uncond checkpoint
silently runs first-frame no matter what the config says. That is the same shape
of fault as a quantization method running under another method's label, so the
mode is explicit here and the capability is asserted after load.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys

import numpy as np

from embodiq.core.types import ActionChunk, GripperConvention, MemoryReport, Policy, PolicySpec

ACTION_INFER_MODES = ("first_frame", "idm")


def native_module(repo):
    repo = Path(repo).resolve()
    # Native eval imports action_ensembler as a top-level module.
    sys.path.insert(0, str(repo / "experiments" / "libero"))
    sys.path.insert(0, str(repo))
    module = importlib.import_module("experiments.libero.eval_libero_single")
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("FastWAM native code was imported from another repository")
    return module


def compose_config(repo, checkpoint, stats, suite, *, seed=42, compile_infer=False,
                   action_infer_mode="first_frame"):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    # Importing the native module registers its OmegaConf resolvers.
    native_module(repo)
    task = ("libero_optional_idm_2cam224_1e-4" if action_infer_mode == "idm"
            else "libero_uncond_2cam224_1e-4")
    with initialize_config_dir(config_dir=str(Path(repo).resolve() / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=[f"task={task}"])
    cfg.ckpt = str(Path(checkpoint).resolve())
    cfg.seed = seed
    cfg.model.redirect_common_files = False
    cfg.EVALUATION.dataset_stats_path = str(Path(stats).resolve())
    cfg.EVALUATION.task_suite_name = suite
    cfg.EVALUATION.sigma_shift = 5.0
    cfg.EVALUATION.compile_action_infer = compile_infer
    # infer_joint (visualize_future_video) is a different thing: it needs
    # model.video_dit_config.action_conditioned=false and writes video out. The
    # temporal switch this policy exposes is action_infer_mode on infer_action.
    # FastWAM's released YAML is struct-locked and does not declare the
    # adapter-only temporal switch.  Add it through OmegaConf's supported
    # escape hatch so the unified adapter can record and pass it later.
    with open_dict(cfg.EVALUATION):
        cfg.EVALUATION.visualize_future_video = False
        cfg.EVALUATION.action_infer_mode = action_infer_mode
        cfg.EVALUATION.use_action_ensembler = False
        cfg.EVALUATION.device = "cuda:0"
    return cfg


class FastWAMPolicy(Policy):
    def __init__(self, checkpoint, suite, *, repo_root, dataset_stats, device_index=0,
                 quant_apply=None, compile_infer=False, seed=42,
                 action_infer_mode="first_frame"):
        if action_infer_mode not in ACTION_INFER_MODES:
            raise ValueError(f"action_infer_mode must be one of {ACTION_INFER_MODES}")
        if not 0 < seed < np.iinfo(np.uint32).max:
            raise ValueError("Native FastWAM requires 0 < seed < uint32 max; use unified.v3")
        if device_index != 0:
            raise ValueError("Select a physical GPU with CUDA_VISIBLE_DEVICES; use logical device 0")
        self.repo = Path(repo_root).resolve()
        self.action_infer_mode = action_infer_mode
        self.cfg = compose_config(self.repo, checkpoint, dataset_stats, suite,
                                  seed=seed, compile_infer=compile_infer,
                                  action_infer_mode=action_infer_mode)
        from omegaconf import OmegaConf
        self.native = native_module(self.repo)
        self.action_horizon = int(self.cfg.data.train.num_frames) - 1
        self.input_h, self.input_w = map(int, self.cfg.data.train.video_size)
        self.quant_apply = quant_apply
        self.model = self.processor = self._probe = None
        self.spec = PolicySpec(
            # The mode is part of the identity: an idm row and a first-frame row
            # are different policies, and must never merge into one table row.
            model_id=f"fastwam-{action_infer_mode.replace('_', '-')}",
            family="fastwam", checkpoint=str(Path(checkpoint).resolve()),
            cameras=("agentview", "wrist"), env_resolution=256, uses_proprio=True,
            chunk_size=self.action_horizon, default_replan_steps=int(self.cfg.EVALUATION.replan_steps),
            gripper=GripperConvention.ENV_NATIVE_BINARIZED, torch_dtype="bfloat16",
            extra={"device_index": 0, "dataset_stats": str(Path(dataset_stats).resolve()),
                   "dataset_stats_sha256": hashlib.sha256(Path(dataset_stats).read_bytes()).hexdigest(),
                   "config_sha256": hashlib.sha256(OmegaConf.to_yaml(OmegaConf.create({
                       "model": OmegaConf.to_container(self.cfg.model, resolve=True),
                       "processor": OmegaConf.to_container(self.cfg.data.train.processor, resolve=True),
                       "evaluation": {k: self.cfg.EVALUATION[k] for k in self.cfg.EVALUATION
                                      if k != "output_dir"},
                       "seed": seed, "horizon": self.action_horizon,
                   })).encode()).hexdigest(),
                   "sigma_shift": 5.0, "num_inference_steps": int(self.cfg.EVALUATION.num_inference_steps),
                   "compile_action_infer": compile_infer, "input_hw": [self.input_h, self.input_w],
                   "action_infer_mode": action_infer_mode,
                   "future_video": action_infer_mode == "idm", "redirect_common_files": False})

    def quantization_roots(self) -> dict[str, object]:
        """Expose the loaded FastWAM module through the common quantizer API."""
        if self.model is None:
            raise RuntimeError("load the policy before requesting quantization roots")
        return {"model": self.model}

    def load(self):
        from hydra.utils import instantiate
        from embodiq.core.memory import MemoryProbe
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(self.repo / "checkpoints")
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        self.native.set_global_seed(int(self.cfg.seed), get_worker_init_fn=False)
        self._probe = MemoryProbe(0)
        self._probe.mark_before_load()
        self._probe.start()
        try:
            dtype = self.native._mixed_precision_to_model_dtype(self.cfg.mixed_precision)
            self.model = instantiate(self.cfg.model, model_dtype=dtype, device="cuda:0")
            self.native._load_model_checkpoint(self.model, self.cfg.ckpt)
            self.model = self.model.to("cuda:0").eval()
            self.processor = instantiate(self.cfg.data.train.processor).eval()
            self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(self.cfg.EVALUATION.dataset_stats_path))
            self._assert_mode_supported()
            if self.quant_apply is not None:
                self.quant_report = self.quant_apply(self.quantization_roots(), "fastwam")
            self._probe.mark_after_load()
        except Exception:
            self.unload()
            raise

    def _assert_mode_supported(self):
        """Refuse to report a mode the loaded checkpoint cannot actually run.

        ``eval_libero_single`` passes ``action_infer_mode`` only when the loaded
        class declares it, so asking the uncond checkpoint for ``idm`` would run
        first-frame and label it idm. Which way the mode was achieved is recorded
        either way, so a result file never has to be taken on trust.
        """
        params = inspect.signature(self.model.infer_action).parameters
        supported = "action_infer_mode" in params
        if self.action_infer_mode == "idm" and not supported:
            raise RuntimeError(
                f"{type(self.model).__name__}.infer_action has no "
                f"`action_infer_mode`, so this checkpoint cannot run temporal "
                f"prediction. The uncond checkpoint is first-frame only; idm "
                f"needs the optional-IDM weights and their dataset stats."
            )
        if self.action_infer_mode == "idm" and "num_video_frames" not in params:
            raise RuntimeError(
                f"{type(self.model).__name__}.infer_action takes no "
                f"`num_video_frames`; idm cannot denoise future frames without it"
            )
        self.spec.extra["native_dispatch"] = (
            "explicit_kwarg" if supported else "absent_kwarg_base_class_is_first_frame")
        self.spec.extra["infer_action_class"] = type(self.model).__name__

    def reset(self, *, task_id, episode_idx, task_description, seed):
        if not 0 < seed < np.iinfo(np.uint32).max:
            raise ValueError("Native FastWAM requires a positive uint32 seed")
        self.cfg.seed = seed
        self.native.set_global_seed(seed, get_worker_init_fn=False)
        # Action queues belong to the shared runner. Native inference recreates
        # video KV caches per query and uses the seed passed through cfg.

    def act(self, obs):
        action, _images, _video = self.native._predict_action_chunk(
            obs=obs.raw, task_description=obs.task_description, model=self.model,
            processor=self.processor, cfg=self.cfg, action_horizon=self.action_horizon,
            input_w=self.input_w, input_h=self.input_h, model_device="cuda:0")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_horizon, 7):
            raise ValueError(f"Unexpected FastWAM action shape: {action.shape}")
        return ActionChunk(action, num_valid=self.spec.default_replan_steps)

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
