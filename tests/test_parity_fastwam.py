"""Opt-in GPU parity; requires the local uncond checkpoint and Wan components."""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("EMBODIQ_FASTWAM_GPU_TEST") != "1",
                                reason="explicit GPU opt-in required")


def test_native_and_adapter_action_chunks_identical():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from embodiq.backends.fastwam.policy import FastWAMPolicy
    from embodiq.core.rollout import build_observation
    repo = ROOT.parent / "FastWAM"
    base = repo / "checkpoints/fastwam_release"
    policy = FastWAMPolicy(base / "libero_uncond_2cam224.pt", "libero_spatial", repo_root=repo,
                           dataset_stats=base / "libero_uncond_2cam224_dataset_stats.json", seed=42)
    with (ROOT / "tests/fixtures/adapter/libero_spatial_task0.pkl").open("rb") as f:
        records = pickle.load(f)
    try:
        policy.load()
        for record in records[:3]:
            raw, desc = record["raw"], record["task_description"]
            policy.reset(task_id=0, episode_idx=0, task_description=desc, seed=42)
            expected, _, _ = policy.native._predict_action_chunk(
                raw, desc, policy.model, policy.processor, policy.cfg,
                action_horizon=32, input_h=224, input_w=448, model_device="cuda:0")
            policy.reset(task_id=0, episode_idx=0, task_description=desc, seed=42)
            chunk = policy.act(build_observation(raw, 0, desc))
            assert chunk.actions.shape == (32, 7)
            assert chunk.num_valid == 10
            np.testing.assert_array_equal(chunk.actions, np.asarray(expected, dtype=np.float32))
    finally:
        policy.unload()
