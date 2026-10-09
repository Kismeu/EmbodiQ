"""V1: the adapter must reproduce the repo's native action path BIT-FOR-BIT.

Fixtures come from tools/capture_parity_fixture.py, which drives the REPO's own
run_episode with get_action and prepare_observation monkeypatched to record. So
the reference actions were produced by the repo's cfg, its model init, its
preprocessing and its post-processing -- not by ours.

Exact equality, not allclose. A dropped 0.9 center crop, the plain prompt instead
of the Qwen chat template, a missed gripper inversion or a reordered resize would
all still produce "close" actions while silently changing the policy, and every
downstream quantization delta would then be measuring our bug.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "adapter" / "libero_spatial_task0.pkl"
CKPT = os.environ.get(
    "EMBODIQ_ADAPTER_CKPT",
    os.environ.get("EMBODIQ_CKPT_ROOT", "/data/jyc/data/VLA-Adapter") + "/LIBERO-Spatial-Pro",
)

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not Path(CKPT).exists(),
    reason="needs the captured fixture and the Spatial-Pro checkpoint",
)


@pytest.fixture(scope="module")
def policy():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import importlib.util
    if importlib.util.find_spec("experiments.robot") is None:
        pytest.skip("needs the VLA-Adapter repository on PYTHONPATH")
    from embodiq.backends.adapter.policy import VLAAdapterPolicy

    p = VLAAdapterPolicy(CKPT, "libero_spatial", device_index=0)
    p.load()
    yield p
    p.unload()


@pytest.fixture(scope="module")
def records():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def test_fixture_is_usable(records):
    assert len(records) >= 5
    r = records[0]
    assert "agentview_image" in r["raw"] and "robot0_eye_in_hand_image" in r["raw"]
    assert r["raw"]["agentview_image"].shape == (256, 256, 3)
    assert len(r["actions"]) == 8, "VLA-Adapter returns an 8-action chunk"
    assert r["actions"][0].shape == (7,)


def test_actions_are_bit_exact(policy, records):
    from embodiq.core.types import Observation
    from embodiq.core.rollout import build_observation

    mismatches = []
    for i, r in enumerate(records):
        obs = build_observation(r["raw"], t=i, task_description=r["task_description"])
        chunk = policy.act(obs)
        ref = np.stack(r["actions"]).astype(np.float32)
        got = chunk.actions[: ref.shape[0]]
        if not np.array_equal(got, ref):
            mismatches.append((i, float(np.abs(got - ref).max())))
    assert not mismatches, (
        f"{len(mismatches)}/{len(records)} queries differ from the native path; "
        f"max abs deltas: {mismatches[:5]}"
    )


def test_chunk_shape_and_gripper_convention(policy, records):
    from embodiq.core.rollout import build_observation

    obs = build_observation(records[0]["raw"], 0, records[0]["task_description"])
    chunk = policy.act(obs)
    assert chunk.actions.shape == (8, 7)
    assert chunk.num_valid == 8, "chunk == replan for VLA-Adapter (unlike FastWAM)"
    # process_action binarizes then inverts, so the gripper dim is exactly +/-1.
    assert set(np.unique(np.abs(chunk.actions[:, 6]))) <= {1.0}
