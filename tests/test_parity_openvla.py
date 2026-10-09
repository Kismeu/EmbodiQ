"""GPU parity against the native OpenVLA query path on saved raw observations.

Run in the OpenVLA conda environment with EMBODIQ_OPENVLA_CKPT set.
The raw fixture originates from LIBERO, not OpenVLA reference actions; reference
actions are independently computed here using the native evaluation sequence.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = os.environ.get("EMBODIQ_OPENVLA_CKPT")
pytestmark = pytest.mark.skipif(not CKPT, reason="needs explicit OpenVLA checkpoint and GPU")


def test_native_and_adapter_actions_identical():
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    repo = ROOT.parent / "OpenVLA"
    sys.path.insert(0, str(repo))
    from experiments.robot import robot_utils
    assert Path(robot_utils.__file__).resolve().is_relative_to(repo.resolve())
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from embodiq.backends.openvla.policy import OpenVLAPolicy
    from embodiq.core.rollout import build_observation

    with (ROOT / "tests/fixtures/adapter/libero_spatial_task0.pkl").open("rb") as f:
        records = pickle.load(f)
    policy = OpenVLAPolicy(CKPT, "libero_spatial")
    try:
        policy.load()
        for record in records[:3]:
            raw = record["raw"]
            native_obs = {
                "full_image": get_libero_image(raw, 224),
                "state": np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]),
                                         raw["robot0_gripper_qpos"])),
            }
            robot_utils.set_seed_everywhere(7)
            with torch.no_grad():
                native = robot_utils.get_action(policy.cfg, policy.model, native_obs,
                                                record["task_description"], processor=policy.processor)
            native = robot_utils.invert_gripper_action(
                robot_utils.normalize_gripper_action(native, binarize=True))
            robot_utils.set_seed_everywhere(7)
            adapted = policy.act(build_observation(raw, 0, record["task_description"]))
            assert adapted.num_valid == 1
            np.testing.assert_array_equal(adapted.actions, np.asarray(native, dtype=np.float32)[None])
    finally:
        policy.unload()
