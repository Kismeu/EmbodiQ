"""FastWAM's temporal-prediction switch, tested without the 4.9 B video DiT.

Task 1.4 lists temporal prediction among the five interfaces to unify, and
FastWAM is the only model that has one. The risk is not that the switch fails
loudly -- it is that it fails silently: `eval_libero_single` forwards
`action_infer_mode` only when the loaded class declares it, so asking the uncond
checkpoint for `idm` would run first-frame inference and label it idm.

These tests pin the guard, not the inference, so they need no checkpoint: the
capability check reads `inspect.signature(model.infer_action)`, which a stand-in
can reproduce exactly.
"""

from __future__ import annotations

import sys
import types
from typing import Optional

import pytest


def load_policy_module():
    """Import the policy module with the FastWAM repo's imports stubbed out.

    The module imports numpy and embodiq.core.types at top level only; the
    native repo is touched inside functions, so nothing needs faking here.
    """
    from embodiq.backends.fastwam import policy

    return policy


class UncondModel:
    """Base FastWAM: first-frame only, no mode kwarg at all."""

    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     num_inference_steps: int = 20, seed: Optional[int] = None):
        raise AssertionError("not called in these tests")


class OptionalIdmModel:
    """OptionalIDM: both modes, dispatched by kwarg."""

    def infer_action(self, prompt, input_image, action_horizon,
                     num_video_frames=None, proprio=None,
                     num_inference_steps: int = 20, seed: Optional[int] = None,
                     action_infer_mode: str = "idm"):
        raise AssertionError("not called in these tests")


class IdmWithoutFrames:
    """Declares the mode but not num_video_frames -- idm cannot work."""

    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     action_infer_mode: str = "idm"):
        raise AssertionError("not called in these tests")


def make_policy(mode, model):
    """A FastWAMPolicy shell with just enough state for the capability check."""
    policy = load_policy_module()
    obj = object.__new__(policy.FastWAMPolicy)
    obj.action_infer_mode = mode
    obj.model = model
    from embodiq.core.types import GripperConvention, PolicySpec

    obj.spec = PolicySpec(
        model_id=f"fastwam-{mode.replace('_', '-')}", family="fastwam",
        checkpoint="/dev/null", cameras=("agentview", "wrist"), env_resolution=256,
        uses_proprio=True, chunk_size=32, default_replan_steps=10,
        gripper=GripperConvention.ENV_NATIVE_BINARIZED, torch_dtype="bfloat16",
        extra={},
    )
    return obj


def test_mode_list_is_the_two_native_modes():
    policy = load_policy_module()
    assert policy.ACTION_INFER_MODES == ("first_frame", "idm")


def test_idm_on_the_uncond_checkpoint_is_refused(caplog):
    """The silent-mislabel case: the kwarg is dropped and first-frame runs."""
    obj = make_policy("idm", UncondModel())
    with pytest.raises(RuntimeError, match="cannot run temporal prediction"):
        obj._assert_mode_supported()


def test_idm_on_the_optional_idm_checkpoint_is_accepted():
    obj = make_policy("idm", OptionalIdmModel())
    obj._assert_mode_supported()
    assert obj.spec.extra["native_dispatch"] == "explicit_kwarg"
    assert obj.spec.extra["infer_action_class"] == "OptionalIdmModel"


def test_idm_without_num_video_frames_is_refused():
    obj = make_policy("idm", IdmWithoutFrames())
    with pytest.raises(RuntimeError, match="num_video_frames"):
        obj._assert_mode_supported()


def test_first_frame_on_the_uncond_checkpoint_records_how_it_was_achieved():
    """Correct, but for a different reason than on OptionalIDM: the base class
    IS first-frame, so no kwarg is sent. The result file says which."""
    obj = make_policy("first_frame", UncondModel())
    obj._assert_mode_supported()
    assert obj.spec.extra["native_dispatch"] == "absent_kwarg_base_class_is_first_frame"


def test_first_frame_on_the_optional_idm_checkpoint_is_an_explicit_kwarg():
    obj = make_policy("first_frame", OptionalIdmModel())
    obj._assert_mode_supported()
    assert obj.spec.extra["native_dispatch"] == "explicit_kwarg"


def test_model_id_separates_the_two_modes():
    """An idm row and a first-frame row are different policies and must never
    collapse into one matrix row."""
    assert (make_policy("idm", OptionalIdmModel()).spec.model_id
            != make_policy("first_frame", UncondModel()).spec.model_id)


def test_unknown_mode_is_rejected_at_construction():
    policy = load_policy_module()
    with pytest.raises(ValueError, match="action_infer_mode must be one of"):
        policy.FastWAMPolicy("/dev/null", "libero_spatial", repo_root="/dev/null",
                             dataset_stats="/dev/null", action_infer_mode="joint")


def test_compose_config_sets_the_native_key():
    """The native dispatch reads cfg.EVALUATION.action_infer_mode; assert the
    name here so a rename upstream fails a test instead of silently defaulting
    to 'idm'."""
    import inspect

    policy = load_policy_module()
    source = inspect.getsource(policy.compose_config)
    assert "cfg.EVALUATION.action_infer_mode = action_infer_mode" in source
    assert "cfg.EVALUATION.visualize_future_video = False" in source
