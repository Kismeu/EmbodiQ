"""Native config integration on CPU; GPU parity is a separate explicit test."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "FastWAM"
CKPT = REPO / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STATS = REPO / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"


@pytest.fixture
def policy():
    pytest.importorskip("fastwam")
    from embodiq.backends.fastwam.policy import FastWAMPolicy
    return FastWAMPolicy(CKPT, "libero_spatial", repo_root=REPO, dataset_stats=STATS)


def test_native_config_and_contract(policy):
    assert (policy.input_h, policy.input_w) == (224, 448)
    assert policy.spec.chunk_size == 32
    assert policy.spec.default_replan_steps == 10
    assert policy.cfg.EVALUATION.sigma_shift == 5.0
    assert policy.cfg.model.redirect_common_files is False
    assert policy.cfg.model.skip_dit_load_from_pretrain is True
    assert policy.cfg.model.load_text_encoder is True
    assert policy.cfg.EVALUATION.compile_action_infer is False


def test_actions_keep_native_values_and_execute_prefix(policy):
    actions = np.arange(224, dtype=np.float32).reshape(32, 7) / 224
    # Already-native outputs must not undergo a second normalization/gripper flip.
    policy.native = SimpleNamespace(_predict_action_chunk=lambda **kw: (actions.copy(), {}, None))
    chunk = policy.act(SimpleNamespace(raw={"sentinel": 1}, task_description="pick"))
    np.testing.assert_array_equal(chunk.actions, actions)
    assert chunk.num_valid == 10


def test_bad_native_horizon_rejected(policy):
    policy.native = SimpleNamespace(_predict_action_chunk=lambda **kw: (np.zeros((10, 7)), {}, None))
    with pytest.raises(ValueError, match="action shape"):
        policy.act(SimpleNamespace(raw={}, task_description="pick"))


def test_reset_changes_native_sampler_seed(policy):
    calls = []
    policy.native = SimpleNamespace(set_global_seed=lambda seed, **kw: calls.append(seed))
    policy.reset(task_id=1, episode_idx=2, task_description="pick", seed=17)
    assert policy.cfg.seed == 17
    assert calls == [17]


def test_config_fingerprint_is_stable(policy):
    from embodiq.backends.fastwam.policy import FastWAMPolicy
    other = FastWAMPolicy(CKPT, "libero_spatial", repo_root=REPO, dataset_stats=STATS)
    assert other.spec.extra["config_sha256"] == policy.spec.extra["config_sha256"]


def test_idm_config_selects_optional_model(policy):
    from embodiq.backends.fastwam.policy import FastWAMPolicy
    release=CKPT.parent
    idm=FastWAMPolicy(release/'libero_optional_idm_2cam224.pt','libero_spatial',
                     repo_root=REPO,dataset_stats=release/'libero_optional_idm_2cam224_dataset_stats.json',
                     action_infer_mode='idm')
    assert idm.cfg.model._target_=='fastwam.runtime.create_fastwam_optional_idm'
    assert policy.cfg.model._target_!=idm.cfg.model._target_
    assert idm.spec.extra['future_video'] is True
    assert policy.spec.extra['future_video'] is False
    assert idm.cfg.EVALUATION.action_infer_mode=='idm'


def test_common_seed_is_accepted_by_native_fastwam(policy):
    from embodiq.core.protocol import PROTOCOLS
    policy.reset(task_id=0, episode_idx=0, task_description="pick", seed=PROTOCOLS["unified.v3"].seed_base)
    assert policy.cfg.seed == 42
    with pytest.raises(ValueError, match="positive"):
        policy.reset(task_id=0, episode_idx=0, task_description="pick", seed=0)


def test_initial_states_match_native_loader(policy):
    import torch
    from libero.libero import get_libero_path
    from embodiq.backends.fastwam.env import FastWAMEnvFactory
    factory = FastWAMEnvFactory()
    task = factory._suite("libero_spatial").get_task(0)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    native = np.asarray(torch.load(path, map_location="cpu", weights_only=False))
    actual = factory.initial_states("libero_spatial", 0, len(native))
    np.testing.assert_array_equal(actual, native)
    with pytest.raises(ValueError, match="distinct"):
        factory.initial_states("libero_spatial", 0, len(native) + 1)
