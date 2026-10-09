# Stage A: unified policy contract

This stage freezes the boundary between the LIBERO runner and model backends.
The runner consumes raw simulator observations and receives an action chunk; it
does not know a model's prompt template, image transform, normalization or
gripper convention.

## Native contracts

| family | raw cameras | proprio | returned chunk | native executed prefix |
|---|---|---:|---:|---:|
| `openvla` | `agentview` | no | 1 | 1 |
| `vla_adapter` | `agentview`, `wrist` | yes, 8 dims | 8 | 8 |
| `fastwam` | `agentview`, `wrist` | yes, 8 dims | 32 | 10 |

The formal table is `embodiq.core.types.MODEL_CONTRACTS`.

## Runtime boundary

```python
observation = Observation(
    images=raw_images,
    proprio=raw_proprio,
    raw=raw_robosuite_observation,
    t=env_step,
    task_description=instruction,
)
chunk = policy(observation)
```

`Policy.act()` means one model query. `ActionChunk.num_valid` tells the common
rollout how many leading actions may be executed. This is required for FastWAM:
the model returns 32 actions but the native protocol executes 10 before the
next query.

The following fields are validated at construction time:

- images are HWC `uint8` RGB arrays;
- proprio is finite numeric shape `(8,)`;
- actions are finite floating arrays of shape `(K, 7)`;
- `1 <= num_valid <= K`;
- a known model family cannot silently change its camera set, chunk size or
  action dimension.

An experiment may override the replan cadence for an isolation row, but it may
not change the model's returned chunk size.

## Current gate

The no-checkpoint gate is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  python -m pytest -q tests/test_contract_and_rollout.py \
    tests/test_rollout_error_surfacing.py
```

It verifies the known contracts, rejects malformed observations/actions and
executes a mock FastWAM rollout with 32 predicted actions and only 10 actions
executed per query.

The VLA-Adapter FP16 path remains the golden implementation and is covered by
`tests/test_parity_adapter.py`. Its actual parity test still requires the
captured fixture, checkpoint and CUDA.

## Test gate as of 2026-10-04

The no-checkpoint gate above still holds. The full suite is now:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q tests/
# 110 passed, 9 skipped   (skips need a checkpoint and/or CUDA)
```

Three suites were added for the pieces the original gate could not reach:

| File | Tests | What it pins |
|---|---:|---|
| `test_quant_artifact.py` | 33 | the AWQ/HSA artifact contract: a relabelled method, a different module range, a different checkpoint, and a half-applied swap are each made impossible |
| `test_controller_status.py` | 22 | controller durability: atomic status writes, a status failure that cannot kill the run, a crash that cannot leave `phase=running`, task-level resume, and a clean stop |
| `test_fastwam_temporal_mode.py` | 9 | the temporal-prediction switch, including the case where the uncond checkpoint silently ignores `action_infer_mode` |

The contract itself gained one field since this document was written: FastWAM's
temporal prediction is selected explicitly with `--fastwam-mode
{first_frame,idm}` and asserted against the loaded class after load, because the
native dispatch forwards the kwarg only when the class declares it — so asking
the uncond checkpoint for `idm` would otherwise run first-frame and label it
idm. `PolicySpec.model_id` carries the mode, so the two are never one table row.
