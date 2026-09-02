# Stage A deliverable: the `Policy` contract, validated against all three signatures

Purpose: pin the abstraction against the model that is actually *different*
before writing any backend. VLA-Adapter and OpenVLA are the same repo (both
Prismatic-derived, same `experiments/robot/` utilities), so designing against
those two validates a diff, not an abstraction. FastWAM is the only genuinely
different one, and it is the one that breaks naive designs.

## The three native entry points, verbatim

```python
# openvla/experiments/robot/robot_utils.py:63
get_action(cfg, model, obs, task_label, processor=None) -> np.ndarray  # (7,)

# VLA-Adapter/experiments/robot/robot_utils.py:99
get_action(cfg, model, obs, task_label, processor=None, action_head=None,
           proprio_projector=None, noisy_action_projector=None,
           use_film=False, use_minivlm=False) -> Union[List[np.ndarray], np.ndarray]

# FastWAM/src/fastwam/models/wan22/fastwam.py:979  (+ _idm.py:365, _optional_idm.py:60)
infer_action(prompt, input_image, action_horizon,
             [num_video_frames,]            # required on IDM, optional on OptionalIDM, absent on base
             proprio=None, context=None, context_mask=None, negative_prompt=None,
             text_cfg_scale=1.0, num_inference_steps=20, sigma_shift=None,
             seed=None, rand_device="cpu", tiled=False, compile_action_infer=False,
             [action_infer_mode="idm",]      # OptionalIDM only
             ) -> dict   # {"action": Tensor[T, D]} (+ "video_latents" on IDM)
```

The FastWAM signature **varies by subclass**, and the repo itself dispatches with
`inspect.signature` (`eval_libero_single.py:410-424`). Any wrapper that hardcodes
kwargs silently drops them. Our backend does the same introspection.

## Interface-by-interface binding

| Interface | OpenVLA | VLA-Adapter | FastWAM |
|---|---|---|---|
| **Image** | 1 cam (agentview), 180-deg rotate, TF lanczos 256->224, optional center-crop 0.9 | 2 cams, both rotated, 256->224, `pixel_values` concatenated on dim=1 -> (1,12,224,224) | 2 cams concatenated **horizontally** -> 224x448, scaled to [-1,1] |
| **Language** | `"In: What action should the robot take to {task}?\nOut:"`, tokenized into the LLM **every step** | Qwen chat template, into the LLM **every step** | `"A video recorded from a robot's point of view executing the following instruction: {task}"` -> UMT5-XXL. **Embeddings are precomputable offline; at rollout the text encoder runs zero times** |
| **Robot state** | computed by the eval loop, **never consumed** | 8-dim, q01/q99-normalized, via `ProprioProjector` | 8-dim, projected to 4096 and appended as one extra T5 cross-attention token |
| **Action output** | 7 discrete tokens = the last 256 ids of the Llama vocab; `generate(max_new_tokens=7)`; correctness is pure **argmax ranking** | one forward, all 25 hidden layers -> L1-regression MLPResNet -> (8,7) | flow-matching, 10 Euler steps -> (32,7) |
| **Temporal prediction** | none | none (open-loop chunk replay, no ensembling) | **yes** -- IDM mode denoises 9 future frames with the 4.9B video DiT first; first-frame mode skips it. Same weights, runtime switch |
| **Quantization roots** | 1 (single HF model) | **2+** (HF model + loose `.pt` action head and proprio projector) | **4** (video DiT, ActionDiT, UMT5, VAE) |
| **Chunk vs executed** | 1 / 1 | 8 / 8 | **32 / 10** |

## What the contract must therefore provide

1. **`act()` is one QUERY, not one env step**, and returns `num_valid` separately
   from `len(actions)`. FastWAM is the only model where these differ; a design
   that conflates them works on two of three models and is unfixable later
   without re-running.
2. **Latency is two numbers, not one.** Query density is 1 : 1/8 : 1/10, so
   per-query and per-env-step differ by 10x across the table. Only
   `per_env_step_s` is comparable. This is the single field that, if gotten
   wrong in the schema, invalidates the cross-model table.
3. **Raw observations in, env-ready actions out.** Preprocessing and the gripper
   convention are model properties and stay inside the Policy, next to the
   repo functions they must match bit-for-bit.
4. **Quantization takes a list of roots**, not a model. Two of three models have
   more than one.
5. **Text-encoder cost is not query cost.** Quantizing FastWAM's UMT5 saves 4.7B
   of VRAM and exactly zero latency. Reporting it as a speedup would be wrong.

## Metric definitions (the other half of the Stage A gate)

- `success_rate = n_success / (n_episodes - n_error)`. Errors are **not**
  failures; see `EpisodeStatus`.
- `avg_success_rate` = unweighted mean of the four per-suite rates (matches all
  three papers, and matches `FastWAM/experiments/libero/summarize_results.py`).
- `query_latency_s` -- CUDA-synced around `act()`, first 5 queries excluded and
  reported separately as `compile_overhead_s`.
- `per_env_step_s = sum(query_latency) / n_env_steps`, wait steps excluded. The
  only cross-model-comparable latency figure.
- `weights_bytes` (resident params after quantization; int4/fp16 must land in
  [0.28, 0.40]) and `nvml_peak_bytes` (true device footprint). Both in the
  headline table.
- Quantization deltas use a **paired** test on identical initial states
  (`protocol.mcnemar_delta`), never two independent Wilson intervals.
  Independent Wilson (`protocol.wilson_ci95`) is only for FP16-vs-published.
