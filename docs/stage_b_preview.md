# Stage B preview: VLA-Adapter FP16 vs RTN-INT4 (scope ablation)

Real `LIBERO-Spatial-Pro` checkpoint (1.471 B params: 1.253 B backbone + 218 M
action head + 0.8 M proprio). Fidelity measured over 80 actions (10 queries x 8)
replayed from the V1 parity fixture, i.e. on observations the FP16 policy
actually reached.

| config | quant% | weights GiB | whole-model | in-scope | p50 ms | p90 ms | speedup | mean\|Δa\| | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | – | 2.97 | – | – | 143.7 | 145.6 | 1.00x | – | – |
| RTN llm | 24.4% | 2.49 | 1.22x | 3.76x | 134.2 | 176.4 | 1.07x | 0.00871 | 1.000 |
| RTN llm+vision | 75.9% | 1.52 | 2.15x | 3.39x | 107.1 | 130.2 | 1.34x | 0.01195 | 1.000 |
| RTN llm+vision+head | 87.7% | 1.27 | 2.64x | 3.44x | 88.9 | 89.1 | 1.62x | 0.01246 | 1.000 |

Measured **uncontended** (all four L40s idle), 5 warmup queries excluded, p50 over
40 timed queries with CUDA sync on both sides.

## Stage/hardware accounting for the 143.7 ms FP16 query

The paper reports 0.036 s per 8-action chunk on an H100, so an L40 at 143.7 ms
is ~4x slower. That is hardware, not a bug:

| stage | p50 |
|---|---:|
| `prepare_observation` (TF-on-CPU resize, 2 cameras) | 7.5 ms |
| `get_action` (vision tower + LLM forward + action head) | 124.6 ms |
| full `act()` | 135.9 ms |

Preprocessing is 6% of the query, so pinning TensorFlow to CPU -- which is what
makes peak-VRAM measurable at all -- costs ~7.5 ms and is worth it. The
remaining 4x tracks the H100/L40 memory-bandwidth ratio (3.4 TB/s vs 864 GB/s).

**Attention A/B, now settled.** VLA-Adapter comments out `flash_attention_2` at
`openvla_utils.py:308`, but the fallback is already `sdpa`, not eager:

| impl | p50 | vs default | max\|Δa\| vs default |
|---|---:|---:|---:|
| default | 135.6 ms | 1.00x | – |
| `sdpa` (explicit) | 136.9 ms | 0.99x | 0.00e+00 |
| `flash_attention_2` | 132.7 ms | 1.02x | 4.56e-03 |

flash-attn buys 2% and **changes the numerics**, which would break bit-exact V1
parity against the captured fixture. Leave the repo default alone; the planned
`QVLA_ATTN_IMPL` A/B is retired.

## Schedule implication

MuJoCo sim measured separately at **17.0 ms/step** uncontended (2-camera 256px
offscreen EGL), against the plan's assumed 20 ms -- the assumption holds. An
earlier 37 ms/step reading was taken under 100% GPU contention and was 2.2x
pessimistic.

| config | ms/env-step | control Hz | 232-step episode | 4 suites x 500 ep |
|---|---:|---:|---:|---:|
| FP16 | 35.0 | 28.6 | 8.1 s | **4.5 h** |
| RTN llm | 33.8 | 29.6 | 7.8 s | 4.4 h |
| RTN llm+vision+head | 28.1 | 35.6 | 6.5 s | 3.6 h |

Against the plan's 3.8 h estimate for the VLA-Adapter lane -- close enough that
the Stage B budget stands.

## What this confirms

**The scope analysis was right a priori.** The walker predicted 1.22x
whole-model compression for LLM-only from a synthetic module tree, before any
checkpoint existed. The measured value is 1.22x. The parameter split it
predicted (LLM 24.6%) matches the measured 24.4%.

**LLM-only is not a defensible headline for this model.** 1.22x compression and
1.17x latency. Extending to vision + action head reaches 2.64x and 1.58x for a
fidelity cost of 0.0087 -> 0.0125 mean absolute action error. This is why
VLA-Adapter reports LLM-only and full-scope as co-primary rather than following
the backbone-only convention.

**In-scope compression stays ~3.4-3.8x regardless of scope**, while whole-model
compression moves 1.22x -> 2.64x. Quoting only the in-scope number would imply a
3.8x memory win that the device never sees; quoting only whole-model would
understate the quantizer. Both are reported.

**Latency gains lag memory gains**, as the L40 microbenchmarks predicted: this
model is built from many small Linears (the Pro action head is 240 x 896x896),
and at ~2 MB per layer INT4 is kernel-launch bound at ~1.3x rather than
bandwidth bound at ~3.6x.

## Headroom signal (informs the Stage C gate)

`headroom_verdict("vla_adapter", ...)` returns HEADROOM for LLM-only, but only
just: mean|Δa| 0.00871 against a 0.008 bound and cos 0.999492 against 0.9995.
Both fail by under 10%. Combined with a perfect gripper agreement of 1.000 over
80 actions, RTN damage on this model sits essentially at the noise floor.

That is consistent with the plan's expectation that VLA-Adapter -- at published
99.6 / 99.6 / 98.2 / 96.4 -- has almost no room for HSA to recover, and it is
why the HSA evidence lane is OpenVLA/libero_long (published 53.7) instead.
