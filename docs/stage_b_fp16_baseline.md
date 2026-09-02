# Stage B / V3: VLA-Adapter FP16 baseline -- PASS

Run `fp16_baseline_20260902_2125`. Protocol `official.vla_adapter` (max_steps 220/280/300/520,
wait 10, replan 8, benchmark-API initial states, seed 7). 4 suites x 10 tasks x
50 episodes = 2000 rollouts, one GPU per suite, 95 min wall clock.

| suite | ours | Wilson 95% CI | published | delta | gate | avg steps |
|---|---:|---:|---:|---:|:--:|---:|
| libero_spatial | 99.0% (495/500) | [97.7, 99.6] | 99.6 | -0.6 | PASS | 102 |
| libero_object | 99.4% (497/500) | [98.3, 99.8] | 99.6 | -0.2 | PASS | 134 |
| libero_goal | 97.2% (486/500) | [95.4, 98.3] | 98.2 | -1.0 | PASS | 112 |
| libero_10 | 95.2% (476/500) | [93.0, 96.8] | 96.4 | -1.2 | PASS | 266 |
| **avg (equal-weight)** | **97.70%** | | 98.45 | -0.75 | | |
| pooled | 97.70% (1954/2000) | | | | | |

Gate: published value inside our 95% CI **or** |delta| <= 2.0 pts. All four pass
on both criteria. **Zero errors across 2000 episodes**, so no success-rate figure
here is contaminated by a swallowed exception.

## The deltas are systematic, not noise

All four are negative, mean -0.75 pt. Four same-signed deviations is not what
sampling noise looks like. Two candidate causes, neither of which we can
separate without a rerun on the authors' exact stack:

* **Different GPU.** VLA-Adapter's own README states results vary slightly off
  an H100; ours ran on L40s.
* **mujoco 3.3.2 vs the authors' 3.3.0.** We deliberately aligned both stacks on
  3.3.2 so the Prismatic and FastWAM lanes share one physics version. That
  choice buys cross-model comparability and costs a small absolute offset.

**This does not threaten the quantization results.** Every quantized variant is
compared against THIS baseline, paired on identical initial states via
`protocol.mcnemar_delta`, so a constant offset cancels in the delta. It only
matters for the absolute-reproduction claim, which is why it is stated here
rather than rounded away.

## Episode-length assumption was wrong for high-accuracy models

The schedule assumed average episode length ~0.6 x max_steps. Measured:

| suite | max_steps | assumed | actual | ratio |
|---|---:|---:|---:|---:|
| libero_spatial | 220 | 132 | 102 | 0.46x |
| libero_object | 280 | 168 | 134 | 0.48x |
| libero_goal | 300 | 180 | 112 | 0.37x |
| libero_10 | 520 | 312 | 266 | 0.51x |

A 99%-success policy terminates early, so the run took 95 min against a 4.5 h
estimate. The bias is systematic and directional: **the more accurate the model,
the cheaper the suite**. OpenVLA (published 84.7 / 88.4 / 79.2 / 53.7) will sit
much closer to the step caps -- especially libero_10 at 53.7% -- so its estimate
is if anything optimistic, and the schedule should not be revised downward on
the strength of this run alone.

## Latency and memory

p50 156.9 ms over 39,244 queries, peak VRAM 5.76 GiB (NVML, process-attributed).
**All 40 task files are tagged `contended=true`** -- the run shared all four
GPUs with a vLLM job and a separate training job. Success rates are unaffected;
these latencies are for sanity only and must not enter the headline table. The
dedicated exclusive-GPU latency pass is still owed.
