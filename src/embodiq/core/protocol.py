"""Rollout protocols.

The three repos do not agree on how to run LIBERO: max_steps is 220/280/300/520
for the two Prismatic models but 400/400/400/700 for FastWAM; the settle period
is 10 steps vs 30; replan is 1 / 8 / 10 steps. Cross-model numbers are only
comparable once these are named and pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_ALIASES = {"long": "libero_10", "libero_long": "libero_10"}

# Published FP16 references, used by the Stage A / Stage B reproduction gate.
PUBLISHED = {
    "openvla-7b": {"libero_spatial": 84.7, "libero_object": 88.4, "libero_goal": 79.2, "libero_10": 53.7},
    "vla-adapter": {"libero_spatial": 97.8, "libero_object": 99.2, "libero_goal": 97.2, "libero_10": 95.0},
    "vla-adapter-pro": {"libero_spatial": 99.6, "libero_object": 99.6, "libero_goal": 98.2, "libero_10": 96.4},
    # first-frame mode of the optional-IDM checkpoint
    "fastwam-first-frame": {"libero_spatial": 98.2, "libero_object": 99.2, "libero_goal": 97.8, "libero_10": 95.8},
}


@dataclass(frozen=True)
class RolloutProtocol:
    name: str
    max_steps: Mapping[str, int]
    num_steps_wait: int
    env_resolution: int
    num_trials_per_task: int
    initial_state_source: Literal["benchmark_api", "init_states_file"]
    seed_base: int
    episode_seed_fn: Literal["constant", "per_episode"]
    # None -> use each policy's native cadence. Forcing a common replan changes
    # the *policy*, not the protocol, so it is only used in an isolation row.
    replan_steps: Optional[int] = None
    binarize_gripper: bool = True
    save_videos: bool = False

    def steps_for(self, suite: str) -> int:
        return self.max_steps[SUITE_ALIASES.get(suite, suite)]

    def episode_seed(self, task_id: int, episode_idx: int) -> int:
        if self.episode_seed_fn == "constant":
            return self.seed_base
        return self.seed_base * 100003 + task_id * 1009 + episode_idx


_OFT_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
_FASTWAM_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}

PROTOCOLS: dict[str, RolloutProtocol] = {
    "official.openvla": RolloutProtocol(
        name="official.openvla", max_steps=_OFT_STEPS, num_steps_wait=10, env_resolution=256,
        num_trials_per_task=50, initial_state_source="benchmark_api", seed_base=7,
        episode_seed_fn="constant", replan_steps=1,
    ),
    "official.vla_adapter": RolloutProtocol(
        name="official.vla_adapter", max_steps=_OFT_STEPS, num_steps_wait=10, env_resolution=256,
        num_trials_per_task=50, initial_state_source="benchmark_api", seed_base=7,
        episode_seed_fn="constant", replan_steps=8,
    ),
    "official.fastwam": RolloutProtocol(
        name="official.fastwam", max_steps=_FASTWAM_STEPS, num_steps_wait=30, env_resolution=256,
        num_trials_per_task=50, initial_state_source="init_states_file", seed_base=0,
        episode_seed_fn="constant", replan_steps=10,
    ),
    # Union-maximum step budget and the longer settle time. Shortening would cut
    # FastWAM below its published operating point; lengthening can only help the
    # Prismatic models (a task solved at step 220 is still solved at 400).
    # replan_steps=None keeps each model's native chunking.
    "unified.v1": RolloutProtocol(
        name="unified.v1", max_steps=_FASTWAM_STEPS, num_steps_wait=30, env_resolution=256,
        num_trials_per_task=50, initial_state_source="init_states_file", seed_base=0,
        episode_seed_fn="constant", replan_steps=None,
    ),
}

# Screening tier: 10 trials/task = 100 eps/suite, 1/5 the cost. Decides; does not
# publish. At p~0.85 the Wilson half-width is +/-3.6 pts at 100 eps vs +/-1.6 at
# 500, so screening cannot resolve a 2-pt HSA-vs-AWQ gap. Tiering is explicit.
def screening(base: str) -> RolloutProtocol:
    import dataclasses

    return dataclasses.replace(
        PROTOCOLS[base], name=f"{base}.screen10", num_trials_per_task=10
    )


def wilson_ci95(n_success: int, n: int) -> tuple[float, float]:
    """Wilson score interval. The normal approximation is unusable at p ~= 0.98,
    which is where two of the three models live."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = n_success / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_delta(paired: list[tuple[bool, bool]]) -> dict:
    """Paired success-rate comparison on IDENTICAL initial states.

    This, not two independent Wilson intervals, is the right statistic for every
    quantization delta in this project. Both arms replay the same LIBERO
    init_states, so the comparison is paired and the discordant pairs carry all
    the information. Independent CIs are wildly over-conservative here: at
    p~=0.85 and n=100 a Wilson half-width is +/-7.0 pts, which would make a 3-pt
    FP16->RTN gap look unresolvable, while the paired test resolves it easily
    when the flips are few.

    Independent Wilson intervals remain the right tool for one other job:
    comparing an absolute FP16 number against a published one.

    Args:
        paired: [(baseline_success, variant_success), ...], one entry per episode.
    Returns:
        delta (variant - baseline, as a fraction), the discordant counts, the
        standard error of the paired difference, and the exact two-sided
        binomial p-value.
    """
    n = len(paired)
    if n == 0:
        return {"n": 0}
    n01 = sum(1 for b, v in paired if b and not v)  # baseline solved, variant lost
    n10 = sum(1 for b, v in paired if v and not b)  # variant recovered
    delta = (n10 - n01) / n
    disc = n01 + n10
    se = (disc ** 0.5) / n if disc else 0.0

    # Exact two-sided binomial test on the discordant pairs (Binomial(disc, 0.5)).
    if disc == 0:
        p_value = 1.0
    else:
        from math import comb

        k = min(n01, n10)
        tail = sum(comb(disc, i) for i in range(0, k + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** disc))

    return {
        "n": n,
        "delta": delta,
        "delta_pts": 100.0 * delta,
        "n_baseline_only": n01,
        "n_variant_only": n10,
        "n_discordant": disc,
        "se": se,
        "half_width_95_pts": 100.0 * 1.959963984540054 * se,
        "p_value": p_value,
    }
