"""Per-step action-fidelity metrics -- the statistically sound headroom screen.

Why this module exists
----------------------
The Stage C "does INT4 damage anything enough for HSA to repair?" question was
originally gated on a success-rate delta at the screening tier. A power
simulation (paired McNemar, alpha=0.05) showed that gate is unusable:

    power to detect a REAL 3-pt FP16->RTN drop
    episodes   disc=5%  disc=9%  disc=15%
         100      0.10     0.10      0.08
         300      0.56     0.34      0.23
         500      0.84     0.57      0.37

At 100 episodes the gate reports "no headroom" ~90% of the time even when a
3-pt drop is real -- and that verdict was going to reframe or kill HSA.

The same rollouts carry 400x more information at the step level (100 eps x ~400
steps x 7 action tokens = 280k paired comparisons vs 100 episode outcomes), and
the metrics below need no rollout at all: run FP16 and the quantized variant on
the same held-out observations and compare.

Logical direction matters. Near-identical actions *imply* near-identical success
rates, so this is a high-power test for "there is nothing to repair". The
converse does not hold -- small per-step errors can compound over a 400-step
episode -- so a FAIL here must still be confirmed with a success-rate run.
Used in that direction (cheap veto, expensive confirmation) it is sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActionFidelity:
    """Paired comparison of a quantized policy against its FP16 self."""

    n_steps: int = 0
    abs_err: list[float] = field(default_factory=list)  # mean |da| per step
    max_err: float = 0.0
    cos: list[float] = field(default_factory=list)
    # OpenVLA only: discrete action tokens, where only argmax RANKING matters.
    n_tokens: int = 0
    n_tokens_agree: int = 0
    gripper_agree: int = 0
    gripper_total: int = 0
    margins: list[float] = field(default_factory=list)

    def add(self, a_ref: np.ndarray, a_q: np.ndarray) -> None:
        a_ref = np.asarray(a_ref, dtype=np.float64).reshape(-1, 7)
        a_q = np.asarray(a_q, dtype=np.float64).reshape(-1, 7)
        d = np.abs(a_ref - a_q)
        self.n_steps += a_ref.shape[0]
        self.abs_err.append(float(d.mean()))
        self.max_err = max(self.max_err, float(d.max()))
        num = float((a_ref * a_q).sum())
        den = float(np.linalg.norm(a_ref) * np.linalg.norm(a_q)) or 1.0
        self.cos.append(num / den)
        # The gripper dim is binarized before env.step, so a sign flip is
        # categorical, not a small error. It gets its own line.
        self.gripper_total += a_ref.shape[0]
        self.gripper_agree += int((np.sign(a_ref[:, 6]) == np.sign(a_q[:, 6])).sum())

    def add_tokens(self, tok_ref: np.ndarray, tok_q: np.ndarray) -> None:
        tok_ref = np.asarray(tok_ref).ravel()
        tok_q = np.asarray(tok_q).ravel()
        self.n_tokens += tok_ref.size
        self.n_tokens_agree += int((tok_ref == tok_q).sum())

    def add_margin(self, margin: float) -> None:
        """Top-1 minus top-2 logit over the 256 action bins.

        A shrinking margin distribution is the honest early-warning signal: it
        shows the model approaching a flip before any token actually flips.
        """
        self.margins.append(float(margin))

    def summary(self) -> dict:
        def ci95(k: int, n: int) -> float:
            if n == 0:
                return float("nan")
            p = k / n
            return 100.0 * 1.959963984540054 * (p * (1 - p) / n) ** 0.5

        out = {
            "n_steps": self.n_steps,
            "mean_abs_err": float(np.mean(self.abs_err)) if self.abs_err else None,
            "max_abs_err": self.max_err,
            "mean_cos": float(np.mean(self.cos)) if self.cos else None,
            "gripper_agreement": (self.gripper_agree / self.gripper_total) if self.gripper_total else None,
            "gripper_agreement_ci95_pt": ci95(self.gripper_agree, self.gripper_total),
        }
        if self.n_tokens:
            out["token_agreement"] = self.n_tokens_agree / self.n_tokens
            out["token_agreement_ci95_pt"] = ci95(self.n_tokens_agree, self.n_tokens)
            out["n_tokens"] = self.n_tokens
        if self.margins:
            m = np.asarray(self.margins)
            out["margin_p50"] = float(np.median(m))
            out["margin_p05"] = float(np.percentile(m, 5))
        return out


# Thresholds for the Stage C headroom screen. These say "INT4 is essentially
# lossless here", i.e. HSA has nothing to repair on this model/suite.
NO_HEADROOM = {
    "openvla": {"token_agreement": 0.995, "mean_abs_err": 0.008, "gripper_agreement": 0.999},
    "vla_adapter": {"mean_abs_err": 0.008, "mean_cos": 0.9995, "gripper_agreement": 0.999},
    "fastwam": {"mean_abs_err": 0.008, "mean_cos": 0.9995, "gripper_agreement": 0.999},
}


def headroom_verdict(family: str, summary: dict) -> dict:
    """NO_HEADROOM / HEADROOM / INCONCLUSIVE.

    NO_HEADROOM is a real verdict backed by ~1e5 paired samples: reframe HSA as
    a VRAM/accuracy-Pareto claim rather than an accuracy-recovery one.
    HEADROOM must still be confirmed by a success-rate run before Stage E.
    """
    th = NO_HEADROOM[family]
    checks, failed = {}, []
    for key, bound in th.items():
        val = summary.get(key)
        if val is None:
            checks[key] = None
            continue
        ok = val <= bound if key == "mean_abs_err" else val >= bound
        checks[key] = {"value": val, "bound": bound, "lossless": ok}
        if not ok:
            failed.append(key)
    if any(v is None for v in checks.values()):
        verdict = "INCONCLUSIVE"
    elif not failed:
        verdict = "NO_HEADROOM"
    else:
        verdict = "HEADROOM"
    return {
        "verdict": verdict,
        "failed_checks": failed,
        "checks": checks,
        "note": (
            "NO_HEADROOM is conclusive on its own. HEADROOM is a screen only -- "
            "confirm with >=500-episode paired success rate before starting Stage E."
        ),
    }
