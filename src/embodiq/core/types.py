"""Core data types shared by all three model backends.

numpy-only by design: this module is installed into three mutually incompatible
conda prefixes (torch 2.2 / numpy 1.26 for the two Prismatic-derived repos,
torch 2.7 / numpy 2.2 for FastWAM). Importing torch here would break that.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

ACTION_DIM = 7

# Camera keys are the LIBERO-native names, not per-model aliases.
CAM_AGENTVIEW = "agentview"
CAM_WRIST = "wrist"


class GripperConvention(str, Enum):
    """Which convention ``ActionChunk.actions[..., -1]`` is already in.

    All three models must emit ENV_NATIVE (-1 open / +1 close, what
    ``env.step`` expects). The conversion belongs inside the Policy, next to
    the model-specific post-processing it pairs with, not in the runner.
    """

    ENV_NATIVE = "env_native"
    ENV_NATIVE_BINARIZED = "env_native_binarized"


@dataclass(frozen=True)
class Observation:
    """A RAW simulator observation. Not rotated, not resized, not normalized.

    Preprocessing is a *model property* (OpenVLA rotates 180 deg and resizes with
    a TF lanczos filter; FastWAM concatenates two cameras horizontally and scales
    to [-1, 1]), so it lives inside each Policy. Keeping this struct raw is what
    lets the three backends see byte-identical inputs.
    """

    images: Mapping[str, np.ndarray]  # uint8 (H, W, 3) RGB at env resolution (256)
    proprio: np.ndarray  # float32 (8,) eef_pos(3) | axisangle(3) | gripper_qpos(2)
    raw: Mapping[str, Any]  # the full robosuite obs dict; FastWAM needs it
    t: int  # env step index, excluding the settle/wait steps
    task_description: str


@dataclass(frozen=True)
class ActionChunk:
    """The result of ONE model query.

    ``num_valid`` is the field that makes the three models interchangeable:
    OpenVLA returns 1 action and executes 1, VLA-Adapter returns 8 and executes 8,
    FastWAM returns 32 and executes 10. Without it, "query latency" and
    "per-env-step latency" collapse into one number and the cross-model table
    silently becomes incomparable (the two differ by 10x).
    """

    actions: np.ndarray  # float32 (K, 7), unnormalized, gripper already env-native
    num_valid: int  # how many leading rows the runner may execute
    query_latency_s: float = 0.0  # set by Policy.__call__, not by act()
    aux: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or self.actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must be (K, {ACTION_DIM}), got {self.actions.shape}")
        if not 1 <= self.num_valid <= self.actions.shape[0]:
            raise ValueError(
                f"num_valid={self.num_valid} outside [1, {self.actions.shape[0]}]"
            )


@dataclass(frozen=True)
class PolicySpec:
    """Static description of a loaded policy. Goes verbatim into the result JSON."""

    model_id: str  # "vla-adapter-pro" | "openvla-7b" | "fastwam-optional-idm"
    family: str  # "vla_adapter" | "openvla" | "fastwam"
    checkpoint: str
    cameras: Sequence[str]
    env_resolution: int  # 256 for all three
    uses_proprio: bool
    chunk_size: int  # actions returned per query: 1 | 8 | 32
    default_replan_steps: int  # actions executed per query: 1 | 8 | 10
    gripper: GripperConvention
    torch_dtype: str
    quant_id: str = "fp16"  # "fp16" | "rtn-w4g128" | "awq-w4g128" | "hsa-w4g128-rho1"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MemoryReport:
    """Four numbers, three attributions.

    weights_bytes proves packing actually happened (int4/fp16 must land in
    [0.28, 0.40]); nvml_peak_bytes is what a deployer actually needs. Both go in
    the headline table. Only valid when exactly one policy process owns the GPU
    and TensorFlow has been pinned to CPU.
    """

    weights_bytes: int = 0
    peak_alloc_bytes: int = 0
    peak_reserved_bytes: int = 0
    nvml_peak_bytes: int = 0
    device_index: int = 0
    # HSA only: the high-precision residual, reported separately so the
    # "bounded extra VRAM" claim is falsifiable.
    residual_bytes: int = 0
    residual_index_bytes: int = 0
    # AWQ only: the cost of keeping the input scale explicit instead of folding.
    input_scale_bytes: int = 0

    @property
    def activation_peak_bytes(self) -> int:
        return max(0, self.peak_alloc_bytes - self.weights_bytes)

    @property
    def runtime_overhead_bytes(self) -> int:
        return max(0, self.nvml_peak_bytes - self.peak_reserved_bytes)


class EpisodeStatus(str, Enum):
    """Four states, not a boolean.

    Both Prismatic eval loops wrap the step body in a bare ``except Exception``
    that records success=False, so an OOM or a broken quantized kernel is
    indistinguishable from a policy failure. Until errors are separated from
    failures, no success-rate delta in this project is interpretable.
    """

    SUCCESS = "success"
    TIMEOUT = "timeout"
    POLICY_ERROR = "policy_error"
    ENV_ERROR = "env_error"


@dataclass
class EpisodeRecord:
    episode_idx: int
    status: EpisodeStatus
    env_steps: int
    n_queries: int
    seed: int
    query_latency_s: list[float] = field(default_factory=list)
    env_step_s_total: float = 0.0
    error: Optional[dict] = None
    aux: dict = field(default_factory=dict)

    @property
    def counts_toward_success_rate(self) -> bool:
        return self.status in (EpisodeStatus.SUCCESS, EpisodeStatus.TIMEOUT)


class Policy(abc.ABC):
    """One model, one precision, one device.

    ``act()`` is ONE QUERY, not one env step. The runner asks for a chunk only
    when its pending buffer drains.
    """

    spec: PolicySpec

    @abc.abstractmethod
    def load(self) -> None:
        """Build weights, apply quantization, move to device.

        Ordering is load-checkpoint-THEN-quantize, always. FastWAM has six
        ``load_state_dict(strict=False)`` sites; quantizing first renames
        weight -> qweight/qzeros/scales, so the checkpoint load silently drops
        every quantized tensor and leaves random init. The model still runs.
        """

    @abc.abstractmethod
    def reset(self, *, task_id: int, episode_idx: int, task_description: str, seed: int) -> None:
        """Clear per-episode state (action queues, caches) and set per-episode RNG."""

    @abc.abstractmethod
    def act(self, obs: Observation) -> ActionChunk:
        """Preprocess -> forward -> unnormalize -> gripper fix. No timing here."""

    @abc.abstractmethod
    def memory_footprint(self) -> MemoryReport:
        ...

    def warmup(self, obs: Observation, n: int = 5) -> list[float]:
        """Discarded queries. FastWAM's first compiled query costs 30-90 s of
        CUDA-graph capture; excluding it is mandatory and must stay visible in
        the result JSON as compile_overhead_s rather than being averaged away."""
        out = []
        for _ in range(n):
            out.append(self(obs).query_latency_s)
        return out

    def unload(self) -> None:
        return None

    def __call__(self, obs: Observation) -> ActionChunk:
        """Times act() with a CUDA sync on both sides."""
        from embodiq.core.timing import cuda_sync_timer

        with cuda_sync_timer(self.spec.extra.get("device_index", 0)) as t:
            chunk = self.act(obs)
        return ActionChunk(
            actions=chunk.actions,
            num_valid=chunk.num_valid,
            query_latency_s=t.elapsed_s,
            aux=chunk.aux,
        )
