"""Core data types shared by all three model backends.

numpy-only by design: this module is installed into three mutually incompatible
conda prefixes (torch 2.2 / numpy 1.26 for the two Prismatic-derived repos,
torch 2.7 / numpy 2.2 for FastWAM). Importing torch here would break that.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

ACTION_DIM = 7

# Camera keys are the LIBERO-native names, not per-model aliases.
CAM_AGENTVIEW = "agentview"
CAM_WRIST = "wrist"


@dataclass(frozen=True)
class ModelContract:
    """The frozen boundary between LIBERO and a model backend.

    The runner only sees canonical simulator names. Each backend owns the
    model-specific image transforms, prompt template, normalization and
    action post-processing behind this boundary.
    """

    family: str
    cameras: tuple[str, ...]
    env_resolution: int
    uses_proprio: bool
    chunk_size: int
    native_replan_steps: int
    action_dim: int = ACTION_DIM


# Native query/execution semantics. ``default_replan_steps`` may be overridden
# for an isolation experiment, but the chunk size and action dimension are part
# of the backend contract and must not drift.
MODEL_CONTRACTS: Mapping[str, ModelContract] = {
    "openvla": ModelContract(
        family="openvla",
        cameras=(CAM_AGENTVIEW,),
        env_resolution=256,
        uses_proprio=False,
        chunk_size=1,
        native_replan_steps=1,
    ),
    "vla_adapter": ModelContract(
        family="vla_adapter",
        cameras=(CAM_AGENTVIEW, CAM_WRIST),
        env_resolution=256,
        uses_proprio=True,
        chunk_size=8,
        native_replan_steps=8,
    ),
    "fastwam": ModelContract(
        family="fastwam",
        cameras=(CAM_AGENTVIEW, CAM_WRIST),
        env_resolution=256,
        uses_proprio=True,
        chunk_size=32,
        native_replan_steps=10,
    ),
}


def model_contract(family: str) -> ModelContract:
    """Return the frozen contract for a known model family."""
    try:
        return MODEL_CONTRACTS[family]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {family!r}") from exc


def validate_policy_spec(spec: "PolicySpec", *, allow_replan_override: bool = True) -> None:
    """Validate a backend's public spec against the frozen model contract.

    ``allow_replan_override`` is intentional: the runner supports controlled
    cadence-ablation rows, while the native chunk size and action width remain
    invariant.
    """
    expected = model_contract(spec.family)
    errors: list[str] = []
    if tuple(spec.cameras) != expected.cameras:
        errors.append(f"cameras={tuple(spec.cameras)!r}, expected {expected.cameras!r}")
    if spec.env_resolution != expected.env_resolution:
        errors.append(f"env_resolution={spec.env_resolution}, expected {expected.env_resolution}")
    if spec.uses_proprio != expected.uses_proprio:
        errors.append(f"uses_proprio={spec.uses_proprio}, expected {expected.uses_proprio}")
    if spec.chunk_size != expected.chunk_size:
        errors.append(f"chunk_size={spec.chunk_size}, expected {expected.chunk_size}")
    if spec.default_replan_steps < 1 or spec.default_replan_steps > spec.chunk_size:
        errors.append(
            f"default_replan_steps={spec.default_replan_steps} outside "
            f"[1, {spec.chunk_size}]"
        )
    if not allow_replan_override and spec.default_replan_steps != expected.native_replan_steps:
        errors.append(
            f"default_replan_steps={spec.default_replan_steps}, "
            f"expected native value {expected.native_replan_steps}"
        )
    if expected.action_dim != ACTION_DIM:
        errors.append(f"internal action dimension mismatch: {expected.action_dim} != {ACTION_DIM}")
    if errors:
        raise ValueError(f"invalid {spec.family} PolicySpec: " + "; ".join(errors))


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

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("Observation.images must contain at least one camera")
        for camera, image in self.images.items():
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Observation.images[{camera!r}] must be a numpy array")
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Observation.images[{camera!r}] must be HWC with 3 channels, "
                    f"got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise TypeError(
                    f"Observation.images[{camera!r}] must be uint8, got {image.dtype}"
                )
        if self.proprio.shape != (8,):
            raise ValueError(f"Observation.proprio must have shape (8,), got {self.proprio.shape}")
        if not np.issubdtype(self.proprio.dtype, np.number):
            raise TypeError(f"Observation.proprio must be numeric, got {self.proprio.dtype}")
        if not np.isfinite(self.proprio).all():
            raise ValueError("Observation.proprio contains NaN or Inf")
        if self.t < 0:
            raise ValueError(f"Observation.t must be non-negative, got {self.t}")


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
        if not isinstance(self.actions, np.ndarray):
            raise TypeError("ActionChunk.actions must be a numpy array")
        if self.actions.ndim != 2 or self.actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must be (K, {ACTION_DIM}), got {self.actions.shape}")
        if not np.issubdtype(self.actions.dtype, np.floating):
            raise TypeError(f"actions must use a floating dtype, got {self.actions.dtype}")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions contains NaN or Inf")
        if not isinstance(self.num_valid, (int, np.integer)):
            raise TypeError(f"num_valid must be an integer, got {type(self.num_valid).__name__}")
        if not 1 <= int(self.num_valid) <= self.actions.shape[0]:
            raise ValueError(
                f"num_valid={self.num_valid} outside [1, {self.actions.shape[0]}]"
            )
        if not np.isfinite(self.query_latency_s) or self.query_latency_s < 0:
            raise ValueError(f"query_latency_s must be finite and non-negative, got {self.query_latency_s}")


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

    def __post_init__(self) -> None:
        if self.env_resolution <= 0:
            raise ValueError(f"env_resolution must be positive, got {self.env_resolution}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.default_replan_steps < 1 or self.default_replan_steps > self.chunk_size:
            raise ValueError(
                f"default_replan_steps={self.default_replan_steps} outside "
                f"[1, {self.chunk_size}]"
            )
        if not self.cameras:
            raise ValueError("PolicySpec.cameras must contain at least one camera")
        if self.family in MODEL_CONTRACTS:
            validate_policy_spec(self)


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
    # The load phase and the steady state are different numbers and both must be
    # reported. Quantization happens in-process AFTER the bf16 checkpoint is
    # resident, so the load peak is bounded by the ORIGINAL precision no matter
    # what the quantized model costs to run: measured in-process, RTN raised the
    # FastWAM process peak from 24.8 to 33.2 GiB while its steady-state
    # allocation fell from 23.3 to 15.2 GiB. Reporting only the process peak
    # would say quantization increases VRAM, which is an artifact of the
    # measurement, not a property of the method.
    load_peak_alloc_bytes: int = 0
    load_peak_reserved_bytes: int = 0
    load_nvml_peak_bytes: int = 0
    # Bytes the caching allocator returned to the driver after quantization.
    # Zero here with a large load/steady gap means the release did not happen
    # and nvml_peak_bytes is still carrying the load phase.
    cache_released_bytes: int = 0
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
        return self.status in (EpisodeStatus.SUCCESS, EpisodeStatus.TIMEOUT, EpisodeStatus.POLICY_ERROR)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        """Rebuild a record read back from a checkpoint.

        Only the fields this class declares are accepted; an unknown key means
        the file was written by a different schema and must not be resumed from.
        """
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"EpisodeRecord has no field(s) {sorted(unknown)}")
        return cls(
            episode_idx=int(d["episode_idx"]),
            status=EpisodeStatus(d["status"]),
            env_steps=int(d["env_steps"]),
            n_queries=int(d["n_queries"]),
            seed=int(d["seed"]),
            query_latency_s=[float(x) for x in d.get("query_latency_s") or []],
            env_step_s_total=float(d.get("env_step_s_total") or 0.0),
            error=d.get("error"),
            aux=dict(d.get("aux") or {}),
        )


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

    def quantization_roots(self) -> Mapping[str, object]:
        """Return named module roots exposed to a quantization backend.

        Adapters own this mapping because some policies have trainable modules
        outside their main checkpoint.  Quantizers consume this stable boundary
        and must not inspect private model-specific attributes.  The concrete
        default keeps lightweight FP16/test policies compatible; a quantized
        adapter must override it and is expected to fail before any swap when
        the model has not been loaded.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose quantization roots"
        )

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
