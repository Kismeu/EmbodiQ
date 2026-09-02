"""Pin TensorFlow to CPU before either Prismatic repo imports it.

Both repos do a module-level ``import tensorflow as tf`` in
``experiments/robot/libero/libero_utils.py`` and use it for exactly two things:
``resize_image`` (lanczos3 256->224) and ``crop_and_resize`` (center crop 0.9).
TF 2.15 nevertheless claims the entire GPU on first use, which

  1. makes torch's peak-VRAM numbers meaningless, and
  2. blocks the second worker per GPU that halves OpenVLA's wall clock.

TF on CPU costs ~3-5 ms per 256x256 resize, which is accounted for in the
schedule. This module MUST be imported before any backend.
"""

from __future__ import annotations

import os
import sys

_applied = False


def apply(force: bool = False) -> bool:
    """Returns True if TF is now GPU-free (or absent). Idempotent."""
    global _applied
    if _applied and not force:
        return True
    if os.environ.get("QVLA_TF_CPU_ONLY", "1") != "1" and not force:
        return False

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if "tensorflow" in sys.modules:
        # Someone imported TF before us. Devices may already be initialized, in
        # which case set_visible_devices raises and the measurement is tainted.
        import tensorflow as tf  # noqa: F401

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError(
                "tensorflow was imported before tf_cpu_guard.apply() and its "
                "devices are already initialized; VRAM measurements would be "
                "invalid. Import embodiq.backends.compat.tf_cpu_guard first."
            ) from exc
    else:
        try:
            import tensorflow as tf
        except ImportError:
            _applied = True
            return True
        tf.config.set_visible_devices([], "GPU")

    _applied = True
    return True


def assert_tf_off_gpu() -> None:
    """Doctor check."""
    try:
        import tensorflow as tf
    except ImportError:
        return
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        raise AssertionError(f"TensorFlow still sees {len(gpus)} GPU(s): {gpus}")
