"""VRAM accounting.

Four numbers, three attributions -- see MemoryReport in types.py for why.
Preconditions for nvml_peak_bytes to be attributable: exactly one policy process
per GPU, TensorFlow pinned to CPU (it otherwise grabs the whole device on first
use for an image resize), MuJoCo EGL on the same device.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from embodiq.core.types import MemoryReport


def nvml_device_index(device_index=0):
    """Resolve CUDA logical numbering through the installed PyTorch runtime.

    Both supported PyTorch versions provide this helper; unlike direct NVML
    indexing it respects CUDA_VISIBLE_DEVICES, including GPU UUID selection.
    """
    import torch
    return torch.cuda._get_nvml_device_index(device_index)


class MemoryProbe:
    """torch allocator counters plus an NVML sampler thread."""

    def __init__(self, device_index: int = 0, sampler_hz: float = 10.0) -> None:
        self.device_index = device_index
        self.sampler_hz = sampler_hz
        self._nvml_peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._weights_bytes = 0
        self._baseline = 0
        self._load_peak_alloc = 0
        self._load_peak_reserved = 0
        self._load_nvml_peak = 0
        self._cache_released = 0
        self._nvml_handle = None
        self._nvml_lock = threading.Lock()

    # -- torch allocator -------------------------------------------------
    def _torch(self):
        try:
            import torch

            return torch if torch.cuda.is_available() else None
        except Exception:
            return None

    def mark_before_load(self) -> None:
        t = self._torch()
        if t is None:
            return
        t.cuda.synchronize(self.device_index)
        t.cuda.reset_peak_memory_stats(self.device_index)
        self._baseline = t.cuda.memory_allocated(self.device_index)

    def mark_after_load(self) -> None:
        """Delta across load() == resident parameters AFTER quantization.

        Also closes the measurement hole this class used to warn about in prose:
        quantization runs in-process on the GPU, so by the time we get here the
        caching allocator is still holding every block the bf16 checkpoint
        needed. ``reset_peak_memory_stats`` zeroes the *counters* but not the
        *pool*, and NVML reports the process's real reservation -- so the next
        sample reads the load-phase footprint straight back and the whole VRAM
        column reports the original precision.

        So: record the load-phase peaks, hand the pool back to the driver, then
        start the steady-state measurement from a clean baseline.
        """
        t = self._torch()
        if t is None:
            return
        t.cuda.synchronize(self.device_index)

        # 1. the load phase is a real, separately reported number
        self._sample_nvml()
        self._load_peak_alloc = t.cuda.max_memory_allocated(self.device_index)
        self._load_peak_reserved = t.cuda.max_memory_reserved(self.device_index)
        self._load_nvml_peak = self._nvml_peak

        delta = t.cuda.memory_allocated(self.device_index) - self._baseline

        # 2. release the bf16-sized pool. empty_cache() frees only blocks with
        #    no live tensors, so this cannot disturb the quantized weights; it
        #    just stops them being counted twice.
        import gc

        reserved_before = t.cuda.memory_reserved(self.device_index)
        gc.collect()
        t.cuda.empty_cache()
        t.cuda.synchronize(self.device_index)
        self._cache_released = max(
            0, reserved_before - t.cuda.memory_reserved(self.device_index))

        # 3. steady state starts here
        t.cuda.reset_peak_memory_stats(self.device_index)
        with self._nvml_lock:
            self._nvml_peak = 0
        self._sample_nvml()

        if delta <= 0:
            raise RuntimeError(
                f"weights_bytes came out {delta} -- the allocator baseline was "
                f"polluted, almost always because a previous model was still "
                f"resident at mark_before_load(). Call Policy.unload() (which "
                f"drops references, gc.collect()s and empty_cache()s) before "
                f"loading the next one."
            )
        self._weights_bytes = delta

    def reset_peak(self) -> None:
        """Call after warmup so CUDA-graph capture does not dominate the peak."""
        t = self._torch()
        if t is not None:
            t.cuda.synchronize(self.device_index)
            t.cuda.reset_peak_memory_stats(self.device_index)
        # Reset sampled process memory too; otherwise load/compile peaks leak
        # into the inference-only measurement. empty_cache() again here because
        # warmup allocates its own transients, and a sample taken between warmup
        # and the first episode would otherwise carry them into the peak.
        if t is not None:
            t.cuda.empty_cache()
            t.cuda.synchronize(self.device_index)
        with self._nvml_lock:
            self._nvml_peak = 0
        self._sample_nvml()

    # -- NVML sampler ----------------------------------------------------
    def start(self) -> None:
        try:
            import pynvml  # provided by nvidia-ml-py
        except Exception:
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index(self.device_index))
        except Exception:
            return
        self._nvml_handle = handle

        def loop() -> None:
            # Resource benchmarks may select a faster steady-state cadence
            # after loading. Read the current setting on every iteration.
            while not self._stop.wait(1.0 / self.sampler_hz):
                self._sample_nvml()

        self._thread = threading.Thread(target=loop, daemon=True, name="nvml-sampler")
        self._thread.start()

    def _sample_nvml(self) -> None:
        """Also sample at measurement boundaries: a query can be < one tick."""
        if self._nvml_handle is None:
            return
        import pynvml
        with self._nvml_lock:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._nvml_handle)
                for proc in procs:
                    if proc.pid == os.getpid() and proc.usedGpuMemory:
                        self._nvml_peak = max(self._nvml_peak, int(proc.usedGpuMemory))
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- report ----------------------------------------------------------
    def report(self) -> MemoryReport:
        t = self._torch()
        peak_alloc = peak_reserved = 0
        if t is not None:
            t.cuda.synchronize(self.device_index)
            peak_alloc = t.cuda.max_memory_allocated(self.device_index)
            peak_reserved = t.cuda.max_memory_reserved(self.device_index)
        self._sample_nvml()
        return MemoryReport(
            weights_bytes=self._weights_bytes,
            peak_alloc_bytes=peak_alloc,
            peak_reserved_bytes=peak_reserved,
            nvml_peak_bytes=self._nvml_peak,
            device_index=self.device_index,
            load_peak_alloc_bytes=self._load_peak_alloc,
            load_peak_reserved_bytes=self._load_peak_reserved,
            load_nvml_peak_bytes=self._load_nvml_peak,
            cache_released_bytes=self._cache_released,
        )
