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

        Only meaningful if the model was built on CPU and moved to GPU already
        quantized. Quantizing in place on the GPU reports the fp16 peak and the
        whole VRAM column becomes meaningless.
        """
        t = self._torch()
        if t is None:
            return
        t.cuda.synchronize(self.device_index)
        delta = t.cuda.memory_allocated(self.device_index) - self._baseline
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

    # -- NVML sampler ----------------------------------------------------
    def start(self) -> None:
        try:
            import pynvml  # provided by nvidia-ml-py
        except Exception:
            return
        try:
            pynvml.nvmlInit()
        except Exception:
            return
        pid = os.getpid()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        interval = 1.0 / self.sampler_hz

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception:
                    continue
                for p in procs:
                    if p.pid == pid and p.usedGpuMemory:
                        self._nvml_peak = max(self._nvml_peak, int(p.usedGpuMemory))

        self._thread = threading.Thread(target=loop, daemon=True, name="nvml-sampler")
        self._thread.start()

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
        return MemoryReport(
            weights_bytes=self._weights_bytes,
            peak_alloc_bytes=peak_alloc,
            peak_reserved_bytes=peak_reserved,
            nvml_peak_bytes=self._nvml_peak,
            device_index=self.device_index,
        )
