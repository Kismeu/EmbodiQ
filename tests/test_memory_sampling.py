"""Short queries must still have a real process-memory sample."""
import os
import sys
from types import SimpleNamespace

from embodiq.core.memory import MemoryProbe


def test_reset_and_report_sample_even_without_a_background_tick(monkeypatch):
    used = [100]
    api = SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=lambda handle: [
        SimpleNamespace(pid=os.getpid(), usedGpuMemory=used[0]),
        SimpleNamespace(pid=-1, usedGpuMemory=999999)])
    monkeypatch.setitem(sys.modules, 'pynvml', api)
    probe = MemoryProbe()
    probe._nvml_handle = object()
    monkeypatch.setattr(probe, '_torch', lambda: None)
    probe._nvml_peak = 1000  # pre-warmup peak must not survive reset
    probe.reset_peak()
    assert probe._nvml_peak == 100
    used[0] = 200
    assert probe.report().nvml_peak_bytes == 200
    used[0] = 150
    assert probe.report().nvml_peak_bytes == 200


def test_unavailable_nvml_stays_missing_instead_of_fabricating_a_peak(monkeypatch):
    probe = MemoryProbe()
    monkeypatch.setattr(probe, '_torch', lambda: None)
    probe.reset_peak()
    assert probe.report().nvml_peak_bytes == 0
