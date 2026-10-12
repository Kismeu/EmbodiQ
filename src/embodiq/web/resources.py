"""Read GPU indices and memory without exposing hardware identity."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
import subprocess
import threading
import time


class GPUMemoryMonitor:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._devices_cache = None
        self._devices_lock = threading.Lock()

    def devices(self) -> dict:
        with self._devices_lock:
            cached = self._devices_cache
            if cached and time.monotonic() - cached[0] < 2:
                return cached[1]
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=2,
                )
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if any(not re.fullmatch(r'[0-9]{1,3}', line) or int(line) > 255 for line in lines):
                    raise ValueError('invalid device index')
                indices = sorted({int(line) for line in lines})
                sample = {'status': 'ok', 'gpus': indices,
                          'message': '' if indices else '未检测到显卡，请检查服务器设备后刷新列表。'}
            except (OSError, subprocess.SubprocessError, ValueError):
                sample = {'status': 'unavailable', 'gpus': [],
                          'message': '暂无法读取显卡列表，请检查服务器显卡状态后重试。'}
            self._devices_cache = (time.monotonic(), sample)
            return sample

    def sample(self, gpu: int) -> dict:
        if type(gpu) is not int or not 0 <= gpu <= 255:
            raise ValueError('显卡编号需为 0–255 的整数')
        # Share short-lived samples between browser tabs without launching
        # multiple nvidia-smi processes for the same device at once.
        with self._lock:
            cached = self._cache.get(gpu)
            if cached and time.monotonic() - cached[0] < 2:
                return cached[1]
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--id=' + str(gpu),
                     '--query-gpu=memory.total,memory.used,memory.free',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=2,
                )
                total, used, free = [float(v.strip()) for v in result.stdout.strip().split(',')]
                if (not all(math.isfinite(v) for v in (total, used, free))
                        or total <= 0 or not 0 <= used <= total or not 0 <= free <= total):
                    raise ValueError('invalid memory measurement')
                sample = {
                    'status': 'ok',
                    # NVIDIA reports MiB. Use the measured free memory rather
                    # than total - used, which would include reserved memory.
                    'memory': {'total_gib': total / 1024, 'used_gib': used / 1024,
                               'free_gib': free / 1024},
                    'sampled_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'message': '',
                }
            except (OSError, subprocess.SubprocessError, ValueError):
                # Driver errors may contain device names or identifiers.
                sample = {'status': 'unavailable', 'memory': None, 'sampled_at': None,
                          'message': '暂无法读取显存，请检查所选显卡编号或稍后刷新。'}
            self._cache[gpu] = (time.monotonic(), sample)
            return sample
