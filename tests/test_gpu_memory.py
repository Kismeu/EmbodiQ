"""Memory units, stale samples, invalid measurements and the public field allowlist."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from embodiq.web import resources
from embodiq.web.resources import GPUMemoryMonitor
from embodiq.web.server import Handler


def test_device_list_preserves_real_indices_and_caches_then_recovers(monkeypatch):
    now, calls, output = [10], [], ['5\n2\n5\n']
    monkeypatch.setattr(resources.time, 'monotonic', lambda: now[0])
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=output[0])
    monkeypatch.setattr(resources.subprocess, 'run', query)
    monitor = GPUMemoryMonitor()
    assert monitor.devices() == {'status': 'ok', 'gpus': [2, 5], 'message': ''}
    assert monitor.devices()['gpus'] == [2, 5] and len(calls) == 1
    assert calls[0] == ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits']
    now[0] += 3
    output[0] = 'invalid driver output'
    assert monitor.devices()['gpus'] == []  # No stale options or invented GPU 0.
    now[0] += 3
    output[0] = '3\n'
    assert monitor.devices()['gpus'] == [3]


@pytest.mark.parametrize('output', ['', ' \n', 'No devices were found', 'GPU-secret-UUID',
                                    '0\nN/A', '-1', '256', '1.5', '0, model-name'])
def test_device_list_empty_or_invalid_never_invents_indices(monkeypatch, output):
    monkeypatch.setattr(resources.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=output))
    result = GPUMemoryMonitor().devices()
    assert result['gpus'] == [] and result['message']
    assert result['status'] == ('ok' if not output.strip() else 'unavailable')
    assert 'GPU-secret' not in json.dumps(result) and 'model-name' not in json.dumps(result)


@pytest.mark.parametrize('error', [FileNotFoundError('nvidia-smi'),
    subprocess.TimeoutExpired('nvidia-smi', 2),
    subprocess.CalledProcessError(1, 'nvidia-smi', stderr='private driver details')])
def test_device_list_failure_is_safe(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(resources.subprocess, 'run', fail)
    result = GPUMemoryMonitor().devices()
    assert result['status'] == 'unavailable' and result['gpus'] == []
    assert 'private driver' not in json.dumps(result)


def test_measured_free_memory_and_private_fields(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='46068, 37302, 8187\n')
    monkeypatch.setattr(resources.subprocess, 'run', query)
    result = GPUMemoryMonitor().sample(3)
    assert result['status'] == 'ok'
    assert result['memory'] == {'total_gib': 46068 / 1024, 'used_gib': 37302 / 1024,
                                'free_gib': 8187 / 1024}
    assert result['memory']['free_gib'] != result['memory']['total_gib'] - result['memory']['used_gib']
    assert set(result) == {'status', 'memory', 'sampled_at', 'message'}
    assert calls[0] == ['nvidia-smi', '--id=3', '--query-gpu=memory.total,memory.used,memory.free',
                        '--format=csv,noheader,nounits']


def test_cache_is_per_device_and_does_not_extend_sample_lifetime(monkeypatch):
    now = [10.0]
    calls = []
    monkeypatch.setattr(resources.time, 'monotonic', lambda: now[0])
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'1024, {len(calls)}, 900')
    monkeypatch.setattr(resources.subprocess, 'run', query)
    monitor = GPUMemoryMonitor()
    first = monitor.sample(0)
    now[0] = 11.9
    assert monitor.sample(0) == first
    assert len(calls) == 1
    assert monitor.sample(1)['memory']['used_gib'] == 2 / 1024
    now[0] = 12.1
    assert monitor.sample(0)['memory']['used_gib'] == 3 / 1024
    assert len(calls) == 3


@pytest.mark.parametrize('output', ['', '1024, [N/A], 900', '1024, nan, 900',
                                   '1024, 0, inf', '0, 0, 0', '1024, -1, 900',
                                   '1024, 2048, 0', '1024, 0, 2048',
                                   '1024, 0, 900\n2048, 0, 1800'])
def test_invalid_measurements_are_unavailable_not_zero(monkeypatch, output):
    monkeypatch.setattr(resources.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=output))
    result = GPUMemoryMonitor().sample(0)
    assert result['status'] == 'unavailable'
    assert result['memory'] is None and result['sampled_at'] is None
    assert output not in result['message'] if output else True


@pytest.mark.parametrize('error', [FileNotFoundError('nvidia-smi'),
    subprocess.TimeoutExpired('nvidia-smi', 2),
    subprocess.CalledProcessError(1, 'nvidia-smi', stderr='GPU-secret-UUID driver version example')])
def test_probe_failure_does_not_expose_identity_or_cached_readings(monkeypatch, error):
    now = [0]
    monkeypatch.setattr(resources.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(resources.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='1024, 1, 900'))
    monitor = GPUMemoryMonitor()
    assert monitor.sample(0)['status'] == 'ok'
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(resources.subprocess, 'run', fail)
    now[0] = 3
    result = monitor.sample(0)
    assert result['status'] == 'unavailable' and result['memory'] is None
    assert 'GPU-secret' not in json.dumps(result) and 'driver version' not in json.dumps(result)


@pytest.mark.parametrize('gpu', [-1, 256, True, 1.5, '0'])
def test_bad_indices_never_spawn(monkeypatch, gpu):
    monkeypatch.setattr(resources.subprocess, 'run', lambda *a, **kw: pytest.fail('must not spawn'))
    with pytest.raises(ValueError):
        GPUMemoryMonitor().sample(gpu)


@pytest.mark.parametrize('query,code,index', [('', 200, 0), ('?gpu=3', 200, 3),
    ('?gpu=-1', 400, None), ('?gpu=256', 400, None), ('?gpu=', 400, None),
    ('?gpu=1&gpu=2', 400, None), ('?gpu=1.5', 400, None), ('?gpu=0&name=1', 400, None)])
def test_memory_route_contract_without_opening_a_socket(query, code, index):
    calls, replies = [], []
    def sample(gpu):
        calls.append(gpu)
        return {'status': 'ok', 'memory': {'total_gib': 1, 'used_gib': 0, 'free_gib': 1}}
    handler = Handler.__new__(Handler)
    handler.server = SimpleNamespace(server_port=8770, app=SimpleNamespace(memory_monitor=SimpleNamespace(sample=sample)))
    handler.headers = {'Host': '127.0.0.1:8770'}
    handler.path = '/api/gpu-memory' + query
    handler.reply = lambda status, value, *args: replies.append((status, value))
    handler.handle_request(False)
    assert replies[0][0] == code
    assert calls == ([index] if index is not None else [])
