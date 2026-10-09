"""Exercise scheduling, cancellation and report admission without model GPUs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from embodiq import pipeline as p
from embodiq import pipeline_report as r
from embodiq.ctl.cli import main
from test_report_matrix import run as result_fixture


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def config(tmp_path, profile='smoke'):
    cfg = p.prepare_config(p.ROOT / 'configs/stage_fp16_rtn.json', models=['openvla'],
                           suites=['libero_spatial'], profile=profile)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model.bin').write_bytes(b'weights')
    repo = tmp_path / 'repo'
    repo.mkdir()
    cfg['models']['openvla'].update(checkpoint=str(checkpoint), repo_root=str(repo))
    return cfg


def fake_bind(root, plan, **kwargs):
    tasks = [{'suite': suite, 'task_id': t,
              'episodes': [{'episode_idx': i, 'seed': 42, 'state_sha256': f'{t}-{i}'} for i in range(50)]}
             for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10') for t in range(10)]
    write(root / 'episode_manifest.json', {'protocol': 'unified.v3', 'tasks': tasks})
    write(root / 'resource_bench/inputs_v1.json', {'manifest_sha256': 'inputs'})
    write(root / 'input_identity.json', {'plan_sha256': p.digest(root / 'plan.json')})
    return p.digest(root / 'input_identity.json')


def fake_worker(cmd, env, log, **kwargs):
    if 'embodiq.pipeline_report' in cmd:
        return 0 if r.generate_report(Path(cmd[-1]))['complete'] else 1
    def arg(k):
        return cmd[cmd.index(k) + 1]
    model, method = arg('--model'), arg('--quant')
    if 'embodiq.entry.evaluate' in cmd:
        n = int(arg('--trials'))
        for task in map(int, arg('--tasks').split(',')):
            d = result_fixture(method=method, suite=arg('--suite'), task=task)
            d['policy']['checkpoint'] = arg('--checkpoint')
            d['protocol']['num_trials_per_task'] = n
            d['run']['status'] = 'ok'
            d['episodes'] = d['episodes'][:n]
            d['metrics'].update(n_episodes=n, n_success=n)
            write(Path(arg('--out')) / arg('--suite') / f'gpu0_task{task}_results.json', d)
        return 0
    plan = p.read(arg('--job-plan'))
    smoke = '--smoke' in cmd
    suites = {}
    for job in plan['jobs']:
        if (job['model'], job['quant']) != (model, method):
            continue
        d = result_fixture(method=method)
        suites[job['suite']] = {'checkpoint': job['checkpoint'], 'n_inputs': 1 if smoke else 50,
                               'n_queries_measured': 1 if smoke else 150, 'passes': 1 if smoke else 3,
                               'chunk_size': 1, 'replan_steps': 1,
                               'policy': d['policy'], 'quant': d['quant'],
                               'memory': {'weights_bytes': 5, 'peak_alloc_bytes': 10, 'nvml_peak_bytes': 20, 'load_peak_alloc_bytes': 30},
                               'latency': {'query_s': {'mean': .2}, 'per_env_step_s': .2}}
    write(Path(arg('--out')), {'schema': 'embodiq/resource_bench/1', 'phase': 'complete',
                               'model': model, 'quant': method, 'contended': False,
                               'contended_before': [], 'contended_after': [], 'validation_only': smoke,
                               'plan_sha256': p.digest(arg('--job-plan')), 'inputs_manifest_sha256': 'inputs',
                               'warmup_queries': 2 if smoke else 20, 'measured_passes': 1 if smoke else 3,
                               'suites': suites})
    return 0


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(p, 'pipeline_preflight', lambda plan: [])
    monkeypatch.setattr(p, 'bind_inputs', fake_bind)
    monkeypatch.setattr(p, 'gpu_processes', lambda gpu: [])
    monkeypatch.setattr(p, 'run_child', fake_worker)


@pytest.mark.parametrize('profile', ['smoke', 'formal'])
def test_one_submit_runs_rollout_resource_report_and_resume_skips_completed(tmp_path, backend, monkeypatch, profile):
    root, _ = p.create_pipeline(config(tmp_path, profile), tmp_path / 'runs')
    assert p.execute_pipeline(root) == 0
    state = p.read(root / 'run.json')
    assert state['phase'] == 'complete'
    assert len(state['jobs']) == 5 and all(j['attempts'] == 1 for j in state['jobs'])
    doc = p.read(root / 'report/report.json')
    assert doc['n_episodes'] == (2 if profile == 'smoke' else 1000)
    assert doc['formal'] is (profile == 'formal')
    assert all(a['four_suite_macro_success_rate'] is None for a in doc['aggregates'])
    monkeypatch.setattr(p, 'run_child', lambda *a, **k: pytest.fail('completed stage reran'))
    assert p.execute_pipeline(root, resume=True) == 0


def test_failure_is_not_complete_and_resume_only_retries_unfinished(tmp_path, backend, monkeypatch):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    def fail_rtn(cmd, *a, **kw):
        if 'embodiq.entry.evaluate' in cmd and 'rtn' in cmd:
            return 7
        return fake_worker(cmd, *a, **kw)
    monkeypatch.setattr(p, 'run_child', fail_rtn)
    assert p.execute_pipeline(root) == 7
    state = p.read(root / 'run.json')
    assert state['phase'] == 'failed' and state['jobs'][0]['status'] == 'complete'
    assert not (root / 'report/report.json').exists()
    monkeypatch.setattr(p, 'run_child', fake_worker)
    assert p.execute_pipeline(root, resume=True) == 0
    state = p.read(root / 'run.json')
    assert state['jobs'][0]['attempts'] == 1 and state['jobs'][1]['attempts'] == 2


def test_busy_gpu_blocks_without_starting_a_worker(tmp_path, backend, monkeypatch):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    monkeypatch.setattr(p, 'gpu_processes', lambda gpu: [999999])
    monkeypatch.setattr(p, 'run_child', lambda *a, **k: pytest.fail('busy GPU used'))
    assert p.execute_pipeline(root) == 2
    state = p.read(root / 'run.json')
    assert state['phase'] == 'blocked' and state['jobs'][0]['status'] == 'blocked'
    assert state['jobs'][0]['attempts'] == 0


def test_stop_marker_and_resume(tmp_path, backend):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    p.request_stop(root)
    assert p.execute_pipeline(root) == 130
    assert p.execute_pipeline(root, resume=True) == 0


def test_stop_during_worker_records_interrupted_and_can_resume(tmp_path, backend, monkeypatch):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    def interrupted(*args, **kwargs):
        p.request_stop(root)
        return 130
    monkeypatch.setattr(p, 'run_child', interrupted)
    assert p.execute_pipeline(root) == 130
    state = p.read(root / 'run.json')
    assert state['phase'] == 'stopped' and state['jobs'][0]['status'] == 'interrupted'
    monkeypatch.setattr(p, 'run_child', fake_worker)
    assert p.execute_pipeline(root, resume=True) == 0


def test_changed_completed_output_is_not_silently_reused(tmp_path, backend):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    assert p.execute_pipeline(root) == 0
    file = next((root / 'results').rglob('*_results.json'))
    file.write_text('{}')
    assert p.execute_pipeline(root, resume=True) == 2
    assert 'output changed' in p.read(root / 'run.json')['reasons'][0]


def test_plan_and_stage_tampering_rejected(tmp_path):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    state = p.read(root / 'run.json')
    state['jobs'][0]['job']['checkpoint'] = '/different'
    write(root / 'run.json', state)
    with pytest.raises(ValueError, match='immutable plan'):
        p.execute_pipeline(root)


@pytest.mark.parametrize('method', ['awq', 'hsa'])
def test_pending_method_rejected_before_run_creation(method):
    with pytest.raises(ValueError, match='pending'):
        p.prepare_config(p.ROOT / 'configs/stage_fp16_rtn.json', methods=[method])


def test_dry_run_has_all_three_models_and_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(p, 'execute_pipeline', lambda *a, **k: pytest.fail('dry-run launched'))
    assert main(['launch', str(p.ROOT / 'configs/stage_fp16_rtn.json'), '--dry-run']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['rollout_jobs'] == 24 and output['resource_jobs'] == 6


def test_real_child_cancellation_stops_only_owned_process_group(tmp_path):
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    started = []
    deadline = time.monotonic() + .7
    try:
        code = p.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], dict(os.environ),
                           tmp_path / 'worker.log', stop=lambda: time.monotonic() > deadline,
                           on_start=started.append)
        assert code == 130 and other.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(started[0], 0)
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_inherited_lock_prevents_orphan_overlap(tmp_path):
    path = tmp_path / 'lock'
    with p.lock(path) as lock:
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], pass_fds=(lock.fileno(),))
    try:
        with pytest.raises(RuntimeError, match='lock_busy'):
            with p.lock(path):
                pass
    finally:
        child.terminate()
        child.wait(timeout=10)
    with p.lock(path):
        pass


@pytest.mark.parametrize('mutation', ['checkpoint', 'profile', 'contention', 'count', 'plan', 'nan'])
def test_bad_resource_cannot_enter_report(tmp_path, backend, mutation):
    root, _ = p.create_pipeline(config(tmp_path), tmp_path / 'runs')
    assert p.execute_pipeline(root) == 0
    path = root / 'resource_bench/openvla_rtn.json'
    d = p.read(path)
    if mutation == 'checkpoint': d['suites']['libero_spatial']['checkpoint'] = '/wrong'
    if mutation == 'profile': d['validation_only'] = False
    if mutation == 'contention': d['contended'] = True
    if mutation == 'count': d['suites']['libero_spatial']['n_queries_measured'] = 0
    if mutation == 'plan': d['plan_sha256'] = 'wrong'
    if mutation == 'nan': d['suites']['libero_spatial']['latency']['query_s']['mean'] = float('nan')
    write(path, d)
    with pytest.raises(ValueError):
        r.validate_resource(root, p.read(root / 'plan.json'), 'openvla', 'rtn')
    doc = r.generate_report(root)
    assert not doc['complete'] and not doc['formal'] and doc['problems']


def test_input_binding_detects_changed_checkpoint(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    resource = tmp_path / 'inputs.json'
    write(resource, {'tasks': [], 'manifest_sha256': hashlib.sha256(b'[]').hexdigest()})
    cfg['resource']['fixed_input_manifest'] = str(resource)
    root, _ = p.create_pipeline(cfg, tmp_path / 'runs')
    plan = p.read(root / 'plan.json')
    frozen = tmp_path / 'episodes.json'
    write(frozen, {'tasks': []})
    plan['episode_source'] = str(frozen)
    monkeypatch.setattr(p, '_source_files', lambda plan: set())
    p.bind_inputs(root, plan)
    (tmp_path / 'checkpoint/model.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        p.bind_inputs(root, plan)


@pytest.mark.parametrize('model,module_name,class_name', [
    ('openvla', 'embodiq.backends.openvla.policy', 'OpenVLAPolicy'),
    ('vla_adapter', 'embodiq.backends.adapter.policy', 'VLAAdapterPolicy'),
    ('fastwam', 'embodiq.backends.fastwam.policy', 'FastWAMPolicy'),
])
def test_resource_loader_uses_submitted_paths_not_historical_defaults(tmp_path, monkeypatch, model, module_name, class_name):
    import importlib
    import run_resource_bench as bench
    monkeypatch.setattr(sys, 'path', list(sys.path))
    module = importlib.import_module(module_name)
    captured = {}
    class ReachedModelLoad(Exception):
        pass
    class CapturePolicy:
        def __init__(self, checkpoint, suite, **kwargs):
            captured.update(checkpoint=checkpoint, suite=suite, kwargs=kwargs)
            raise ReachedModelLoad
    monkeypatch.setattr(module, class_name, CapturePolicy)
    monkeypatch.setattr(bench.base, 'checkpoint', lambda *a: pytest.fail('historical checkpoint used'))
    job = {'repo_root': str(tmp_path / 'custom-repo'), 'checkpoint': str(tmp_path / 'custom-checkpoint'),
           'dataset_stats': str(tmp_path / 'custom-stats.json'), 'seed': 42}
    with pytest.raises(ReachedModelLoad):
        bench.bench_suite(model, 'libero_spatial', 'fp16', None, 0, [], 'first_frame', job=job)
    assert captured['checkpoint'] == job['checkpoint']
    if model == 'fastwam':
        assert str(captured['kwargs']['dataset_stats']) == job['dataset_stats']
        assert str(captured['kwargs']['repo_root']) == job['repo_root']
