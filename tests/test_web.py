"""HTTP contract plus real subprocess lifecycle, with CPU-only model fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import pytest

from embodiq import pipeline as p
from embodiq.web.server import APIError, Server, Workbench
from test_pipeline import config


@pytest.fixture
def app(tmp_path):
    cfg = config(tmp_path, profile="formal")
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    return Workbench(path, tmp_path / 'runs')


def body(**kwargs):
    return dict(models=['openvla'], methods=['fp16'], suites=['libero_spatial'], profile='formal',
                gpu=0, wait_seconds=0, **kwargs)


@pytest.fixture
def http(app):
    server = Server(('127.0.0.1', 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        h = {'Content-Type': 'application/json', 'X-Embodiq-Token': app.token}
        h.update(headers or {})
        req = Request(f'http://127.0.0.1:{server.server_port}{path}', data=data, headers=h)
        try:
            response = build_opener(ProxyHandler({})).open(req, timeout=15)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            result = json.loads(raw) if 'application/json' in response.headers['Content-Type'] else raw
            return response.status, result
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def wait_until(fn, timeout=30):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(.05)
    pytest.fail('timed out waiting for controller')


@pytest.fixture
def cpu_worker(monkeypatch):
    original = subprocess.Popen
    # Preserve the real launch, inherited file descriptor, signals and controller.
    # Only model/GPU operations are replaced in the child, as in test_pipeline.
    code = '''
import sys,time
from pathlib import Path
from embodiq import pipeline as p
sys.path.insert(0,str(p.ROOT / 'tests'))
from test_pipeline import fake_bind,fake_worker
p.pipeline_preflight=lambda plan: []
p.bind_inputs=fake_bind
p.gpu_processes=lambda gpu: []
def worker(cmd,env,log,**kw):
    for _ in range(20):
        if kw['stop'](): return 130
        time.sleep(.05)
    return fake_worker(cmd,env,log,**kw)
p.run_child=worker
sys.exit(p.execute_pipeline(sys.argv[1],gpu=int(sys.argv[2]),wait_seconds=int(sys.argv[3])))
'''
    def spawn(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[1:3] == ['-m', 'embodiq.web.worker']:
            cmd = [sys.executable, '-c', code, *cmd[3:]]
        return original(cmd, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', spawn)


def test_http_capabilities_assets_preview_and_empty_list(http, app):
    status, caps = http('/api/capabilities')
    assert status == 200 and caps['csrf_token'] == app.token
    assert caps['methods']['awq'] == caps['methods']['hsa'] == 'pending'
    for path in ('/', '/style.css', '/app.js'):
        status, content = http(path)
        assert status == 200 and content
    assert http('/api/runs')[1]['runs'] == []
    status, plan = http('/api/preview', body())
    assert status == 200 and plan['episodes'] == 500 and plan['total_stages'] == 3
    formal = body(); formal.update(models=['openvla'], methods=['fp16', 'rtn'], profile='formal',
                                  suites=['libero_spatial','libero_object','libero_goal','libero_10'])
    assert http('/api/preview', formal)[1]['episodes'] == 4000
    assert list(app.root.iterdir()) == []  # preview does not create runs


@pytest.mark.parametrize('headers', [
    {'X-Embodiq-Token': ''}, {'Origin': 'http://evil.example'}, {'Host': 'evil.example'},
])
def test_reject_cross_origin_mutations(http, app, headers):
    assert http('/api/runs', body(), headers)[0] == 403
    assert not list(app.root.iterdir())


@pytest.mark.parametrize('updates', [
    {'profile':'legacy'}, {'profile':None}, {'methods':['awq']}, {'methods':['hsa']}, {'models':[]}, {'suites':['unknown']},
    {'models':['openvla','openvla']}, {'gpu':-1}, {'gpu':True}, {'gpu':1.5},
    {'wait_seconds':float('nan')}, {'config':'/tmp/arbitrary.json'}, {'models':[{}]},
])
def test_bad_selection_cannot_create_or_spawn(http, app, updates):
    request = body(); request.update(updates)
    assert http('/api/runs', request)[0] == 400
    assert not list(app.root.iterdir())


def test_path_and_log_traversal_rejected(http, app):
    root, _ = p.create_pipeline(p.prepare_config(app.config), app.root)
    assert http('/api/runs/%2e%2e')[0] == 400
    assert http(f'/api/runs/{root.name}/logs?stage=../../etc/passwd')[0] == 404
    assert http(f'/api/runs/{root.name}/report/../../etc/passwd')[0] == 404
    symlink = app.root / 'linked'
    symlink.symlink_to(root)
    assert http('/api/runs/linked')[0] == 404
    assert http(f'/api/runs/{root.name}/report')[0] == 409
    (root / 'controller.log').write_text('x' * 100000)
    result = http(f'/api/runs/{root.name}/logs')[1]
    assert result['truncated'] and len(result['text']) == 65536


def test_real_child_stop_restart_resume_report_and_tamper(http, app, cpu_worker):
    status, run = http('/api/runs', body())
    assert status == 202
    run_id = run['run_id']
    restarted = Workbench(app.config, app.root)
    # Inherited dispatch lock protects the interval before pipeline lock acquired.
    assert restarted.status(run_id)['active']
    with pytest.raises(APIError) as exc:
        restarted.resume(run_id, {})
    assert exc.value.status == 409
    assert http(f'/api/runs/{run_id}/cancel', {})[0] == 202
    wait_until(lambda: not restarted.status(run_id)['active'])
    assert restarted.status(run_id)['phase'] == 'stopped'
    assert restarted.status(run_id)['stop_requested']
    assert http(f'/api/runs/{run_id}/resume', {})[0] == 202
    wait_until(lambda: not restarted.status(run_id)['active'])
    state = restarted.status(run_id)
    assert state['phase'] == 'complete', state
    assert state['completed'] == state['total'] == 3
    status, report = http(f'/api/runs/{run_id}/report')
    assert status == 200 and report['complete'] and report['formal']
    assert report['n_episodes'] == 500
    for name in ('report.md','cells.csv','tradeoffs.png','tradeoffs.svg'):
        assert http(f'/api/runs/{run_id}/report/{name}')[0] == 200
    assert 'report' in http(f'/api/runs/{run_id}/logs')[1]['text']
    (app.path(run_id) / 'report/report.md').write_text('tampered')
    assert http(f'/api/runs/{run_id}/report')[0] == 409


def test_unexpected_controller_exit_displayed_without_rewriting_evidence(app):
    root, state = p.create_pipeline(p.prepare_config(app.config), app.root)
    state['phase'] = 'running'
    p._atomic(root / 'run.json', state)
    assert app.status(root.name)['phase'] == 'interrupted'
    assert p.read(root / 'run.json')['phase'] == 'running'


def test_cli_web_dispatch(monkeypatch):
    from embodiq.ctl.cli import main
    from embodiq.web import server
    calls = []
    monkeypatch.setattr(server, 'serve', lambda *args: calls.append(args))
    assert main(['web','--config','cfg.json','--runs-root','runs','--port','9001']) == 0
    assert calls == [('cfg.json','runs',9001)]


def test_default_profile_is_formal_and_old_records_are_hidden(http, app):
    request = body()
    request.pop('profile')
    status, preview = http('/api/preview', request)
    assert status == 200 and preview['profile'] == 'formal' and preview['episodes'] == 500
    assert http('/api/capabilities')[1]['profiles'] == ['formal']
    root, state = p.create_pipeline(p.prepare_config(app.config), app.root)
    plan = p.read(root / 'plan.json')
    plan['config']['profile'] = 'legacy'
    plan['config']['trials_per_task'] = 1
    p._atomic(root / 'plan.json', plan)
    state['plan_sha256'] = p.digest(root / 'plan.json')
    p._atomic(root / 'run.json', state)
    assert http('/api/runs')[1] == {'runs': [], 'errors': []}
    assert http(f'/api/runs/{root.name}')[0] == 404
    assert http(f'/api/runs/{root.name}/report')[0] == 404
    assert (root / 'run.json').exists()
