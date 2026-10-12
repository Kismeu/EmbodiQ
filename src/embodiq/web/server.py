"""Loopback-only HTTP API. Controllers survive HTTP restarts via inherited locks."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from embodiq import pipeline as p
from embodiq.config import build_plan
from embodiq.web.resources import GPUMemoryMonitor

STATIC = Path(__file__).parent / 'static'
ARTIFACTS = {'report.md': 'text/markdown; charset=utf-8', 'cells.csv': 'text/csv; charset=utf-8',
             'tradeoffs.png': 'image/png', 'tradeoffs.svg': 'image/svg+xml'}


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def busy(path):
    try:
        with p.lock(path):
            return False
    except RuntimeError:
        return True


class Workbench:
    def __init__(self, config, runs_root):
        self.config = Path(config).resolve()
        self.root = Path(runs_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.mutex = threading.Lock()
        self.memory_monitor = GPUMemoryMonitor()

    def path(self, run_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
            raise APIError(400, '运行编号无效')
        path = self.root / run_id
        if path.is_symlink() or path.resolve().parent != self.root or not (path / 'run.json').is_file():
            raise APIError(404, '找不到该运行')
        if p.read(path / 'run.json').get('schema') != p.SCHEMA:
            raise APIError(400, '仅支持统一自动评测运行')
        if p.read(path / 'plan.json')['config'].get('profile', 'formal') != 'formal':
            raise APIError(404, '该记录不属于正式评测')
        return path

    def capabilities(self):
        cfg = p.prepare_config(self.config)
        caps = p.capabilities()
        caps.update(models=list(cfg['models']), suites=cfg['benchmark']['suites'],
                    csrf_token=self.token, default_gpu=0, runs_root=str(self.root))
        return caps

    def selection(self, body):
        if not isinstance(body, dict) or set(body) - {'models', 'methods', 'suites', 'profile', 'gpu', 'wait_seconds'}:
            raise APIError(400, '请求包含未知字段')
        for field in ('models', 'methods', 'suites'):
            value = body.get(field)
            if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
                raise APIError(400, f'{field} 必须是非空列表')
        if body.get('profile', 'formal') != 'formal':
            raise APIError(400, '仅支持正式评测')
        gpu, wait = body.get('gpu', 0), body.get('wait_seconds', 0)
        if type(gpu) is not int or not 0 <= gpu <= 255 or type(wait) is not int or not 0 <= wait <= 86400:
            raise APIError(400, 'GPU 需为 0–255 的整数，等待秒数需为 0–86400 的整数')
        try:
            cfg = p.prepare_config(self.config, profile='formal',
                                   **{k: body[k] for k in ('models', 'methods', 'suites')})
        except (ValueError, KeyError, TypeError) as exc:
            raise APIError(400, str(exc)) from exc
        return cfg, gpu, wait

    def preview(self, body):
        cfg, gpu, wait = self.selection(body)
        jobs = build_plan(cfg)
        resources = len(cfg['models']) * len(cfg['enabled_methods'])
        return {'profile': cfg['profile'], 'rollout_stages': len(jobs), 'resource_stages': resources,
                'total_stages': len(jobs) + resources + 1,
                'episodes': sum(len(j['task_ids']) * j['trials'] for j in jobs),
                'gpu': gpu, 'wait_seconds': wait}

    def start(self, path, gpu, wait, *, resume):
        # Held by the spawned controller for its whole lifetime, including startup
        # before execute_pipeline acquires its own lock. Survives HTTP restarts.
        try:
            with p.lock(path / '.web-dispatch.lock') as dispatch:
                if busy(path / '.controller.lock'):
                    raise APIError(409, '该运行已有控制器或工作进程')
                if resume:
                    (path / 'STOP_REQUESTED').unlink(missing_ok=True)
                env = os.environ.copy()
                env['PYTHONPATH'] = str(p.ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
                env['PYTHONUNBUFFERED'] = '1'
                with (path / 'controller.log').open('a') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'embodiq.web.worker', str(path),
                                             str(gpu), str(wait)], cwd=p.ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, pass_fds=(dispatch.fileno(),))
                # Reap children without making their lifetime depend on this server.
                threading.Thread(target=proc.wait, daemon=True).start()
        except RuntimeError as exc:
            raise APIError(409, '该运行已经启动，请刷新状态') from exc

    def create(self, body):
        cfg, gpu, wait = self.selection(body)
        with self.mutex:
            path, state = p.create_pipeline(cfg, self.root)
            self.start(path, gpu, wait, resume=False)
        return self.status(state['run_id'])

    def status(self, run_id):
        path = self.path(run_id)
        state = p.read(path / 'run.json')
        plan = p.read(path / 'plan.json')
        active = busy(path / '.web-dispatch.lock') or busy(path / '.controller.lock')
        phase = state['phase']
        if active and phase in {'planned', 'stopped', 'blocked', 'failed', 'complete'}:
            phase = 'starting'
        elif not active and phase in {'running', 'preflight', 'waiting_for_gpu'}:
            phase = 'interrupted'
        cfg = plan['config']
        return {'run_id': run_id, 'phase': phase, 'saved_phase': state['phase'], 'active': active,
                'created': state.get('created'), 'profile': cfg['profile'], 'gpu': state.get('gpu'),
                'models': list(cfg['models']), 'methods': cfg['enabled_methods'],
                'suites': cfg['benchmark']['suites'], 'reasons': state.get('reasons', []),
                'stop_requested': (path / 'STOP_REQUESTED').exists(),
                'completed': sum(j['status'] == 'complete' for j in state['jobs']),
                'total': len(state['jobs']),
                'jobs': [{k: j[k] for k in ('job_id', 'stage', 'status', 'attempts', 'error') if k in j}
                         for j in state['jobs']]}

    def runs(self):
        rows, errors = [], []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir() or path.is_symlink() or not (path / 'run.json').exists():
                continue
            try:
                if p.read(path / 'plan.json')['config'].get('profile', 'formal') != 'formal':
                    continue
                rows.append(self.status(path.name))
            except (APIError, OSError, ValueError, KeyError) as exc:
                errors.append({'run_id': path.name, 'error': str(exc)})
        return {'runs': rows, 'errors': errors}

    def cancel(self, run_id):
        with self.mutex:
            return p.request_stop(self.path(run_id))

    def resume(self, run_id, body):
        if not isinstance(body, dict) or set(body) - {'gpu', 'wait_seconds'}:
            raise APIError(400, '续跑仅允许指定 GPU 和等待秒数')
        path = self.path(run_id)
        cfg = p.read(path / 'plan.json')['config']
        _, gpu, wait = self.selection(dict(models=list(cfg['models']), methods=cfg['enabled_methods'],
                                          suites=cfg['benchmark']['suites'], profile=cfg['profile'], **body))
        with self.mutex:
            self.start(path, gpu, wait, resume=True)
        return self.status(run_id)

    def logs(self, run_id, stage=None):
        root = self.path(run_id)
        path = root / 'controller.log'
        if stage:
            if stage not in {j['job_id'] for j in p.read(root / 'run.json')['jobs']}:
                raise APIError(404, '找不到该阶段')
            path = root / 'logs' / f'{stage}.log'
        if path.resolve().is_relative_to(root) is False:
            raise APIError(400, '日志路径无效')
        if not path.exists():
            return {'text': '', 'truncated': False}
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 65536))
            return {'text': stream.read(65536).decode('utf-8', errors='replace'), 'truncated': size > 65536}

    def report(self, run_id, artifact=None):
        root = self.path(run_id)
        if busy(root / '.web-dispatch.lock'):
            raise APIError(409, '运行尚未结束')
        try:
            with p.lock(root / '.controller.lock'):
                state = p.read(root / 'run.json')
                if state['phase'] != 'complete':
                    raise APIError(409, '尚无完成且通过校验的报告')
                if p.digest(root / 'plan.json') != state['plan_sha256']:
                    raise APIError(409, '运行计划已变更，报告校验失败')
                for stage in state['jobs']:
                    if stage['status'] != 'complete' or not stage.get('outputs'):
                        raise APIError(409, '运行阶段尚未完成')
                    for name, expected in stage['outputs'].items():
                        target = Path(name).resolve()
                        if not target.is_relative_to(root) or not target.is_file() or p.digest(target) != expected:
                            raise APIError(409, '结果文件已变更，报告校验失败')
                name = artifact or 'report.json'
                target = root / 'report' / name
                outputs = next(s['outputs'] for s in state['jobs'] if s['stage'] == 'report')
                if str(target) not in outputs:
                    raise APIError(404, '报告未包含该文件')
                return target.read_bytes() if artifact else p.read(target)
        except RuntimeError as exc:
            raise APIError(409, '运行正在执行，请稍后查看报告') from exc


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log CSRF tokens or request bodies.
        sys.stderr.write(f'{self.address_string()} {fmt % args}\n')

    def reply(self, code, value, mime='application/json; charset=utf-8'):
        payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.handle_request(False)

    def do_POST(self):
        self.handle_request(True)

    def handle_request(self, post):
        try:
            port = self.server.server_port
            allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if self.headers.get('Host') not in allowed_hosts:
                raise APIError(403, '仅接受本地工作台请求')
            app = self.server.app
            url = urlsplit(self.path)
            route = url.path.strip('/').split('/')
            body = None
            if post:
                if self.headers.get('X-Embodiq-Token') != app.token:
                    raise APIError(403, '页面会话已失效，请刷新页面')
                origin = self.headers.get('Origin')
                if origin and origin not in {f'http://{host}' for host in allowed_hosts}:
                    raise APIError(403, '请求来源无效')
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise APIError(415, '请使用 application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise APIError(400, '请求体需为 1–16384 字节')
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
            if not post and url.path in {'/', '/app.js', '/style.css'}:
                name = 'index.html' if url.path == '/' else url.path[1:]
                mime = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript; charset=utf-8',
                        'style.css': 'text/css; charset=utf-8'}[name]
                return self.reply(200, (STATIC / name).read_bytes(), mime)
            if route == ['api', 'capabilities'] and not post:
                return self.reply(200, app.capabilities())
            if route == ['api', 'gpus'] and not post:
                return self.reply(200, app.memory_monitor.devices())
            if route == ['api', 'gpu-memory'] and not post:
                query = parse_qs(url.query, keep_blank_values=True)
                indices = query.get('gpu', ['0'])
                if (set(query) - {'gpu'} or len(indices) != 1
                        or not re.fullmatch(r'[0-9]{1,3}', indices[0]) or int(indices[0]) > 255):
                    raise APIError(400, '显卡编号需为 0–255 的整数')
                return self.reply(200, app.memory_monitor.sample(int(indices[0])))
            if route == ['api', 'preview'] and post:
                return self.reply(200, app.preview(body))
            if route == ['api', 'runs']:
                return self.reply(202 if post else 200, app.create(body) if post else app.runs())
            if len(route) in {3, 4, 5} and route[:2] == ['api', 'runs']:
                run_id = route[2]
                action = route[3] if len(route) >= 4 else ''
                if len(route) == 3 and not post:
                    return self.reply(200, app.status(run_id))
                if len(route) == 4:
                    if post and action == 'cancel':
                        return self.reply(202, app.cancel(run_id))
                    if post and action == 'resume':
                        return self.reply(202, app.resume(run_id, body))
                    if not post and action == 'logs':
                        return self.reply(200, app.logs(run_id, parse_qs(url.query).get('stage', [None])[0]))
                    if not post and action == 'report':
                        return self.reply(200, app.report(run_id))
                if len(route) == 5 and not post and action == 'report' and route[4] in ARTIFACTS:
                    return self.reply(200, app.report(run_id, route[4]), ARTIFACTS[route[4]])
            raise APIError(404, '接口不存在')
        except APIError as exc:
            self.reply(exc.status, {'error': exc.message})
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except (OSError, subprocess.SubprocessError) as exc:
            self.reply(500, {'error': f'本地服务错误：{exc}'})


def serve(config, runs_root, port=8765):
    app = Workbench(config, runs_root)
    with p.lock(app.root / '.web-server.lock'):
        with Server(('127.0.0.1', port), app) as server:
            print(f'EmbodiQ 工作台：http://127.0.0.1:{server.server_port}', flush=True)
            print('关闭网页服务不会停止评测；请在工作台或 CLI 中停止任务。', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
