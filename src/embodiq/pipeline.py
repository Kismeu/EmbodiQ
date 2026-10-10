"""One-command local evaluation pipeline. No torch imports in the controller.

The immutable plan owns paths and dependencies; run.json only owns progress.
Advisory locks are inherited by workers so a killed controller cannot start a
second worker over an orphan. Only this controller's process group is stopped.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from embodiq.config import build_plan, load_config
from embodiq.runner import (_atomic, _python_for_job, _resolve_checkpoint,
                            command_for_job, preflight_job)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'embodiq/pipeline/1'
AVAILABLE = ('fp16', 'rtn')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def capabilities():
    return {'models': ['openvla', 'vla_adapter', 'fastwam'],
            'methods': {'fp16': 'ready', 'rtn': 'ready', 'awq': 'pending', 'hsa': 'pending'},
            'benchmark': 'libero', 'profiles': ['formal'],
            'execution': 'local, sequential, exclusive GPU', 'fastwam_mode': 'first_frame'}


def prepare_config(path, *, models=None, methods=None, suites=None, profile=None):
    cfg = load_config(path)
    if models is not None:
        if not models or len(models) != len(set(models)) or set(models) - set(cfg['models']):
            raise ValueError('models must be unique registered models in the config')
        cfg['models'] = {m: cfg['models'][m] for m in models}
    if methods is not None:
        cfg['enabled_methods'] = methods
    if suites is not None:
        cfg['benchmark']['suites'] = suites
    if profile is not None:
        cfg['profile'] = profile
    cfg.setdefault('profile', 'formal')
    for method in cfg['enabled_methods']:
        if method not in AVAILABLE:
            raise ValueError(f'{method} is pending; automated execution supports fp16/rtn only')
        if not cfg.get('quantization', {}).get(method, {}).get('enabled', True):
            raise ValueError(f'{method} is disabled')
        q = cfg['quantization'][method]
        if method == 'rtn' and (q.get('w_bit', 4) != 4 or q.get('group_size', 128) != 128):
            raise ValueError('automated RTN currently requires W4/group128')
    if cfg.get('runtime', {}).get('fastwam_mode', 'first_frame') != 'first_frame':
        raise ValueError('automated comparison currently requires first_frame')
    if cfg.get('runtime', {}).get('compile_action_infer') or cfg.get('runtime', {}).get('hsa_enabled'):
        raise ValueError('compiled inference and HSA are not enabled in this pipeline')
    if not cfg.get('resource', {}).get('measure', True):
        raise ValueError('one-command evaluation requires the resource measurement stage')
    if not cfg.get('resource', {}).get('exclusive_gpu_required', True):
        raise ValueError('resource measurement requires an exclusive GPU')
    build_plan(cfg)  # validate before creating any files
    return cfg


def create_pipeline(config, out_root=None):
    jobs = build_plan(config)
    run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:10]
    root = Path(out_root or config['outputs']['root']).resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    stages = []
    for job in jobs:
        checkpoint = _resolve_checkpoint(job)
        if checkpoint is None:
            raise ValueError(f'unresolved checkpoint: {job["model"]}/{job["suite"]}')
        job.update(checkpoint=str(checkpoint.resolve()), python=str(Path(_python_for_job(job)).absolute()))
        for k in ('repo_root', 'dataset_stats', 'checkpoint_root'):
            if job.get(k):
                job[k] = str(Path(job[k]).resolve())
        sid = f'rollout.{job["model"]}.{job["quant"]}.{job["suite"]}'
        stages.append({'job_id': sid, 'stage': 'rollout', 'job': job,
                       'depends_on': [], 'status': 'planned', 'attempts': 0})
    for model in config['models']:
        for method in config['enabled_methods']:
            deps = [s['job_id'] for s in stages if s['stage'] == 'rollout'
                    and s['job']['model'] == model and s['job']['quant'] == method]
            stages.append({'job_id': f'resource.{model}.{method}', 'stage': 'resource',
                           'model': model, 'method': method, 'depends_on': deps,
                           'status': 'planned', 'attempts': 0})
    deps = [s['job_id'] for s in stages]
    stages.append({'job_id': 'report', 'stage': 'report', 'depends_on': deps,
                   'status': 'planned', 'attempts': 0})
    source = Path(config['resource']['fixed_input_manifest'])
    if not source.is_absolute():
        source = ROOT / source
    plan = {'schema': SCHEMA, 'run_id': run_id, 'config': config, 'jobs': jobs,
            'stages': [{k: v for k, v in s.items() if k not in {'status', 'attempts'}} for s in stages],
            'resource_inputs_source': str(source.resolve()),
            'episode_source': str(ROOT / 'results/baseline_unified_v3_20260919/episode_manifest.json')}
    _atomic(root / 'plan.json', plan)
    state = {'schema': SCHEMA, 'run_id': run_id, 'phase': 'planned',
             'created': time.strftime('%FT%T%z'), 'plan_sha256': digest(root / 'plan.json'),
             'jobs': stages}
    _atomic(root / 'run.json', state)
    return root, state


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'lock_busy:{path}') from exc
        yield f


def gpu_processes(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, timeout=15, check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def pipeline_preflight(plan):
    reasons = []
    for job in plan['jobs']:
        reasons.extend(f'{job["model"]}/{job["suite"]}: {r}' for r in preflight_job(job))
        if not job.get('repo_root') or not Path(job['repo_root']).is_dir():
            reasons.append(f'{job["model"]}: repo_root missing')
    for key in ('resource_inputs_source', 'episode_source'):
        if not Path(plan[key]).is_file():
            reasons.append(f'{key} missing: {plan[key]}')
    return sorted(set(reasons))


def _source_files(plan):
    files = set((ROOT / 'src').rglob('*.py')) | set((ROOT / 'tools').glob('*.py'))
    files.add(ROOT / 'pyproject.toml')
    for job in plan['jobs']:
        repo = Path(job['repo_root'])
        dirs = ('src', 'configs', 'experiments') if job['model'] == 'fastwam' else ('prismatic', 'experiments', 'vla-scripts')
        for directory in dirs:
            files.update(p for p in (repo / directory).rglob('*')
                         if p.is_file() and p.suffix in {'.py', '.yaml', '.yml'})
    files.update((ROOT.parent / 'LIBERO' / 'libero').rglob('*.py'))
    return {p.resolve() for p in files if '__pycache__' not in p.parts}


def bind_inputs(root, plan, *, stop=lambda: False):
    """Rehash real inputs on launch/resume; never trust path-only identity."""
    manifest = read(plan['resource_inputs_source'])
    actual = hashlib.sha256(json.dumps(manifest['tasks'], sort_keys=True).encode()).hexdigest()
    if manifest.get('manifest_sha256') != actual:
        raise ValueError('resource manifest content mismatch')
    files = _source_files(plan)
    for key in ('resource_inputs_source', 'episode_source'):
        files.add(Path(plan[key]))
    for job in plan['jobs']:
        ckpt = Path(job['checkpoint'])
        weights = {ckpt} if ckpt.is_file() else {
            p for p in ckpt.rglob('*') if p.is_file() and
            p.suffix in {'.safetensors', '.bin', '.pt', '.pth', '.json', '.model', '.py'}
            and '.cache' not in p.parts and '.git' not in p.parts}
        if not weights:
            raise ValueError(f'checkpoint has no model files: {ckpt}')
        files.update(weights)
        if job.get('dataset_stats'):
            files.add(Path(job['dataset_stats']))
    selected_suites = set(plan['config']['benchmark']['suites'])
    for task in manifest['tasks']:
        if task['suite'] in selected_suites:
            for row in task['observations']:
                p = ROOT / row['path']
                if digest(p) != row['sha256']:
                    raise ValueError(f'resource observation changed: {p}')
                files.add(p)
    episodes = read(plan['episode_source'])
    for task in episodes['tasks']:
        if task['suite'] in selected_suites:
            p = Path(task['initial_state_path'])
            if digest(p) != task['initial_state_file_sha256']:
                raise ValueError(f'initial states changed: {p}')
            files.add(p)
    hashes = {}
    for p in sorted(files):
        if stop():
            raise InterruptedError('stop requested during input hashing')
        hashes[str(p)] = digest(p)
    envs = {}
    code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata.get('Name',''),d.version) for d in m.distributions())}))"
    for binary in sorted({j['python'] for j in plan['jobs']}):
        envs[binary] = json.loads(subprocess.check_output([binary, '-c', code], text=True))
    doc = {'files': hashes, 'environments': envs, 'plan_sha256': digest(root / 'plan.json')}
    target = root / 'input_identity.json'
    if target.exists() and read(target) != doc:
        raise ValueError('inputs, code or environments changed; create a new run instead of mixing results')
    _atomic(target, doc)
    (root / 'resource_bench').mkdir(exist_ok=True)
    _atomic(root / 'resource_bench/inputs_v1.json', manifest)
    _atomic(root / 'episode_manifest.json', episodes)
    return digest(target)


def stage_command(root, plan, stage, gpu):
    if stage['stage'] == 'rollout':
        j = stage['job']
        out = root / 'results' / j['quant'] / j['model'] / j['suite']
        return command_for_job(j, output=out, gpu=gpu)
    job = next(j for j in plan['jobs'] if stage['stage'] == 'report' or
               (j['model'] == stage['model'] and j['quant'] == stage['method']))
    _, env = command_for_job(job, output=root, gpu=gpu)
    if stage['stage'] == 'report':
        return [job['python'], '-m', 'embodiq.pipeline_report', str(root)], env
    cmd = [job['python'], str(ROOT / 'tools/run_resource_bench.py'),
           '--model', stage['model'], '--quant', stage['method'], '--gpu', str(gpu),
           '--job-plan', str(root / 'plan.json'),
           '--inputs', str(root / 'resource_bench/inputs_v1.json'),
           '--suites', ','.join(plan['config']['benchmark']['suites']),
           '--out', str(root / 'resource_bench' / f'{stage["model"]}_{stage["method"]}.json')]
    return cmd, env


def terminate_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    except ProcessLookupError:
        proc.wait()


def run_child(cmd, env, log, *, stop, fds=(), gpu=None, on_start=lambda pid: None):
    with Path(log).open('a') as stream:
        stream.write(json.dumps({'command': cmd, 'started': time.strftime('%FT%T%z')}) + '\n')
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=tuple(fds))
        last_gpu_check = 0.0
        try:
            on_start(proc.pid)
            while proc.poll() is None:
                if stop():
                    terminate_group(proc)
                    return 130
                if gpu is not None and time.monotonic() - last_gpu_check >= 2:
                    last_gpu_check = time.monotonic()
                    foreign = []
                    for pid in gpu_processes(gpu):
                        try:
                            if os.getpgid(pid) != proc.pid:
                                foreign.append(pid)
                        except ProcessLookupError:
                            pass
                    if foreign:
                        terminate_group(proc)
                        raise RuntimeError(f'GPU contention during execution: {foreign}')
                time.sleep(.5)
            return proc.returncode
        finally:
            terminate_group(proc)


def request_stop(root):
    root = Path(root).resolve()
    state = read(root / 'run.json')
    if state['schema'] != SCHEMA:
        raise ValueError('not a pipeline run')
    (root / 'STOP_REQUESTED').touch()
    return {'run_id': state['run_id'], 'phase': state['phase'], 'stop_requested': True}


def execute_pipeline(root, *, gpu=0, resume=False, wait_seconds=0):
    root = Path(root).resolve()
    with lock(root / '.controller.lock') as controller_lock:
        state = read(root / 'run.json')
        if state['schema'] != SCHEMA:
            raise ValueError('not a pipeline run')
        if digest(root / 'plan.json') != state['plan_sha256']:
            raise ValueError('plan was edited; create a new run')
        plan = read(root / 'plan.json')
        build_plan(plan['config'])  # Reject obsolete or incomplete evaluation protocols.
        definitions = plan['stages']
        if len(definitions) != len(state['jobs']) or any(
                any(s.get(k) != v for k, v in expected.items())
                for s, expected in zip(state['jobs'], definitions)):
            raise ValueError('stage definitions differ from immutable plan')
        if (root / 'STOP_REQUESTED').exists() and not resume:
            state['phase'] = 'stopped'
            _atomic(root / 'run.json', state)
            return 130
        if resume:
            (root / 'STOP_REQUESTED').unlink(missing_ok=True)
        stop = lambda: (root / 'STOP_REQUESTED').exists()
        previous_handlers = {}
        def handle_stop(signum, frame):
            (root / 'STOP_REQUESTED').touch()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, handle_stop)
        def save():
            state['updated'] = time.strftime('%FT%T%z')
            state['summary'] = {s: sum(j['status'] == s for j in state['jobs'])
                                for s in ('planned', 'running', 'complete', 'blocked', 'failed', 'interrupted')}
            _atomic(root / 'run.json', state)
        current = None
        try:
            reasons = pipeline_preflight(plan)
            if reasons:
                state.update(phase='blocked', reasons=reasons)
                save()
                return 2
            state.update(phase='preflight', gpu=gpu, controller_pid=os.getpid())
            state.pop('reasons', None)
            save()
            print(f'[{state["run_id"]}] verifying checkpoint, input and source hashes', flush=True)
            identity = bind_inputs(root, plan, stop=stop)
            if state.get('input_identity_sha256') not in (None, identity):
                raise ValueError('input identity file changed')
            state['input_identity_sha256'] = identity
            # Completed stages are only reusable if their saved outputs still match.
            for stage in state['jobs']:
                if stage['status'] == 'complete':
                    if not stage.get('outputs') or any(not Path(p).is_file() or digest(p) != h
                                                     for p, h in stage['outputs'].items()):
                        raise ValueError(f'completed stage output changed: {stage["job_id"]}')
                elif stage['status'] == 'running':
                    stage['status'] = 'interrupted'
            for current in state['jobs']:
                if current['status'] == 'complete':
                    continue
                if stop():
                    raise InterruptedError('stop requested')
                done = {s['job_id'] for s in state['jobs'] if s['status'] == 'complete'}
                if not set(current['depends_on']).issubset(done):
                    raise ValueError(f'dependencies not complete: {current["job_id"]}')
                deadline = time.monotonic() + wait_seconds
                while True:
                    if stop():
                        raise InterruptedError('stop requested while waiting for GPU')
                    try:
                        gpu_lock = contextlib.ExitStack()
                        gpu_fd = None
                        if current['stage'] != 'report':
                            gpu_fd = gpu_lock.enter_context(lock(ROOT / 'results/.gpu_locks' / f'gpu{gpu}.lock'))
                            active = gpu_processes(gpu)
                            if active:
                                raise RuntimeError(f'gpu_busy:{gpu}: {active}')
                        break
                    except (RuntimeError, subprocess.SubprocessError):
                        gpu_lock.close()
                        if time.monotonic() >= deadline:
                            raise
                        state['phase'] = 'waiting_for_gpu'
                        save()
                        time.sleep(min(2, max(.1, deadline - time.monotonic())))
                with gpu_lock:
                    current.update(status='running', attempts=current['attempts'] + 1)
                    current.pop('error', None)
                    state['phase'] = 'running'
                    log = root / 'logs' / f'{current["job_id"]}.log'
                    log.parent.mkdir(exist_ok=True)
                    current['log'] = str(log)
                    cmd, env = stage_command(root, plan, current, gpu)
                    current['command'] = cmd
                    save()
                    print(f'[{state["run_id"]}] {current["job_id"]}', flush=True)
                    def started(pid):
                        current['pid'] = pid
                        save()
                    code = run_child(cmd, env, log, stop=stop, gpu=gpu if gpu_fd else None,
                                     fds=[f.fileno() for f in (controller_lock, gpu_fd) if f is not None],
                                     on_start=started)
                    current['returncode'] = code
                    current.pop('pid', None)
                    if code == 130 or stop():
                        raise InterruptedError('worker stopped')
                    if code:
                        current.update(status='failed', error=f'worker exited {code}; see {log}')
                        state['phase'] = 'failed'
                        save()
                        return code
                    from embodiq.pipeline_report import validate_stage
                    outputs = validate_stage(root, plan, current)
                    current.update(status='complete', outputs={str(p): digest(p) for p in outputs})
                    save()
            state['phase'] = 'complete'
            save()
            return 0
        except InterruptedError as exc:
            if current and current['status'] == 'running':
                current.update(status='interrupted', error=str(exc))
                current.pop('pid', None)
            state['phase'] = 'stopped'
            save()
            return 130
        except Exception as exc:
            if current and current['status'] != 'complete':
                current.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
                current.pop('pid', None)
            state.update(phase='blocked', reasons=[f'{type(exc).__name__}: {exc}'])
            save()
            print(state['reasons'][0], file=sys.stderr, flush=True)
            return 2
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
