"""Reports bound to one immutable run; historical result roots are never read."""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

from embodiq.core.admission import finite_positive
from embodiq.core.protocol import PROTOCOLS
from embodiq.pipeline import ROOT, digest, read


def matrix_module():
    spec = importlib.util.spec_from_file_location('pipeline_matrix', ROOT / 'tools/report_matrix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rollout_root(root, job):
    return root / 'results' / job['quant'] / job['model'] / job['suite']


def rollout_records(root, job):
    return sorted(rollout_root(root, job).rglob('*_results.json'))


def validate_rollout(root, plan, job):
    paths = rollout_records(root, job)
    ids = []
    frozen = {(t['suite'], t['task_id']): t for t in read(root / 'episode_manifest.json')['tasks']}
    expected = dataclasses.asdict(dataclasses.replace(PROTOCOLS[job['protocol']],
                               num_trials_per_task=job['trials'], seed_base=job['seed'],
                               save_videos=job.get('save_videos', False)))
    expected = json.loads(json.dumps(expected))
    conditions = set()
    matrix = matrix_module()
    for path in paths:
        d = read(path)
        task = d['task_id']
        ids.append(task)
        if d['task_suite'] != job['suite'] or d['policy']['family'] != job['model']:
            raise ValueError(f'wrong model/suite in {path}')
        if Path(d['policy']['checkpoint']).resolve() != Path(job['checkpoint']).resolve():
            raise ValueError(f'checkpoint mismatch in {path}')
        if d['quant']['method'] != job['quant'] or d['protocol'] != expected:
            raise ValueError(f'quantization/protocol mismatch in {path}')
        scope = [] if job['quant'] == 'fp16' else job['scopes']
        if set(d['quant'].get('scope', [])) != set(scope):
            raise ValueError(f'quantization scope mismatch in {path}')
        if job['quant'] == 'rtn' and (d['quant'].get('mode'), d['quant'].get('w_bits'), d['quant'].get('group_size')) != ('packed', 4, 128):
            raise ValueError(f'RTN precision/backend mismatch in {path}')
        conditions.add(matrix.condition_identity(d['policy'], d['quant'], d['protocol']))
        episodes = d['episodes']
        if len(episodes) != job['trials'] or [e['episode_idx'] for e in episodes] != list(range(job['trials'])):
            raise ValueError(f'incomplete/duplicate episodes in {path}')
        f = frozen[(job['suite'], task)]
        for ep in episodes:
            ref = f['episodes'][ep['episode_idx']]
            aux = ep.get('aux') or {}
            if ep['seed'] != ref['seed'] or aux.get('initial_state_sha256') != ref['state_sha256']:
                raise ValueError(f'episode identity mismatch in {path}')
            if ep['status'] not in {'success', 'timeout'}:
                raise ValueError(f'error/invalid episode in {path}')
        m = d['metrics']
        counts = {'n_episodes': len(episodes), 'n_success': sum(e['status'] == 'success' for e in episodes),
                  'n_timeout': sum(e['status'] == 'timeout' for e in episodes),
                  'n_error': 0, 'n_invalid': 0, 'n_policy_error': 0}
        if any(m.get(k) != value for k, value in counts.items()):
            raise ValueError(f'metric/episode mismatch in {path}')
        if d.get('run', {}).get('status') != 'ok':
            raise ValueError(f'run not finalized in {path}')
    if sorted(ids) != sorted(job['task_ids']) or len(conditions) != 1:
        raise ValueError(f'missing/extra/duplicate tasks or mixed conditions: {job["model"]}/{job["suite"]}')
    if plan['config']['profile'] == 'formal':
        cells, problems = matrix.load_cells({job['quant']: [str(rollout_root(root, job))]})
        matrix.validate_episode_manifest(cells, problems, root / 'episode_manifest.json')
        cell = cells.get((job['model'], job['quant'], job['suite']))
        if problems or cell is None or not cell['complete']:
            raise ValueError(f'formal admission failed: {problems}')
    return paths


def validate_resource(root, plan, model, method):
    path = root / 'resource_bench' / f'{model}_{method}.json'
    d = read(path)
    smoke = plan['config']['profile'] == 'smoke'
    manifest = read(root / 'resource_bench/inputs_v1.json')
    if d.get('schema') != 'embodiq/resource_bench/1' or d.get('phase') != 'complete':
        raise ValueError('resource result incomplete')
    if (d.get('model'), d.get('quant')) != (model, method):
        raise ValueError('resource model/method mismatch')
    if d.get('contended') is not False or d.get('contended_before') or d.get('contended_after'):
        raise ValueError('resource measurement was contended')
    if d.get('validation_only') is not smoke:
        raise ValueError('resource profile mismatch')
    if d.get('plan_sha256') != digest(root / 'plan.json'):
        raise ValueError('resource plan mismatch')
    if d.get('inputs_manifest_sha256') != manifest['manifest_sha256']:
        raise ValueError('resource input manifest mismatch')
    if d.get('warmup_queries') != (2 if smoke else 20) or d.get('measured_passes') != (1 if smoke else 3):
        raise ValueError('resource warmup/pass mismatch')
    if set(d['suites']) != set(plan['config']['benchmark']['suites']):
        raise ValueError('resource suites incomplete')
    for suite, b in d['suites'].items():
        job = next(j for j in plan['jobs'] if (j['model'], j['quant'], j['suite']) == (model, method, suite))
        p = read(rollout_records(root, job)[0])
        if Path(b['checkpoint']).resolve() != Path(job['checkpoint']).resolve():
            raise ValueError('resource checkpoint mismatch')
        if b['n_inputs'] != (1 if smoke else 50) or b['n_queries_measured'] != (1 if smoke else 150):
            raise ValueError('resource measurement count mismatch')
        if b['passes'] != (1 if smoke else 3):
            raise ValueError('resource passes mismatch')
        if b['chunk_size'] != p['policy']['chunk_size'] or b['replan_steps'] != p['policy']['default_replan_steps']:
            raise ValueError('resource action schedule mismatch')
        for key in ('model_id', 'torch_dtype'):
            if b['policy'][key] != p['policy'][key]:
                raise ValueError(f'resource policy {key} mismatch')
        if model == 'fastwam':
            for key in ('action_infer_mode', 'dataset_stats_sha256', 'compile_action_infer', 'sigma_shift', 'num_inference_steps', 'input_hw'):
                if b['policy']['extra'].get(key) != p['policy']['extra'].get(key):
                    raise ValueError(f'resource temporal/runtime {key} mismatch')
        if b['quant']['method'] != method or set(b['quant']['scope']) != set(p['quant']['scope']):
            raise ValueError('resource quant scope mismatch')
        if method != 'fp16' and b['quant']['spec_hash'] != p['quant']['spec_hash']:
            raise ValueError('resource quant spec mismatch')
        if any(not finite_positive(b['memory'].get(k)) for k in ('weights_bytes', 'peak_alloc_bytes', 'nvml_peak_bytes', 'load_peak_alloc_bytes')):
            raise ValueError('invalid resource memory metrics')
        if not finite_positive(b['latency']['query_s'].get('mean')) or not finite_positive(b['latency'].get('per_env_step_s')):
            raise ValueError('invalid resource latency metrics')
    return [path]


def validate_stage(root, plan, stage):
    if stage['stage'] == 'rollout':
        return validate_rollout(root, plan, stage['job'])
    if stage['stage'] == 'resource':
        return validate_resource(root, plan, stage['model'], stage['method'])
    paths = [root / 'report' / name for name in ('report.json', 'report.md', 'cells.csv', 'tradeoffs.png', 'tradeoffs.svg')]
    d = read(paths[0])
    if d['run_id'] != plan['run_id'] or d['problems'] or not d['complete']:
        raise ValueError('report did not pass admission')
    if d['plan_sha256'] != digest(root / 'plan.json'):
        raise ValueError('report plan mismatch')
    if not all(p.is_file() for p in paths):
        raise ValueError('report outputs missing')
    return paths


def generate_report(root):
    root = Path(root).resolve()
    state, plan = read(root / 'run.json'), read(root / 'plan.json')
    if state['plan_sha256'] != digest(root / 'plan.json'):
        raise ValueError('plan mismatch')
    if state.get('input_identity_sha256') != digest(root / 'input_identity.json'):
        raise ValueError('input identity mismatch')
    problems, rows = [], []
    if plan['config']['profile'] == 'formal':
        matrix = matrix_module()
        sources = {m: [str(root / 'results' / m)] for m in plan['config']['enabled_methods']}
        cells, issues = matrix.load_cells(sources)
        matrix.validate_episode_manifest(cells, issues, root / 'episode_manifest.json')
        expected_keys = {(j['model'], j['quant'], j['suite']) for j in plan['jobs']}
        if set(cells) != expected_keys or not all(c['complete'] for c in cells.values()):
            issues.append('requested formal cells are missing, extra or incomplete')
        problems.extend(issues)
    resources = {}
    for stage in state['jobs']:
        if stage['stage'] == 'report':
            continue
        try:
            if stage['status'] != 'complete':
                raise ValueError(f'stage status is {stage["status"]}')
            paths = validate_stage(root, plan, stage)
            if any(stage.get('outputs', {}).get(str(p)) != digest(p) for p in paths):
                raise ValueError('stage output changed after completion')
            if stage['stage'] == 'resource':
                resources[(stage['model'], stage['method'])] = read(paths[0])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            problems.append(f'{stage["job_id"]}: {exc}')
    for job in plan['jobs']:
        files = rollout_records(root, job)
        episodes = []
        for path in files:
            try:
                episodes.extend(read(path)['episodes'])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                problems.append(f'{path}: unreadable episodes: {exc}')
        n, ok = len(episodes), sum(ep['status'] == 'success' for ep in episodes)
        bench = resources.get((job['model'], job['quant']), {}).get('suites', {}).get(job['suite'])
        rows.append({'model': job['model'], 'method': job['quant'], 'suite': job['suite'],
                     'n_episodes': n, 'n_success': ok, 'success_rate': ok / n if n and not problems else None,
                     'query_ms': bench['latency']['query_s']['mean'] * 1000 if bench else None,
                     'per_action_ms': bench['latency']['per_env_step_s'] * 1000 if bench else None,
                     'steady_nvml_gib': bench['memory']['nvml_peak_bytes'] / 2**30 if bench else None,
                     'steady_alloc_gib': bench['memory']['peak_alloc_bytes'] / 2**30 if bench else None,
                     'load_alloc_gib': bench['memory']['load_peak_alloc_bytes'] / 2**30 if bench else None,
                     'result_files': [str(p) for p in files]})
    aggregates = []
    for model in plan['config']['models']:
        for method in plan['config']['enabled_methods']:
            selected = [r for r in rows if r['model'] == model and r['method'] == method]
            good = not problems and all(r['success_rate'] is not None for r in selected)
            avg = sum(r['success_rate'] for r in selected) / len(selected) if good else None
            full = len(selected) == 4 and plan['config']['profile'] == 'formal'
            aggregates.append({'model': model, 'method': method,
                               'selected_suites_success_rate': avg,
                               'four_suite_macro_success_rate': avg if full else None,
                               'steady_nvml_gib': max(r['steady_nvml_gib'] for r in selected) if good else None,
                               'query_ms': sum(r['query_ms'] for r in selected) / len(selected) if good else None,
                               'per_action_ms': sum(r['per_action_ms'] for r in selected) / len(selected) if good else None})
    for row in aggregates:
        base = next((r for r in aggregates if r['model'] == row['model'] and r['method'] == 'fp16'), None)
        row['delta_pp_vs_fp16'] = (100 * (row['selected_suites_success_rate'] - base['selected_suites_success_rate'])
                                    if base and row['selected_suites_success_rate'] is not None and base['selected_suites_success_rate'] is not None else None)
    doc = {'schema': 'embodiq/pipeline-report/1', 'run_id': plan['run_id'],
           'profile': plan['config']['profile'], 'formal': plan['config']['profile'] == 'formal' and not problems,
           'complete': not problems, 'plan_sha256': digest(root / 'plan.json'),
           'input_identity_sha256': digest(root / 'input_identity.json'),
           'n_episodes': sum(r['n_episodes'] for r in rows), 'cells': rows,
           'aggregates': aggregates, 'problems': problems, 'pending_methods': ['awq', 'hsa'],
           'caveats': ['原精度配置名 fp16，实际模型使用 BF16。',
                       'smoke 仅验证流程；只有正式档位四集合完整才报告四集合宏平均。',
                       '资源输入取自成功训练演示的开发观测，不代表失败状态分布。',
                       '动作步摊销推理耗时不是机器人端到端响应时间。',
                       '权重、输入与源码在启动/恢复时校验，外部文件运行期间必须保持不变。']}
    out = root / 'report'
    out.mkdir(exist_ok=True)
    (out / 'report.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    fields = [k for k in rows[0] if k != 'result_files']
    with (out / 'cells.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    title = '正式协议评测' if doc['formal'] else '开发验证 / 未通过正式准入'
    md = [f'# {title}', '', f'运行：`{plan["run_id"]}`；档位：`{doc["profile"]}`；回合：{doc["n_episodes"]}。', '',
          '| 模型 | 方法 | 所选集合平均成功率 % | 相对原精度 pp | 稳态 NVML GiB | 查询 ms |',
          '|---|---|---:|---:|---:|---:|']
    def fmt(x, multiplier=1):
        return '—' if x is None else f'{x * multiplier:.2f}'
    for r in aggregates:
        md.append(f'| {r["model"]} | {r["method"]} | {fmt(r["selected_suites_success_rate"],100)} | {fmt(r["delta_pp_vs_fp16"])} | {fmt(r["steady_nvml_gib"])} | {fmt(r["query_ms"])} |')
    md.extend(['', '![性能与资源对比](tradeoffs.png)', '', *doc['caveats'], '', 'AWQ/HSA 待接入。'])
    if problems:
        md.extend(['', '准入问题：', '', *[f'- {p}' for p in problems]])
    (out / 'report.md').write_text('\n'.join(md) + '\n')
    if not problems:
        plot(doc, out)
    return doc


def plot(doc, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    colors = {'openvla': '#3567a7', 'vla_adapter': '#c36b27', 'fastwam': '#21866b'}
    for r in doc['aggregates']:
        for ax, key in zip(axes, ('steady_nvml_gib', 'per_action_ms')):
            ax.scatter(r[key], r['selected_suites_success_rate'] * 100,
                       color=colors[r['model']], marker='o' if r['method'] == 'fp16' else 's',
                       label=f'{r["model"]} / {r["method"]}')
    for ax in axes:
        ax.set_ylabel('Selected-suite mean success (%)')
        ax.set_ylim(-3, 103)
        ax.grid(alpha=.2)
    axes[0].set_xlabel('Steady-state NVML peak (GiB)')
    axes[1].set_xlabel('Inference cost per action step (ms)')
    axes[1].legend(fontsize=7)
    fig.suptitle('SMOKE: development validation only' if doc['profile'] == 'smoke' else 'Managed evaluation')
    for ext in ('png', 'svg'):
        fig.savefig(out / f'tradeoffs.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    result = generate_report(Path(sys.argv[1]))
    print(f'report: {Path(sys.argv[1]) / "report/report.md"}; problems={result["problems"]}')
    raise SystemExit(0 if result['complete'] else 1)
