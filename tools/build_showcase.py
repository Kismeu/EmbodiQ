"""Export an offline, read-only UI showcase from verified completed runs.

Only allowlisted display fields are exported. Checkpoints, source paths, hashes,
raw logs and authentication material never enter the demonstration payload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build(run_paths, out):
    from embodiq.web.server import Workbench
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states, reports = [], {}
    for path in map(lambda x: Path(x).resolve(), run_paths):
        app = Workbench(ROOT / 'configs/stage_fp16_rtn.json', path.parent)
        status = app.status(path.name)
        doc = app.report(path.name)  # checks stored plan + all stage output hashes
        if status['profile'] != 'formal' or not doc['formal'] or status['phase'] != 'complete' or not doc['complete']:
            raise ValueError(f'not a completed verified run: {path.name}')
        # Workbench.status already uses a display allowlist. Remove any text that
        # could contain paths from previous failures before publishing.
        status['reasons'] = []
        for job in status['jobs']:
            job.pop('error', None)
        report = {key: doc[key] for key in ('schema','run_id','profile','formal','complete','n_episodes','aggregates','pending_methods','caveats')}
        report['cells'] = [{k:v for k,v in cell.items() if k != 'result_files'} for cell in doc['cells']]
        states.append(status)
        reports[path.name] = report
    write_showcase(states, reports, out)


def write_showcase(states, reports, out):
    from embodiq.pipeline import capabilities
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    caps = capabilities()
    caps.update(suites=['libero_spatial','libero_object','libero_goal','libero_10'],csrf_token='read-only-demo',default_gpu=0)
    payload = {'capabilities':caps,'runs':states,'reports':reports,'read_only':True}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2).replace('<','\\u003c')
    (out / 'demo-data.js').write_text('window.EMBODIQ_DEMO = '+serialized+';\n')
    static = ROOT / 'src/embodiq/web/static'
    html = (static / 'index.html').read_text().replace('href="/style.css"','href="./style.css"')
    html = html.replace('<script defer src="/app.js"></script>', '<script defer src="./demo-data.js"></script><script defer src="./demo.js"></script><script defer src="./app.js"></script><script defer src="./guide.js"></script>')
    html = html.replace('</head>', '<link rel="stylesheet" href="./guide.css"></head>')
    html = html.replace('<body>','<body><div class="demo-banner">只读演示 · 按三个步骤浏览已完成的评测，不会启动模型。<a href="../README.md">安装与使用</a></div>')
    (out / 'index.html').write_text(html)
    shutil.copy2(static / 'app.js',out / 'app.js')
    for name in ('demo.js', 'guide.js', 'guide.css'):
        source = ROOT / 'demo' / name
        if source.resolve() != (out / name).resolve():
            shutil.copy2(source, out / name)
    css = (static / 'style.css').read_text()+'\n.demo-banner{background:var(--accent-soft);color:var(--accent-ink);text-align:center;padding:9px 16px;font-size:12px;border-bottom:1px solid var(--line);height:40px;position:relative;z-index:6}.demo-banner a{margin-left:15px;text-decoration:underline}.sidebar{top:40px}.topbar{top:0}@media(max-width:680px){.sidebar{top:auto}.demo-banner{height:auto}}\n'
    (out / 'style.css').write_text(css)
    print(f'exported {len(states)} verified runs to {out}')


def build_matrix(path, out):
    """Export an admitted historical matrix; do not fabricate managed job logs."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get('schema') != 'embodiq/result_matrix/2' or doc.get('problems') or doc.get('source_overrides'):
        raise ValueError('expected an admitted matrix from the standard formal sources')
    models = ['openvla', 'vla_adapter', 'fastwam']
    methods = ['fp16', 'rtn']
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    expected = {(m, q, s) for m in models for q in methods for s in suites}
    cells = [c for c in doc['cells'] if c['method'] in methods]
    if len(cells) != 24 or {(c['model'], c['method'], c['suite']) for c in cells} != expected:
        raise ValueError('the formal showcase requires all 24 model/method/suite cells')
    benches, hashes = {}, {}
    for c in cells:
        if (not c['complete'] or c['n_tasks'] != 10 or c['n_episodes'] != 500
                or any(c[k] for k in ('n_error', 'n_invalid', 'n_policy_error'))):
            raise ValueError('incomplete formal task cell')
        ref = c['resource_bench']
        if ref['status'] != 'complete':
            raise ValueError('formal showcase requires measured resource data')
        source = Path(ref['source'])
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != ref['sha256']:
            raise ValueError('historical resource evidence changed')
        key = (c['model'], c['method'])
        benches[key] = json.loads(source.read_text())
        hashes['/'.join(key)] = actual
    rows = []
    for c in cells:
        bench = benches[c['model'], c['method']]['suites'][c['suite']]
        if bench['n_inputs'] != 50 or bench['n_queries_measured'] != 150 or bench['passes'] != 3:
            raise ValueError('incomplete resource observations')
        memory, latency = bench['memory'], bench['latency']
        rows.append(dict(model=c['model'], method=c['method'], suite=c['suite'],
                         n_episodes=c['n_episodes'], n_success=c['n_success'], success_rate=c['success_rate'],
                         query_ms=latency['query_s']['mean'] * 1000,
                         per_action_ms=latency['per_env_step_s'] * 1000,
                         steady_nvml_gib=memory['nvml_peak_bytes'] / 2**30,
                         steady_alloc_gib=memory['peak_alloc_bytes'] / 2**30,
                         load_alloc_gib=memory['load_peak_alloc_bytes'] / 2**30))
    aggregates = []
    for model in models:
        baseline = sum(c['success_rate'] for c in rows if c['model'] == model and c['method'] == 'fp16') / 4
        for method in methods:
            selected = [c for c in rows if c['model'] == model and c['method'] == method]
            average = sum(c['success_rate'] for c in selected) / 4
            aggregates.append(dict(model=model, method=method, selected_suites_success_rate=average,
                four_suite_macro_success_rate=average, delta_pp_vs_fp16=100 * (average - baseline),
                query_ms=sum(c['query_ms'] for c in selected) / 4,
                per_action_ms=sum(c['per_action_ms'] for c in selected) / 4,
                steady_nvml_gib=max(c['steady_nvml_gib'] for c in selected)))
    run_id = 'formal_history_v3'
    caveats = ['原始精度采用 BF16；RTN 使用 4 位权重，每组 128 个权重。',
        '展示已完成评测的汇总结果，不是实时运行。',
        '本组显存数据的采样间隔为 100 毫秒，可能遗漏短时峰值。',
        '本组资源记录包含权重路径与量化配置，未保存完整权重文件指纹。',
        '查询耗时宜在同一模型内比较；不同模型的动作块长度与重规划频率不同。',
        '资源观测来自训练演示，仅用于测量耗时和显存；成功率来自独立的 12,000 个正式任务回合。']
    report = dict(schema='embodiq/showcase-report/1', run_id=run_id, profile='formal', formal=True,
        complete=True, n_episodes=12000, cells=rows, aggregates=aggregates, pending_methods=['awq', 'hsa'],
        display_note='评测结果汇总 · 24 个组合 · 12,000 回合', caveats=caveats,
        provenance=dict(source_schema=doc['schema'], matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        generated=doc['generated'], resource_sha256=hashes))
    status = dict(run_id=run_id, phase='complete', saved_phase='complete', active=False, source='archive',
        created='LIBERO 四类任务', profile='formal', models=models, methods=methods, suites=suites,
        n_episodes=12000, reasons=[],
        stop_requested=False, completed=0, total=0, jobs=[])
    write_showcase([status], {run_id: report}, out)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='*',type=Path)
    ap.add_argument('--matrix',type=Path,help='admitted formal history matrix from tools/report_matrix.py')
    ap.add_argument('--out',type=Path,default=ROOT/'demo')
    args=ap.parse_args()
    if args.matrix and args.runs:
        ap.error('choose either completed runs or --matrix')
    if args.matrix:
        build_matrix(args.matrix,args.out)
    elif args.runs:
        build(args.runs,args.out)
    else:
        ap.error('provide completed formal runs or --matrix')
