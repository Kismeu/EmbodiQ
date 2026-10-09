"""Export an offline, read-only UI showcase from verified completed runs.

Only allowlisted display fields are exported. Checkpoints, source paths, hashes,
raw logs and authentication material never enter the demonstration payload.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build(run_paths, out):
    from embodiq.web.server import Workbench
    from embodiq.pipeline import capabilities
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states, reports = [], {}
    for path in map(lambda x: Path(x).resolve(), run_paths):
        app = Workbench(ROOT / 'configs/stage_fp16_rtn.json', path.parent)
        status = app.status(path.name)
        doc = app.report(path.name)  # checks stored plan + all stage output hashes
        if status['phase'] != 'complete' or not doc['complete']:
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
    caps = capabilities()
    caps.update(suites=['libero_spatial','libero_object','libero_goal','libero_10'],csrf_token='read-only-demo',default_gpu=0)
    payload = {'capabilities':caps,'runs':states,'reports':reports,'read_only':True}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2).replace('<','\\u003c')
    (out / 'demo-data.js').write_text('window.EMBODIQ_DEMO = '+serialized+';\n')
    static = ROOT / 'src/embodiq/web/static'
    html = (static / 'index.html').read_text().replace('href="/style.css"','href="./style.css"')
    html = html.replace('<script defer src="/app.js"></script>', '<script defer src="./demo-data.js"></script><script defer src="./demo.js"></script><script defer src="./app.js"></script>')
    html = html.replace('<body>','<body><div class="demo-banner">只读演示 · 展示真实快速检查记录，配置预览不会启动模型。<a href="../README.md">安装与使用</a></div>')
    (out / 'index.html').write_text(html)
    shutil.copy2(static / 'app.js',out / 'app.js')
    css = (static / 'style.css').read_text()+'\n.demo-banner{background:#edf2fd;color:#284fa5;text-align:center;padding:10px 16px;font-size:12px;border-bottom:1px solid #d7e1fa}.demo-banner a{margin-left:15px;text-decoration:underline}.sidebar{top:40px}@media(max-width:680px){.sidebar{top:auto}}\n'
    (out / 'style.css').write_text(css)
    print(f'exported {len(states)} verified runs to {out}')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',nargs='+',type=Path)
    ap.add_argument('--out',type=Path,default=ROOT/'demo')
    args=ap.parse_args()
    build(args.runs,args.out)
