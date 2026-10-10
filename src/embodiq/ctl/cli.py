"""User-facing command line entry point for the unified evaluation harness.

The project script is intentionally a thin dispatcher.  Model-specific
evaluation remains in :mod:`embodiq.entry.evaluate`, while environment checks
remain in :mod:`embodiq.ctl.doctor`; this keeps the CLI independent from any
particular quantizer implementation.
"""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodiq",
        description="统一 VLA/WAM 适配、量化评测和环境检查入口",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    web = sub.add_parser('web', help='启动本地 Web/API 评测工作台')
    web.add_argument('--config', default='configs/stage_fp16_rtn.json')
    web.add_argument('--runs-root', default='results/platform')
    web.add_argument('--port', type=int, default=8765)

    sub.add_parser('capabilities', help='列出自动评测已支持和待接入的方法')
    launch = sub.add_parser('launch', help='一键执行：预检、闭环评测、资源专测、报告')
    launch.add_argument('path', help='实验配置 JSON/YAML')
    launch.add_argument('--models', nargs='+', choices=('openvla', 'vla_adapter', 'fastwam'))
    launch.add_argument('--methods', nargs='+', choices=('fp16', 'rtn', 'awq', 'hsa'))
    launch.add_argument('--suites', nargs='+')
    launch.add_argument('--profile', choices=('formal',), default=None)
    launch.add_argument('--gpu', type=int, default=0)
    launch.add_argument('--wait-seconds', type=float, default=0, help='GPU 忙碌时最多等待秒数；超时保留可恢复状态')
    launch.add_argument('--out-root', default=None)
    launch.add_argument('--dry-run', action='store_true', help='只显示计划，不创建运行或占用 GPU')
    resume_run = sub.add_parser('resume-run', help='恢复整个一键评测流程，跳过已校验完成阶段')
    resume_run.add_argument('path')
    resume_run.add_argument('--gpu', type=int, default=0)
    resume_run.add_argument('--wait-seconds', type=float, default=0)

    doctor = sub.add_parser("doctor", help="运行模型环境预检")
    doctor.add_argument("--family", required=True,
                        choices=("openvla", "adapter", "fastwam"))
    doctor.add_argument("--json", type=str, default=None)

    # Disable the dispatcher-level help flag for this pass-through command so
    # ``embodiq evaluate --help`` reaches the real evaluator and shows its full
    # model/quantization options.
    config = sub.add_parser("validate-config", help="校验统一实验配置")
    config.add_argument("path")

    plan = sub.add_parser("plan", help="展开配置为可执行任务清单")
    plan.add_argument("path")
    plan.add_argument("--out", default=None)

    run = sub.add_parser("run", help="创建带状态的实验运行目录")
    run.add_argument("path")
    run.add_argument("--out-root", default=None)

    status = sub.add_parser("status", help="查看实验运行状态")
    status.add_argument("path")

    preflight = sub.add_parser("preflight", help="检查运行计划的模型、权重和量化工件")
    preflight.add_argument("path")

    execute = sub.add_parser("execute", help="执行运行中的一个任务")
    execute.add_argument("path")
    execute.add_argument("--job-id", required=True)
    execute.add_argument("--gpu", type=int, default=0)

    resume = sub.add_parser("resume", help="恢复并执行运行中的一个任务")
    resume.add_argument("path")
    resume.add_argument("--job-id", required=True)
    resume.add_argument("--gpu", type=int, default=0)

    stop = sub.add_parser("stop", help="请求停止一个托管运行")
    stop.add_argument("path")

    report = sub.add_parser("report", help="从逐任务结果生成统一矩阵报告")
    report.add_argument("--out", default=None)
    report.add_argument('--run', default=None, help='只汇总指定自动评测运行')

    evaluate = sub.add_parser(
        "evaluate",
        help="运行统一 LIBERO 评测；其余参数原样传给评测入口",
        add_help=False,
    )
    evaluate.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # The evaluator owns a large, model-specific argument surface.  Dispatch
    # it before the small top-level parser so its ``--help`` and all option
    # validation are preserved verbatim.
    import json
    if raw and raw[0] == "evaluate":
        from embodiq.entry.evaluate import main as evaluate_main

        return int(evaluate_main(raw[1:]))

    args = _parser().parse_args(raw)
    if args.command == 'web':
        from embodiq.web.server import serve
        serve(args.config, args.runs_root, args.port)
        return 0
    if args.command in {'capabilities', 'launch', 'resume-run'}:
        from embodiq.pipeline import (capabilities, prepare_config, create_pipeline, execute_pipeline)
        if args.command == 'capabilities':
            print(json.dumps(capabilities(), ensure_ascii=False, indent=2))
            return 0
        if args.gpu < 0 or args.wait_seconds < 0:
            raise ValueError('gpu and wait-seconds must be nonnegative')
        if args.command == 'resume-run':
            return execute_pipeline(args.path, gpu=args.gpu, resume=True, wait_seconds=args.wait_seconds)
        config = prepare_config(args.path, models=args.models, methods=args.methods,
                                suites=args.suites, profile=args.profile)
        if args.dry_run:
            from embodiq.config import build_plan
            jobs = build_plan(config)
            print(json.dumps({'profile': config['profile'], 'rollout_jobs': len(jobs),
                              'resource_jobs': len(config['models']) * len(config['enabled_methods']),
                              'report_jobs': 1, 'jobs': jobs}, ensure_ascii=False, indent=2))
            return 0
        path, state = create_pipeline(config, args.out_root)
        print(f'run: {path}', flush=True)
        return execute_pipeline(path, gpu=args.gpu, wait_seconds=args.wait_seconds)
    if args.command == "validate-config":
        from embodiq.config import validate_file
        config = validate_file(args.path)
        print(f"valid: {args.path} ({config.get('name', 'unnamed')})")
        return 0

    if args.command == "run":
        from embodiq.runner import create_run
        run_dir, state = create_run(args.path, args.out_root)
        print(f"created: {run_dir}")
        print(f"planned jobs: {len(state['jobs'])}")
        return 0

    if args.command in {"execute", "resume"}:
        from embodiq.runner import execute_job, load_run
        if load_run(args.path).get('schema') == 'embodiq/pipeline/1':
            raise ValueError('use resume-run to resume an automatic pipeline')
        return int(execute_job(args.path, args.job_id, gpu=args.gpu, resume=(args.command == "resume")))

    if args.command == "stop":
        from embodiq.pipeline import SCHEMA, read, request_stop
        from pathlib import Path
        if read(Path(args.path) / 'run.json').get('schema') == SCHEMA:
            print(json.dumps(request_stop(args.path), ensure_ascii=False))
            return 0
        from embodiq.runner import stop_run
        state = stop_run(args.path)
        print(json.dumps({"run_id": state.get("run_id"), "phase": state.get("phase")}, ensure_ascii=False))
        return 0

    if args.command == "report":
        import subprocess
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        if args.run:
            if args.out:
                raise ValueError('--run reports are written inside that run; do not use --out')
            from embodiq.pipeline import read, lock, digest
            run = Path(args.run).resolve()
            with lock(run / '.controller.lock'):
                state = read(run / 'run.json')
                plan = read(run / 'plan.json')
                if state['plan_sha256'] != digest(run / 'plan.json'):
                    raise ValueError('plan changed')
                if state['phase'] == 'complete':
                    for stage in state['jobs']:
                        if not stage.get('outputs') or any(not Path(p).is_file() or digest(p) != h
                                                         for p, h in stage['outputs'].items()):
                            raise ValueError('saved outputs changed; report is not verified')
                    print(str(run / 'report/report.md'))
                    return 0
                from embodiq.runner import command_for_job
                _, env = command_for_job(plan['jobs'][0], output=run, gpu=0)
                return subprocess.run([plan['jobs'][0]['python'], '-m', 'embodiq.pipeline_report', str(run)],
                                      env=env, cwd=root).returncode
        cmd = [sys.executable, str(root / "tools" / "report_matrix.py")]
        if args.out:
            cmd += ["--out", args.out]
        return int(subprocess.run(cmd, cwd=root).returncode)

    if args.command == "status":
        from embodiq.runner import load_run, summarize_run
        state = load_run(args.path)
        if state.get('schema') == 'embodiq/pipeline/1':
            from pathlib import Path
            print(json.dumps({'run_id': state['run_id'], 'phase': state['phase'],
                              'stop_requested': (Path(args.path) / 'STOP_REQUESTED').exists(),
                              'reasons': state.get('reasons', []),
                              'summary': summarize_run(state),
                              'jobs': [{k: j[k] for k in ('job_id', 'status', 'attempts', 'log', 'error') if k in j}
                                       for j in state['jobs']]}, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps({"run_id": state.get("run_id"), "phase": state.get("phase"), "summary": summarize_run(state)}, ensure_ascii=False))
        return 0

    if args.command == "preflight":
        from embodiq.runner import load_run, preflight_job
        state = load_run(args.path)
        if state.get('schema') == 'embodiq/pipeline/1':
            from pathlib import Path
            from embodiq.pipeline import read, pipeline_preflight
            reasons = pipeline_preflight(read(Path(args.path) / 'plan.json'))
            print(json.dumps({'ready': not reasons, 'reasons': reasons}, ensure_ascii=False, indent=2))
            return 2 if reasons else 0
        rows = []
        for job in state.get("jobs", []):
            reasons = preflight_job(job)
            rows.append({"job_id": job["job_id"], "model": job["model"],
                         "suite": job["suite"], "quant": job["quant"],
                         "ready": not reasons, "reasons": reasons})
        payload = {"run_id": state.get("run_id"), "ready": sum(r["ready"] for r in rows),
                   "blocked": sum(not r["ready"] for r in rows), "jobs": rows}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["blocked"] == 0 else 2

    if args.command == "plan":
        import json
        from embodiq.config import build_plan, validate_file
        config = validate_file(args.path)
        jobs = build_plan(config)
        payload = {"schema": "embodiq/experiment-plan/1", "config": args.path, "jobs": jobs}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            from pathlib import Path
            Path(args.out).write_text(text + "\n")
        else:
            print(text)
        print(f"planned jobs: {len(jobs)}", file=sys.stderr)
        return 0

    if args.command == "doctor":
        from embodiq.ctl.doctor import main as doctor_main

        forwarded = ["--family", args.family]
        if args.json is not None:
            forwarded.extend(["--json", args.json])
        return int(doctor_main(forwarded))

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
