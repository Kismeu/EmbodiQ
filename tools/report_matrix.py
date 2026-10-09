"""The single source of truth for the 1.4 result matrix.

Scans every saved run, validates it, and emits ONE machine-readable file plus a
Markdown rendering of the same numbers:

    results/matrix/matrix.json
    results/matrix/matrix.md

Everything downstream -- the report tables, the tradeoff figures, the
visualization page -- reads matrix.json and recomputes nothing. A second
computation path is a second set of numbers, and the protocol forbids that.

Design rules that this file enforces, not merely documents:

* A suite cell is `complete` only with task_id 0..9, 500 episodes, and zero
  error/invalid/policy-error episodes. Anything else keeps its real counts and
  is reported as incomplete -- never padded, never dropped, never counted as 0.
* A macro average exists only when all four suites of that configuration are
  complete (protocol section 8: "missing results must not be disguised as a
  complete average").
* Quantization deltas are paired McNemar on identical `initial_state_sha256`,
  via core.protocol.mcnemar_delta. Independent intervals are not used here.
* Peak VRAM is reported as three separate numbers because they mean different
  things, and `nvml_peak_bytes` is currently contaminated by the BF16 load
  phase for quantized runs (see memory_caveat in the output).
* Latency is reported as two numbers: `query_s` is comparable only within a
  model, `per_env_step_s` is the only cross-model-comparable figure.

Usage:
    python tools/report_matrix.py [--out results/matrix]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodiq.core.protocol import mcnemar_delta  # noqa: E402
from embodiq.core.admission import (run_problems, condition_identity, artifact_problems,
                                   finite_positive, policy_mode)  # noqa: E402

MODELS = ("openvla", "vla_adapter", "fastwam")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
MODEL_LABELS = {
    "openvla": "OpenVLA",
    "vla_adapter": "VLA-Adapter Pro",
    "fastwam": "FastWAM",
}
# Where each method's runs live. Add AWQ/HSA directories here when 1.2/1.3
# deliver their artifacts; nothing else in this file needs to change.
METHOD_SOURCES: dict[str, tuple[str, ...]] = {
    "fp16": ("results/baseline_unified_v3_20260919/full",),
    "rtn": ("results/stage5_rtn_full",),
    "awq": ("results/stage6_awq_full",),
    "hsa": ("results/stage7_hsa_full",),
}
METHOD_LABELS = {"fp16": "原精度", "rtn": "RTN", "awq": "AWQ", "hsa": "HSA"}
BASELINE_METHOD = "fp16"
EXPECTED_TASKS = 10
EXPECTED_EPISODES_PER_TASK = 50

# Rows written before run_result/3 measured their process peak while the
# caching allocator was still holding the bf16 load-phase pool, so for those
# rows nvml_peak_bytes and peak_reserved_bytes describe the ORIGINAL precision,
# not the quantized model. schema/3 releases the pool after quantization and
# reports the load phase separately, so both numbers mean what they say.
MEMORY_CAVEAT_LEGACY = (
    "schema/2 行的 nvml_peak_bytes 与 peak_reserved_bytes 被 BF16 加载阶段污染："
    "量化在同进程权重加载之后进行，caching allocator 的 reserved 段保留了加载期峰值。"
    "这些行的稳态部署占用请读 peak_alloc_bytes；nvml_peak_bytes 只能解释为"
    "“加载+推理的进程峰值”，不可用于量化前后的峰值显存对比。"
)
MEMORY_CAVEAT_CURRENT = (
    "schema/3 行在量化后释放 allocator 缓存并重置峰值，"
    "load_peak_* 为加载期峰值、nvml_peak_bytes 为稳态进程峰值，两者分列。"
)
SCHEMA_WITH_LOAD_SPLIT = "embodiq/run_result/3"
LATENCY_CAVEAT = (
    "query_s 只能在同一模型内比较：三模型查询密度为 1 : 1/8 : 1/10"
    "（OpenVLA 每步重规划、VLA-Adapter 每 8 步、FastWAM 每 10 步）。"
    "跨模型只有 per_env_step_s 可比。"
)


def iter_result_files(source: Path):
    if not source.is_dir():
        return
    yield from sorted(source.glob("**/*_results.json"))


def episode_success(ep: dict) -> bool:
    return ep.get("status") == "success"


def load_cells(sources_by_method=None) -> tuple[dict, list[str]]:
    """-> {(model, method, suite): cell}, plus a list of integrity complaints."""
    cells: dict[tuple[str, str, str], dict] = {}
    problems: list[str] = []
    conditions = {}
    rejected_conditions = set()
    checkpoint_conditions = {}
    rejected_cells = set()
    for method, sources in (METHOD_SOURCES if sources_by_method is None else sources_by_method).items():
        tasks: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for source in sources:
            for path in iter_result_files(ROOT / source):
                try:
                    d = json.loads(path.read_text())
                    model = d["policy"]["family"]
                    suite = d["task_suite"]
                    task_id = d["task_id"]
                except (ValueError, KeyError, TypeError) as exc:
                    problems.append(f"{path}: unreadable result: {exc}")
                    continue
                declared = d.get("quant", {}).get("method", "fp16")
                if declared != method:
                    problems.append(
                        f"{path}: quant.method={declared!r} "
                        f"but found under the {method!r} source"
                    )
                    continue
                if model not in MODELS or suite not in SUITES:
                    problems.append(f"{path}: unknown model/suite")
                    continue
                issues = run_problems(d, model=model, method=method, suite=suite)
                if task_id not in range(EXPECTED_TASKS): issues.append('task_id out of range')
                if issues:
                    problems.extend(f"{path}: {issue}" for issue in issues)
                    rejected_cells.add((model, method, suite))
                    continue
                identity = condition_identity(d['policy'], d['quant'], d['protocol'])
                key = (model, method)
                if key in conditions and conditions[key] != identity:
                    problems.append(f"{model}/{method}: mixed implementation/scope/policy/protocol")
                    rejected_conditions.add(key)
                conditions[key] = identity
                art = d['quant'].get('artifact') or {}
                checkpoint_identity = (str(Path(d['policy']['checkpoint']).resolve()),
                                       art.get('artifact_sha256'), art.get('source_checkpoint_sha256'))
                ck = (model, method, suite)
                if ck in checkpoint_conditions and checkpoint_conditions[ck] != checkpoint_identity:
                    problems.append(f"{model}/{method}/{suite}: mixed checkpoint or artifact")
                    rejected_cells.add(ck)
                checkpoint_conditions[ck] = checkpoint_identity
                if task_id in tasks[(model, suite)]:
                    rejected_cells.add((model, method, suite))
                    problems.append(
                        f"{path}: duplicate task_id {task_id} "
                        f"for {model}/{method}/{suite}"
                    )
                    continue
                tasks[(model, suite)][task_id] = d
        for (model, suite), by_task in tasks.items():
            if (model, method) in rejected_conditions or (model, method, suite) in rejected_cells:
                continue
            cells[(model, method, suite)] = build_cell(
                model, method, suite, by_task, problems
            )
    return cells, problems


def build_cell(model, method, suite, by_task: dict[int, dict], problems) -> dict:
    problems_before = len(problems)
    episodes: dict[str, bool] = {}
    n_ep = n_success = n_error = n_invalid = n_policy_error = n_timeout = 0
    contended = False
    quant = None
    policy = None
    versions = None
    schemas = set()
    lat_query, lat_step, lat_p95 = [], [], []
    mem = defaultdict(int)
    for task_id in sorted(by_task):
        d = by_task[task_id]
        m = d["metrics"]
        n_ep += m["n_episodes"]
        n_success += m["n_success"]
        n_error += m["n_error"]
        n_invalid += m["n_invalid"]
        n_policy_error += m["n_policy_error"]
        n_timeout += m.get("n_timeout", 0)
        if len(d["episodes"]) != m["n_episodes"]:
            problems.append(
                f"{model}/{method}/{suite}/task{task_id}: episode list "
                f"({len(d['episodes'])}) != n_episodes ({m['n_episodes']})"
            )
        if m["n_episodes"] != EXPECTED_EPISODES_PER_TASK:
            problems.append(
                f"{model}/{method}/{suite}/task{task_id}: "
                f"{m['n_episodes']} episodes, expected {EXPECTED_EPISODES_PER_TASK}"
            )
        for ep in d["episodes"]:
            state = (ep.get("aux") or {}).get("initial_state_sha256")
            if state is None:
                problems.append(
                    f"{model}/{method}/{suite}/task{task_id}/ep{ep['episode_idx']}: "
                    f"no initial_state_sha256; cannot be paired"
                )
                continue
            key = f"{task_id}:{ep.get('seed')}:{state}"
            if key in episodes:
                problems.append(f"{model}/{method}/{suite}/task{task_id}: duplicate episode state/seed")
            if ep.get('seed') != 42:
                problems.append(f"{model}/{method}/{suite}/task{task_id}: episode seed mismatch")
            episodes[key] = episode_success(ep)
        ids = [ep.get('episode_idx') for ep in d['episodes']]
        if sorted(ids) != list(range(len(ids))):
            problems.append(f"{model}/{method}/{suite}/task{task_id}: duplicate/missing episode index")
        statuses = [ep.get('status') for ep in d['episodes']]
        if any(s not in {'success','timeout','policy_error','env_error'} for s in statuses):
            problems.append(f"{model}/{method}/{suite}/task{task_id}: unknown episode status")
        if m['n_success'] != statuses.count('success') or m.get('n_timeout',0) != statuses.count('timeout'):
            problems.append(f"{model}/{method}/{suite}/task{task_id}: metric/episode counts disagree")
        if (m['n_error'] != statuses.count('policy_error')+statuses.count('env_error') or
            m['n_policy_error'] != statuses.count('policy_error') or m['n_invalid'] != statuses.count('env_error')):
            problems.append(f"{model}/{method}/{suite}/task{task_id}: error counts disagree")
        schemas.add(d.get("schema_version"))
        contended = contended or bool(d["run"].get("contended"))
        quant = quant or d.get("quant")
        policy = policy or d.get("policy")
        versions = versions or d["run"].get("versions")
        lat = m.get("latency") or {}
        if lat.get("query_s", {}).get("mean") is not None:
            lat_query.append(lat["query_s"]["mean"])
            lat_p95.append(lat["query_s"].get("p95"))
        if lat.get("per_env_step_s") is not None:
            lat_step.append(lat["per_env_step_s"])
        for key, value in (m.get("memory") or {}).items():
            if isinstance(value, (int, float)) and key.endswith("bytes"):
                mem[key] = max(mem[key], value)

    task_ids = sorted(by_task)
    complete = (
        task_ids == list(range(EXPECTED_TASKS))
        and n_ep == EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK
        and n_error == 0
        and n_invalid == 0
        and n_policy_error == 0
        and len(problems) == problems_before
        and len(episodes) == n_ep
    )
    planned = EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK
    return {
        "model": model,
        "method": method,
        "suite": suite,
        "complete": complete,
        "missing_task_ids": [t for t in range(EXPECTED_TASKS) if t not in by_task],
        "n_tasks": len(task_ids),
        "n_episodes": n_ep,
        "n_planned": planned,
        "n_success": n_success,
        "n_timeout": n_timeout,
        "n_error": n_error,
        "n_invalid": n_invalid,
        "n_policy_error": n_policy_error,
        # Reported for every cell, complete or not. A rate over a partial cell
        # is a progress indicator, never a result: `complete` is the gate.
        "success_rate": (n_success / n_ep) if n_ep else None,
        "contended": contended,
        "latency": {
            "query_s_mean": mean(lat_query),
            "query_s_p95": mean([x for x in lat_p95 if x is not None]),
            "per_env_step_s": mean(lat_step),
            "note": LATENCY_CAVEAT,
        },
        "schema_versions": sorted(s for s in schemas if s),
        "memory": {
            **dict(mem),
            # Which peak numbers this cell's own schema lets you trust.
            "load_phase_separated": SCHEMA_WITH_LOAD_SPLIT in schemas,
            "note": (MEMORY_CAVEAT_CURRENT if schemas == {SCHEMA_WITH_LOAD_SPLIT}
                     else MEMORY_CAVEAT_LEGACY),
        },
        "quant": prune_quant(quant),
        "policy": policy,
        "versions": versions,
        "_episodes": episodes,  # stripped before writing; used for pairing
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def validate_episode_manifest(cells, problems, path=None):
    path=Path(path or ROOT/'results/baseline_unified_v3_20260919/episode_manifest.json')
    try:
        manifest=json.loads(path.read_text())
        if manifest['protocol']!='unified.v3':raise ValueError('protocol mismatch')
        expected=defaultdict(set)
        for task in manifest['tasks']:
            for ep in task['episodes']:
                expected[task['suite']].add(f"{task['task_id']}:{ep['seed']}:{ep['state_sha256']}")
        if set(expected)!=set(SUITES) or any(len(v)!=500 for v in expected.values()):
            raise ValueError('manifest does not contain 500 distinct episodes per suite')
    except (OSError,KeyError,TypeError,ValueError) as exc:
        problems.append(f'{path}: cannot validate episode identity: {exc}')
        cells.clear();return
    for key,cell in list(cells.items()):
        if not set(cell['_episodes'])<=expected[cell['suite']]:
            problems.append(f"{'/'.join(key)}: states/seeds differ from frozen episode manifest")
            del cells[key]


def prune_quant(quant: dict | None) -> dict | None:
    """Keep the accounting fields; drop the per-layer list (thousands of rows)."""
    if quant is None:
        return None
    out = {k: v for k, v in quant.items() if k not in ("layers", "by_skip_reason")}
    out["by_skip_reason"] = {
        k: v["n"] for k, v in (quant.get("by_skip_reason") or {}).items()
    }
    return out


def paired_delta(baseline: dict | None, variant: dict | None) -> dict | None:
    """McNemar on episodes present in BOTH arms with the same initial state."""
    if not baseline or not variant:
        return None
    shared = sorted(set(baseline["_episodes"]) & set(variant["_episodes"]))
    if not shared:
        return None
    pairs = [(baseline["_episodes"][k], variant["_episodes"][k]) for k in shared]
    out = mcnemar_delta(pairs)
    out["n_baseline_episodes"] = len(baseline["_episodes"])
    out["n_variant_episodes"] = len(variant["_episodes"])
    out["unpaired"] = len(variant["_episodes"]) - len(shared)
    return out


def build_configs(cells: dict) -> list[dict]:
    """One row per (model, method): the 12 rows of the headline table."""
    rows = []
    for model in MODELS:
        for method in METHOD_SOURCES:
            per_suite = {}
            for suite in SUITES:
                cell = cells.get((model, method, suite))
                base = cells.get((model, BASELINE_METHOD, suite))
                per_suite[suite] = {
                    "present": cell is not None,
                    "complete": bool(cell and cell["complete"]),
                    "n_episodes": cell["n_episodes"] if cell else 0,
                    "n_planned": EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK,
                    "success_rate": cell["success_rate"] if cell else None,
                    "missing_task_ids": cell["missing_task_ids"] if cell else list(range(EXPECTED_TASKS)),
                    "paired_vs_fp16": (
                        None if method == BASELINE_METHOD else paired_delta(base, cell)
                    ),
                }
            all_complete = all(s["complete"] for s in per_suite.values())
            macro = (
                sum(per_suite[s]["success_rate"] for s in SUITES) / len(SUITES)
                if all_complete
                else None
            )
            present = [cells[(model, method, s)] for s in SUITES if (model, method, s) in cells]
            rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "suites": per_suite,
                "n_complete_suites": sum(s["complete"] for s in per_suite.values()),
                "n_episodes": sum(s["n_episodes"] for s in per_suite.values()),
                # Protocol section 8: no average unless all four suites are complete.
                "macro_success_rate": macro,
                "macro_blocked_reason": None if all_complete else "四集合未全部完整",
                "latency": {
                    "query_s_mean": mean([c["latency"]["query_s_mean"] for c in present]),
                    "per_env_step_s": mean([c["latency"]["per_env_step_s"] for c in present]),
                },
                "memory": {
                    key: max([c["memory"].get(key) or 0 for c in present], default=None)
                    for key in ("weights_bytes", "peak_alloc_bytes",
                                "peak_reserved_bytes", "nvml_peak_bytes",
                                "load_peak_alloc_bytes", "load_peak_reserved_bytes",
                                "load_nvml_peak_bytes", "cache_released_bytes",
                                "residual_bytes", "residual_index_bytes",
                                "input_scale_bytes")
                } if present else {},
                "quant": present[0]["quant"] if present else None,
                "contended": any(c["contended"] for c in present),
            })
    return rows


def attach_resource_benches(configs, cells, problems, *, root=None, expected_suites=None):
    """Use verified fixed-input benchmarks only; never fall back to rollout peaks."""
    root = Path(root or ROOT / 'results/resource_bench')
    expected_suites = tuple(SUITES if expected_suites is None else expected_suites)
    manifest_path = root / 'inputs_v1.json'
    expected_hash = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        actual = hashlib.sha256(json.dumps(manifest['tasks'],sort_keys=True).encode()).hexdigest()
        if actual == manifest.get('manifest_sha256'):
            expected_hash = actual
        else: problems.append(f'{manifest_path}: input manifest hash mismatch')
    for row in configs:
        model,method=row['model'],row['method']
        row['rollout_memory']=row['memory']; row['rollout_latency']=row['latency']
        row['rollout_contended']=row['contended']
        row['memory']={};row['latency']={'query_s_mean':None,'per_env_step_s':None}
        row['contended']=None
        path=root/f'{model}_{method}.json'
        row['resource_bench']={'status':'missing','source':str(path)}
        if not path.is_file():continue
        try:
            d=json.loads(path.read_text()); issues=[]
            if d.get('schema')!='embodiq/resource_bench/1' or d.get('phase')!='complete':
                issues.append('incomplete or unknown resource schema')
            if d.get('contended') is not False:issues.append('GPU exclusivity not confirmed')
            if d.get('model')!=model or d.get('quant')!=method:issues.append('resource model/method mismatch')
            if not expected_hash or d.get('inputs_manifest_sha256')!=expected_hash:issues.append('resource input manifest mismatch')
            if d.get('warmup_queries')!=20 or d.get('measured_passes')!=3:issues.append('resource warmup/pass mismatch')
            suites=d.get('suites',{})
            if set(suites)!=set(expected_suites):issues.append('resource suites incomplete')
            for suite,b in suites.items():
                cell=cells.get((model,method,suite))
                if cell is None or not cell['complete']:
                    issues.append(f'{suite}: no complete matching closed-loop cell');continue
                p=cell['policy'];q=cell['quant'];bp=b['policy'];bq=b['quant']
                if Path(b['checkpoint']).resolve()!=Path(p['checkpoint']).resolve():issues.append(f'{suite}: checkpoint mismatch')
                if b.get('n_inputs')!=50 or b.get('n_queries_measured')!=150 or b.get('passes')!=3:
                    issues.append(f'{suite}: resource measurement count mismatch')
                if b.get('chunk_size')!=p['chunk_size'] or b.get('replan_steps')!=p['default_replan_steps']:
                    issues.append(f'{suite}: action schedule mismatch')
                if bp.get('torch_dtype')!=p.get('torch_dtype'):issues.append(f'{suite}: dtype mismatch')
                if model=='fastwam':
                    if policy_mode(bp)!='first_frame' or d.get('fastwam_mode')!='first_frame':issues.append(f'{suite}: temporal mode mismatch')
                    for k in ('sigma_shift','num_inference_steps','compile_action_infer','input_hw','dataset_stats_sha256'):
                        if bp.get('extra',{}).get(k)!=p.get('extra',{}).get(k):issues.append(f'{suite}: runtime {k} mismatch')
                elif bp.get('model_id')!=p.get('model_id'):issues.append(f'{suite}: model identity mismatch')
                if bq.get('method')!=method or set(bq.get('scope',[]))!=set(q.get('scope',[])):
                    issues.append(f'{suite}: quantization scope/method mismatch')
                if method!='fp16' and bq.get('spec_hash')!=q.get('spec_hash'):issues.append(f'{suite}: quantization spec mismatch')
                if method in ('awq','hsa'):
                    art=bq.get('artifact') or {}
                    issues.extend(artifact_problems(art,method=method,model=model,suite=suite))
                    if art.get('artifact_sha256')!=(q.get('artifact') or {}).get('artifact_sha256'):
                        issues.append(f'{suite}: resource artifact mismatch')
                for k in ('peak_alloc_bytes','nvml_peak_bytes','load_peak_alloc_bytes','weights_bytes'):
                    if not finite_positive(b['memory'].get(k)):issues.append(f'{suite}: missing/invalid {k}')
                if not finite_positive(b['latency']['query_s'].get('mean')) or not finite_positive(b['latency'].get('per_env_step_s')):
                    issues.append(f'{suite}: invalid latency')
            if issues:
                row['resource_bench'].update(status='rejected',reasons=issues)
                problems.extend(f'{path}: {e}' for e in issues);continue
            provenance=dict(status='complete',source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            inputs_manifest_sha256=expected_hash,n_queries=150*len(expected_suites),gpu=d.get('gpu'),
                            started=d.get('started'),finished=d.get('finished'),input_caveat=d.get('inputs_caveat'),
                            checkpoint_binding='resolved path and quantization spec; legacy bench lacks checkpoint file hash')
            row['resource_bench']=provenance;row['contended']=False
            row['memory']={k:max(b['memory'].get(k,0) for b in suites.values())
                           for k in ('weights_bytes','peak_alloc_bytes','peak_reserved_bytes','nvml_peak_bytes',
                                     'load_peak_alloc_bytes','load_peak_reserved_bytes','load_nvml_peak_bytes',
                                     'cache_released_bytes','residual_bytes','residual_index_bytes','input_scale_bytes')}
            row['memory']['load_phase_separated']=True
            row['latency']={'query_s_mean':mean([b['latency']['query_s']['mean'] for b in suites.values()]),
                            'per_env_step_s':mean([b['latency']['per_env_step_s'] for b in suites.values()])}
            for suite,b in suites.items():
                cells[(model,method,suite)]['resource_bench']={**provenance,'memory':b['memory'],'latency':b['latency']}
        except (KeyError,TypeError,ValueError) as exc:
            row['resource_bench'].update(status='rejected',reasons=[str(exc)])
            problems.append(f'{path}: malformed resource result: {exc}')


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}%" if x is not None else "—"


def gib(x):
    return f"{x / 2 ** 30:.2f}" if x else "—"


def render_markdown(doc: dict) -> str:
    L = [
        "# 1.4 结果矩阵",
        "",
        doc["generated"],
        "",
        f"完整单元 **{doc['summary']['complete_cells']}/{doc['summary']['total_cells']}**，"
        f"已落盘回合 {doc['summary']['saved_episodes']}/{doc['summary']['planned_episodes']}。",
        "",
        ("> **非标准来源**：" + "、".join(f"{k} ← {v}" for k, v in doc["source_overrides"].items())
         + "。本表为开发校验用，不得作为任务结果引用。\n"
         if doc["source_overrides"] else ""),
        "本文件由 `tools/report_matrix.py` 从逐任务结果 JSON 生成，",
        "与 `matrix.json` 同源。未完成单元保留真实计数，不补零、不省略、不计入宏平均。",
        "",
        "## 12 行配置汇总",
        "",
        "| 模型 | 方法 | Spatial | Object | Goal | Long | 宏平均 | Δ vs 原精度 | 稳态分配峰值 GiB | 稳态进程峰值 GiB | 查询 ms | 动作步摊销 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in doc["configs"]:
        cells = []
        for suite in SUITES:
            s = row["suites"][suite]
            if not s["present"]:
                cells.append("未开始")
            elif s["complete"]:
                cells.append(pct(s["success_rate"]))
            else:
                cells.append(f"{pct(s['success_rate'])}*({s['n_episodes']}/{s['n_planned']})")
        deltas = [
            row["suites"][s]["paired_vs_fp16"]["delta_pts"]
            for s in SUITES
            if row["suites"][s].get("paired_vs_fp16")
        ]
        delta = f"{sum(deltas) / len(deltas):+.2f} pp" if len(deltas)==4 and row['n_complete_suites']==4 else "—"
        if row["method"] == BASELINE_METHOD:
            delta = "基准"
        L.append(
            "| " + " | ".join([
                row["model_label"], row["method_label"], *cells,
                pct(row["macro_success_rate"]), delta,
                gib(row["memory"].get("peak_alloc_bytes")),
                gib(row["memory"].get("nvml_peak_bytes")),
                f"{row['latency']['query_s_mean']*1000:.2f}"
                if row["latency"]["query_s_mean"] else "—",
                f"{row['latency']['per_env_step_s']*1000:.2f}"
                if row["latency"]["per_env_step_s"] else "—",
            ]) + " |"
        )
    L += [
        "",
        "`*` 标记的成功率来自未完整集合，只是进度指示，不是结果。",
        "宏平均仅在四集合全部完整时给出；Δ 为四集合配对差值的等权平均。",
        "主表资源只取通过身份校验的固定输入专测；缺失或拒绝时留白，绝不回填闭环加载峰值。",
        "每配置 200 固定输入、预热每集合 20 次、测量 3 遍共 600 次；显存取四集合最大值，延迟取四集合均值。",
        "",
        "## 量化核算（任何显存结论必须与本表同读）",
        "",
        "三模型量化了自身**很不同的参数比例**，所以单看压缩比或显存会误导：",
        "",
        "| 模型 | 方法 | 模块范围 | 量化参数占比 | 范围内压缩 | 权重字节 GiB | 方法开销 MiB | 争用 |",
        "|---|---|---|---:|---:|---:|---:|:--:|",
    ]
    for row in doc["configs"]:
        q = row["quant"] or {}
        if row["method"] == BASELINE_METHOD or not q:
            continue
        art = (q.get("artifact") or {})
        overhead = art.get("method_overhead_bytes")
        frac = q.get("quantized_param_fraction")
        comp = q.get("compression_in_scope")
        L.append("| " + " | ".join([
            row["model_label"], row["method_label"],
            "+".join(q.get("scope") or []) or "—",
            f"{100 * frac:.1f}%" if frac is not None else "—",
            f"{comp:.2f}x" if comp else "—",
            gib(row["memory"].get("weights_bytes")),
            f"{overhead / 2 ** 20:.1f}" if overhead else "—",
            "未确认" if row["contended"] is None else ("是" if row["contended"] else "否"),
        ]) + " |")
    L += [
        "",
        "`方法开销` = 高精度残差 + gather 索引 + 输入缩放字节。固定基础位宽不等于相同总预算。",
        "`争用=是` 的行其闭环资源数字不可与独占运行比较；主表资源取自固定输入专测。",
        "",
        "## 配对检验（McNemar，精确二项，同一初始状态）",
        "",
        "| 模型 | 方法 | 集合 | n | 退化 | 改善 | Δ pp | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in doc["configs"]:
        if row["method"] == BASELINE_METHOD:
            continue
        for suite in SUITES:
            d = row["suites"][suite].get("paired_vs_fp16")
            if not d:
                continue
            L.append(
                f"| {row['model_label']} | {row['method_label']} | {SUITE_LABELS[suite]} "
                f"| {d['n']} | {d['n_baseline_only']} | {d['n_variant_only']} "
                f"| {d['delta_pts']:+.2f} | {d['p_value']:.4g} |"
            )
    L += ["", "p 为未校正的探索性配对检验；12 次比较若采用 Bonferroni 校正，阈值为 0.05/12。", "",
          "## 固定输入资源来源", "",
          "| 模型 | 方法 | 状态 | GPU | 结果文件 |", "|---|---|---|---:|---|"]
    for row in doc['configs']:
        b=row['resource_bench']
        L.append(f"| {row['model_label']} | {row['method_label']} | {b['status']} | {b.get('gpu','—')} | {b['source']} |")
    L += ["", "输入来自训练成功演示的开发划分，仅用于资源测量，不能推导成功率。",
          "VLA-Adapter 原精度/RTN 使用不同 L40 卡，未固定频率；历史专测缺 checkpoint 文件哈希，当前绑定依据为路径、量化规格及策略参数。",
          "加载期峰值仍单独保存在 matrix.json，不代表量化模型可以直接在稳态容量的设备上加载。", ""]
    legacy = [c for c in doc["cells"] if not c["memory"].get("load_phase_separated")]
    L += ["", "## 指标口径", "",
          f"- 显存（schema/2，当前 {len(legacy)} 个单元）：{MEMORY_CAVEAT_LEGACY}",
          f"- 显存（schema/3）：{MEMORY_CAVEAT_CURRENT}",
          f"- 延迟：{LATENCY_CAVEAT}", ""]
    L += [
        "## 48 单元明细",
        "",
        "| 模型 | 方法 | 集合 | 完整 | 任务 | 回合 | 成功 | 超时 | 模型失败 | INVALID | 成功率 | schema | 争用 |",
        "|---|---|---|:--:|---:|---:|---:|---:|---:|---:|---:|---|:--:|",
    ]
    for cell in doc["cells"]:
        L.append("| " + " | ".join([
            MODEL_LABELS[cell["model"]], METHOD_LABELS[cell["method"]],
            SUITE_LABELS[cell["suite"]],
            "✓" if cell["complete"] else "—",
            f"{cell['n_tasks']}/10", f"{cell['n_episodes']}/{cell['n_planned']}",
            str(cell["n_success"]), str(cell["n_timeout"]),
            str(cell["n_policy_error"]), str(cell["n_invalid"]),
            pct(cell["success_rate"]),
            ",".join(s.rsplit("/", 1)[-1] for s in cell["schema_versions"]) or "—",
            "是" if cell["contended"] else "否",
        ]) + " |")
    missing_n = len(doc["summary"]["missing_cells"])
    L += ["", f"另有 {missing_n} 个单元未开始，见文末清单。", ""]
    if doc["problems"]:
        L += ["## 完整性问题", ""] + [f"- {p}" for p in doc["problems"]] + [""]
    if doc["summary"]["missing_cells"]:
        L += ["## 缺失单元", ""]
        for key in doc["summary"]["missing_cells"]:
            L.append(f"- {key}")
        L.append("")
    return "\n".join(L)


def write_csv(doc: dict, path: Path) -> None:
    """One row per saved cell, for the write-up. Same numbers as matrix.json."""
    import csv

    columns = ["model", "method", "suite", "complete", "n_tasks", "n_episodes",
               "n_planned", "n_success", "n_timeout", "n_policy_error", "n_invalid",
               "success_rate", "contended", "schema_versions",
               "query_s_mean", "per_env_step_s",
               "weights_bytes", "peak_alloc_bytes", "nvml_peak_bytes",
               "load_peak_alloc_bytes", "load_phase_separated",
               "scope", "quantized_param_fraction", "compression_in_scope",
               "method_overhead_bytes", "implementation",
               "delta_pts_vs_fp16", "mcnemar_p", "resource_source", "resource_inputs_manifest_sha256"]
    by_key = {(c["model"], c["method"], c["suite"]): c for c in doc["cells"]}
    paired = {(r["model"], r["method"], s): r["suites"][s].get("paired_vs_fp16")
              for r in doc["configs"] for s in doc["suites"]}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for key, cell in sorted(by_key.items()):
            q = cell["quant"] or {}
            art = q.get("artifact") or {}
            bench=cell.get('resource_bench') or {}
            mem = bench.get('memory') or {}
            lat = bench.get('latency') or {}
            delta = paired.get(key) or {}
            writer.writerow({
                "model": cell["model"], "method": cell["method"], "suite": cell["suite"],
                "complete": cell["complete"], "n_tasks": cell["n_tasks"],
                "n_episodes": cell["n_episodes"], "n_planned": cell["n_planned"],
                "n_success": cell["n_success"], "n_timeout": cell["n_timeout"],
                "n_policy_error": cell["n_policy_error"], "n_invalid": cell["n_invalid"],
                "success_rate": cell["success_rate"], "contended": cell["contended"],
                "schema_versions": ";".join(cell["schema_versions"]),
                "query_s_mean": lat.get('query_s',{}).get('mean'),
                "per_env_step_s": lat.get('per_env_step_s'),
                "weights_bytes": mem.get("weights_bytes"),
                "peak_alloc_bytes": mem.get("peak_alloc_bytes"),
                "nvml_peak_bytes": mem.get("nvml_peak_bytes"),
                "load_peak_alloc_bytes": mem.get("load_peak_alloc_bytes"),
                "load_phase_separated": bool(bench) if bench else None,
                "scope": "+".join(q.get("scope") or []),
                "quantized_param_fraction": q.get("quantized_param_fraction"),
                "compression_in_scope": q.get("compression_in_scope"),
                "method_overhead_bytes": art.get("method_overhead_bytes"),
                "implementation": q.get("implementation"),
                "delta_pts_vs_fp16": delta.get("delta_pts"),
                "mcnemar_p": delta.get("p_value"),
                "resource_source": bench.get('source'),
                "resource_inputs_manifest_sha256": bench.get('inputs_manifest_sha256'),
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/matrix")
    ap.add_argument("--source", action="append", default=[], metavar="METHOD=PATH",
                    help="read a method from a different directory, e.g. "
                         "awq=results/validation_awq. Use for development runs; "
                         "the override is recorded in the output so a matrix built "
                         "from non-standard sources is never mistaken for the real one.")
    args = ap.parse_args()

    overrides = {}
    for item in args.source:
        if "=" not in item:
            ap.error(f"--source expects METHOD=PATH, got {item!r}")
        method, _, path = item.partition("=")
        if method not in METHOD_SOURCES:
            ap.error(f"--source: unknown method {method!r}")
        METHOD_SOURCES[method] = (path,)
        overrides[method] = path

    cells, problems = load_cells()
    validate_episode_manifest(cells, problems)
    configs = build_configs(cells)
    attach_resource_benches(configs, cells, problems)

    total = len(MODELS) * len(METHOD_SOURCES) * len(SUITES)
    complete = sum(1 for c in cells.values() if c["complete"])
    missing = [
        f"{m}/{q}/{s}"
        for m in MODELS for q in METHOD_SOURCES for s in SUITES
        if (m, q, s) not in cells
    ]
    incomplete = [
        f"{c['model']}/{c['method']}/{c['suite']} "
        f"({c['n_episodes']}/{c['n_planned']}，缺 task {c['missing_task_ids']})"
        for c in cells.values() if not c["complete"]
    ]
    doc = {
        "schema": "embodiq/result_matrix/2",
        "generated": time.strftime("%FT%T%z"),
        "task": "1.4 VLA 与 WAM 模型适配及任务验证",
        "protocol": "unified.v3",
        # Non-empty means this matrix is NOT built from the standard directories
        # and must not be quoted as the task's result.
        "source_overrides": overrides,
        "models": list(MODELS),
        "methods": list(METHOD_SOURCES),
        "suites": list(SUITES),
        "labels": {"model": MODEL_LABELS, "method": METHOD_LABELS, "suite": SUITE_LABELS},
        "caveats": {"memory_legacy": MEMORY_CAVEAT_LEGACY,
                     "memory_current": MEMORY_CAVEAT_CURRENT,
                     "latency": LATENCY_CAVEAT},
        "summary": {
            "total_cells": total,
            "complete_cells": complete,
            "incomplete_cells": incomplete,
            "missing_cells": missing,
            "saved_episodes": sum(c["n_episodes"] for c in cells.values()),
            "planned_episodes": total * EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK,
            "complete_resource_configs": sum(r['resource_bench']['status']=='complete' for r in configs),
            "pending_methods": [m for m in METHOD_SOURCES if not any(c['method']==m for c in cells.values())],
            "任务完成条件": "48 单元无缺失且 12 配置均有真实资源数据；当前未满足即为未完成。",
        },
        "configs": configs,
        "cells": [
            {k: v for k, v in c.items() if k != "_episodes"}
            for c in sorted(cells.values(),
                            key=lambda c: (c["model"], c["method"], c["suite"]))
        ],
        "problems": problems,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "matrix.md").write_text(render_markdown(doc), encoding="utf-8")
    write_csv(doc, args.out / "cells.csv")

    print(f"cells complete {complete}/{total}; "
          f"episodes {doc['summary']['saved_episodes']}/{doc['summary']['planned_episodes']}")
    for p in problems:
        print(f"  PROBLEM {p}")
    for i in incomplete:
        print(f"  INCOMPLETE {i}")
    print(f"wrote {args.out / 'matrix.json'}, matrix.md and cells.csv")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
