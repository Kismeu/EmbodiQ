"""Validate saved baseline episodes and report complete suites only."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/baseline_unified_v3_20260919"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    refs = {(r["suite"], r["task_id"]): r for r in json.loads((RUN / "episode_manifest.json").read_text())["tasks"]}
    output = {"updated": time.strftime("%FT%T%z"), "models": {}, "saved_episodes": 0, "complete_suites": 0}
    lines = ["# 原精度基线结果快照", "", output["updated"], "",
             "仅对完整集合给出集合成功率；未落盘回合不纳入已完成数量。", "",
             "| 模型 | Spatial | Object | Goal | Long | 四集合平均 |", "|---|---:|---:|---:|---:|---:|"]
    for model in ("openvla", "vla_adapter", "fastwam"):
        data = {}
        cells = []
        for suite in SUITES:
            ds = [json.loads(p.read_text()) for p in (RUN / "full" / model / suite).glob("*_results.json")]
            task_ids = [d["task_id"] for d in ds]
            assert len(set(task_ids)) == len(task_ids), "Duplicate task files"
            for d in ds:
                assert d["policy"]["family"] == model and d["protocol"]["name"] == "unified.v3"
                assert not d["error_episodes"] and len(d["episodes"]) == d["total_episodes"] == 50
                assert {e["episode_idx"] for e in d["episodes"]} == set(range(50))
                assert d["successes"] == sum(e["status"] == "success" for e in d["episodes"])
                for e in d["episodes"]:
                    ref = refs[(suite, d["task_id"])]["episodes"][e["episode_idx"]]
                    assert e["seed"] == ref["seed"] and e["aux"]["initial_state_sha256"] == ref["state_sha256"]
            n = sum(d["total_episodes"] for d in ds)
            s = sum(d["successes"] for d in ds)
            complete = set(task_ids) == set(range(10)) and n == 500
            data[suite] = dict(tasks=len(ds), episodes=n, successes=s, complete=complete,
                               success_rate=s / n if complete else None)
            output["saved_episodes"] += n
            output["complete_suites"] += int(complete)
            cells.append(f"{100*s/n:.2f}%" if complete else f"未完成（{n}/500）")
        avg = sum(x["success_rate"] for x in data.values()) / 4 if all(x["complete"] for x in data.values()) else None
        output["models"][model] = dict(suites=data, macro_average=avg)
        lines.append("| " + " | ".join([model] + cells + ([f"{100*avg:.2f}%"] if avg is not None else ["—"])) + " |")
    lines += ["", f"已保存 {output['saved_episodes']}/6000 回合；完整集合 {output['complete_suites']}/12。",
              "", "查询延迟/显存的严格部署对比另行执行固定输入资源专测，本表不据零散回合推断加速。"]
    (RUN / "baseline_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (RUN / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__": main()
