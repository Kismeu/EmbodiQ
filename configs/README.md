# 实验配置

`stage_fp16_rtn.json` 是当前可执行的主配置，覆盖三模型与原始精度/RTN。支持 `${EMB_ROOT}`、`${DATA_ROOT}`、`${OPENVLA_ROOT}` 路径变量；模型解释器通过 `EMBODIQ_PYTHON_OPENVLA`、`EMBODIQ_PYTHON_VLA_ADAPTER`、`EMBODIQ_PYTHON_FASTWAM` 设置。

`stage_awq.json`、`stage_full_pending_hsa.json` 是待接入方法的开发模板。它们可以用于检查工件接口与展开计划，但不代表 AWQ/HSA 已完成。自动 `launch` 与网页入口均拒绝执行这两种待接入方法。

```bash
embodiq validate-config configs/stage_fp16_rtn.json
embodiq launch configs/stage_fp16_rtn.json --profile formal --dry-run
embodiq web --config configs/stage_fp16_rtn.json
```

本机路径建议放在 `configs/*.local.json`，这些文件已被 Git 忽略。

旧 `run` 命令只创建计划；`execute` / `resume` 运行单个旧式任务。整批自动流程使用 `launch` / `resume-run`，用法见 [安装与配置](../docs/getting_started.md)和[自动评测说明](../docs/automatic_evaluation.md)。
