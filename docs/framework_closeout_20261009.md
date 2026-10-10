# 任务 1.4 框架收尾记录

> 后续更新：本机 CLI 的一键式原精度/RTN 流程现已实现，见[自动评测说明](automatic_evaluation.md)。本记录与 `task14_framework_20261009` 快照保留平台化之前的交付边界。

日期：2026-10-09。范围：统一评测底座、既有原精度/RTN 实验的交付与追溯。
AWQ/HSA 按用户决定暂时留白；自动评测平台是下一阶段，见[平台化方案](platform_research_20261009.md)。

## 已完成的收尾

1. `tools/report_matrix.py` 将六组固定输入资源专测接入正式 JSON、Markdown 和 CSV。
   主资源列只使用通过来源与条件检查的专测结果，历史闭环资源仍以 `rollout_*` 保存。
   新增性能与显存、动作步摊销推理耗时对比图，均从同一份 `matrix.json` 生成。
2. `core/admission.py` 与汇总器拒绝开发工件、混合实现/协议/范围、缺失或重复任务、
   回合计数不一致、非冻结初始状态和失配资源结果。AWQ/HSA 正式工件要求明确
   `formal=true`、`validation_only=false`、实现版本与来源、校准和完整层覆盖。
   缺失方法留白，不补零，不回退为 RTN。
3. 批量入口补齐 AWQ/HSA 的显式工件路径或目录解析，HSA 的 `rho` 校验与传递。
   预检只检查路径与必要文件是否存在；工件内容由评测加载器校验，正式汇总另有准入检查。
4. 修复 FastWAM IDM 原先选用 uncond 配置的问题，选择真实 OptionalIDM 配置，
   并通过未来视频去噪前向与一个真实 LIBERO 回合验证。
5. 提供 `tools/freeze_release.py`，固化工作区代码、逐任务证据、资源测量、图表、
   三套环境包版本和本地上游相关源码；逐文件 SHA256 及压缩包校验均可检查。

## 实验结果与边界

正式结果沿用并核查既有实验，本次没有重跑 12,000 回合。
三模型 × 两方法 × 四集合，共 24 个完整单元、240 个任务文件、12,000 回合。
统一矩阵 `problems=[]`；AWQ/HSA 24 个预留单元为空，四方法总任务尚未完成。

| 模型 | 原精度成功率 | RTN 成功率 | 原精度 / RTN 稳态 NVML 峰值 GiB | 原精度 / RTN 查询 ms |
|---|---:|---:|---:|---:|
| OpenVLA | 76.95% | 73.95% | 15.01 / 6.70 | 207.33 / 318.80 |
| VLA-Adapter Pro | 97.55% | 93.75% | 4.47 / 3.49 | 64.88 / 86.15 |
| FastWAM first-frame | 96.80% | 96.85% | 24.02 / 16.74 | 311.84 / 370.32 |

成功率为四集合宏平均；查询耗时为固定输入专测，不能跨模型忽略动作块长度直接比较效率。
当前 RTN 降低显存、增加查询耗时，不能宣传为推理加速。FastWAM 的 +0.05 pp 不作为提升证据。
原精度配置名为 `fp16`，记录中的实际模型 dtype 为 BF16。

证据：[正式矩阵](../results/matrix/matrix.md)、[机器可读矩阵](../results/matrix/matrix.json)、
[逐集合 CSV](../results/matrix/cells.csv)、[对比图](../results/matrix/tradeoffs.png)。

资源结果的限制：

- 六配置各 600 次测量，均报告无 GPU 争用，共用冻结的 200 条输入。
- 资源输入来自训练演示中已划分的开发观测，主要是成功演示；它衡量统一输入条件下的推理资源，
  不代表失败状态的完整分布，也不是闭环泛化实验。与原协议期望的状态覆盖差异已保留在结果元数据。
- 旧闭环 schema/2 的 NVML/reserved 峰值可能混入加载缓存，未用于主表资源对比。
- 历史专测没有 checkpoint 文件哈希；当前检查路径、策略参数、量化规格和来源，
  不能据此证明历史权重文件内容未变化。后续新运行需补齐内容哈希。

## FastWAM IDM 真实验证

使用 `libero_optional_idm_2cam224.pt`、对应归一化统计与 `FastWAMOptionalIDM`。
一次开发输入查询返回 `(32, 7)` 有限动作，执行长度为 10；视频去噪调用 10 次，
视频 latent 形状为 `[1, 48, 3, 14, 28]`，确认存在未来时间维。
随后通过统一 rollout 跑 Spatial task 0，一个回合成功，70 个环境步、7 次模型查询。

GPU 0 首次尝试因其他进程占用导致 OOM，失败记录也保留；换空闲 GPU 3 后通过。
这是接口实测，`formal=false`，不混入 first-frame 主矩阵，也不宣称完成 IDM 四集合评测。

## 验证

| 检查 | 结果 | 日志 |
|---|---|---|
| OpenVLA 环境主测试集 | 182 passed，23 skipped | [日志](../results/framework_closeout_20261009/tests_openvla.log) |
| FastWAM 环境适配与时序测试 | 17 passed，3 warnings | [日志](../results/framework_closeout_20261009/tests_fastwam.log) |
| 最后新增冻结回合准入后的汇总测试 | 24 passed | [日志](../results/framework_closeout_20261009/tests_report.log) |

这些检查有重叠，不相加为独立用例总数。跳过项不计为通过。
绘图已做视觉检查；表格重新生成后仍为 24 个完整单元、12,000 回合、六组完整资源、无完整性问题。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /data/jyc/tools/anaconda3/envs/openvla/bin/python -m pytest -q tests/
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /data/jyc/tools/anaconda3/envs/fastwam/bin/python -m pytest -q \
  tests/test_fastwam_adapter.py tests/test_fastwam_temporal_mode.py
python tools/report_matrix.py
/data/jyc/tools/anaconda3/envs/openvla/bin/python tools/plot_tradeoffs.py
```

## 源码与证据固化

交付目录：`releases/task14_framework_20261009/`，含 `source_and_evidence.tar.gz`、
`manifest.json` 与 `SHA256SUMS`。生成器拒绝覆盖已有交付目录。
本轮固化包含工作区尚未提交的真实文件；Git HEAD 只是基线，不声称工作区干净或已提交全部变更。

```bash
python tools/freeze_release.py --out releases/task14_framework_20261009
cd releases/task14_framework_20261009
sha256sum -c SHA256SUMS
# 如需恢复，解压到一个新建的空目录；不要直接覆盖现有实验目录。
tar -xzf source_and_evidence.tar.gz -C /path/to/new-empty-directory
```

上游相关源码保存在压缩包 `upstream/`，各仓库基线提交见 manifest。
恢复环境时按对应提交取得完整上游仓库，再核对和恢复归档内源码；这不是包含所有依赖的容器镜像。
大权重、数据集和仿真资产仍位于外部路径，未复制、未在固化过程中重新计算所有大文件哈希；
历史输入清单中的期望值保留供后续验证。归档保证代码及现有证据可追溯，不等同于全新环境已复现实验。

## 下一阶段

当前框架已有配置、计划、单任务执行、状态、日志与回合续跑；`run` 命令本身只创建计划。
整批调度、GPU 预约、停止进程与崩溃恢复、资源专测自动衔接、按运行绑定报告和 Web/API
仍需实现。具体以[平台方案](platform_research_20261009.md)为准。
