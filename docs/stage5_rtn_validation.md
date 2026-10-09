# RTN 量化验证阶段

## 2026-10-03 核验更正（本节为最新状态，下文保留历史记录）

下文"5050/6000 回合、10/12 集合"及 OpenVLA Goal/Long 未完成的快照**已过期**。
实际状态：

- **原精度：12/12 集合、6000/6000 回合全部完成并通过逐回合核验**
  （OpenVLA 补齐为 Goal 78.20%、Long 52.20%，宏平均 76.95%）。
- **RTN：10.9/12 集合**。缺口为 OpenVLA `libero_10` 全部缺失（0/500）、
  OpenVLA `libero_goal` 缺 task 9（450/500）。
  **2026-10-04 矫正**：`libero_goal` 已补齐 10/10（75.60%，配对 Δ −2.60 pp），
  RTN 现为 **11/12 集合**；`libero_10` 四卡 8 进程跑中。

缺口原因已定位：项目目录自 `/data/jyc/space/Embodied AI` 改名为
`/data/jyc/space/Embodied`，控制器 PID 715299 在写 `status.tmp` 时因旧绝对路径
失效而崩溃（`results/stage5_rtn_full/controller.log` 末尾 `FileNotFoundError`），
导致 Goal task 9 在第 6 回合被截断、Long 从未入队。`status.json` 至今把
OpenVLA Goal 留在 `running`，但该进程已不存在。
2026-10-03 已建立 `Embodied AI -> Embodied` 符号链接恢复可运行性。

RTN 配对结果（McNemar 精确二项，同一初始状态）与资源数字见
`docs/task14_gap_analysis.md` §2.2 与 §3.1。**注意其中的峰值显存口径有缺陷
（`nvml_peak` 量化后反升），修复前不得写入正式报告。**

## 原精度结果快照

本轮检查时原精度已落盘 5050/6000 回合、10/12 完整集合（2026-10-04 矫正：原精度已 12/12、6000/6000 全部完成并逐回合核验）；逐回合初始状态哈希/种子验证一致，已落盘任务无错误。

| 模型 | Spatial | Object | Goal | Long | 四集合平均 |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 88.2% | 89.2% | 未完成 | 未开始 | — |
| VLA-Adapter Pro | 98.0% | 99.4% | 97.4% | 95.4% | 97.55% |
| FastWAM | 97.0% | 99.2% | 97.4% | 93.6% | 96.80% |

`tools/report_baseline.py` 可重新核查并刷新 `baseline_unified_v3_20260919/baseline_summary.md/json`。OpenVLA Goal 正常运行，Long 在其后；此快照不把未落盘回合记为已完成。

## 隔离开发与实现

运行中的基线源码保持不变。RTN 工作副本位于 `experiments/quant_stage5/src`，包含当时原框架的副本；以独立 PYTHONPATH 运行，记录自身源代码 SHA256。

修复 FastWAM 的实际模块别名匹配：named_modules 首先看到 video_expert/action_expert，不是旧规则中的 mot.mixtures；别名去重后正确选择一次，不重复量化共享模块。

加入防护：选择零个量化层直接报错；AWQ/HSA 没有校准产物时明确拒绝，不能仅换标签执行 RTN。量化后统计所有参数/缓冲区的实际唯一 storage 字节数，避免把仅 Linear 范围的估算写成整模型压缩。

## 当前固定 RTN 试验设置

- W4A16，非对称 group size=128，真实 packed tinygemm 内核。
- OpenVLA：LLM blocks。
- VLA-Adapter：LLM、两视觉编码器、projector、action head；不可分组层按现有规则回退 INT8，报告中单列，不能称为纯 INT4 全模型。
- FastWAM：video/action DiT blocks，保留文本编码器、VAE、关键调制/I/O 等原精度。
- 原始 checkpoint、预处理、unified.v3 协议、seed=42、初始状态和模型动作节奏与原精度一致。
- 此范围为 RTN 试验配置，后续 AWQ/HSA 公平比较需沿用相同范围或显式做预算匹配，不能挑选不同范围的最优结果拼表。

## 验证结果

两套环境各通过 15 项测试（PyTorch 2.2.0+cu121 与 2.6.0），包含真实 GPU INT4 packing/反量化一致性和方法/空范围防护。

Spatial task 0、两回合、同一初始状态与种子的 RTN 闭环：

| 模型 | RTN 成功 | 运行错误 | 真实模块替换 |
|---|---|---|---|
| OpenVLA | 1/2（另一个超时） | 0 | 224 个 INT4 Linear |
| VLA-Adapter | 2/2 | 0 | 568 个 INT4、28 个 INT8 Linear |
| FastWAM | 2/2 | 0 | 600 个 INT4 Linear |

这证明部署与闭环链路可运行；两回合样本不能推断整体成功率损失，更不能宣布 RTN 无损。

结果路径：`results/stage5_rtn_pilot/<model>/libero_spatial/*_results.json`，逐层报告在同模型 `quant_reports/`。

重要口径：旧底层 render/to_dict 的 `compression_whole_model` 实际是 Linear 范围估计。主结果 quant 字段已将其重命名为 `compression_linear_modules_estimate`；整模型张量存储使用 `tensor_storage_bytes_before/after` 和 `tensor_storage_compression`。张量存储不等于峰值显存，也不包含所有运行时激活/缓存；不能据此宣称同幅度显存下降或推理加速。

## 已启动下一步

`tools/run_rtn_validation.py --background` 启动全量 RTN 配对验证，启动时 PID 715299。

- 状态：`results/stage5_rtn_full/status.json`。
- 三个模型分别使用 GPU 0/2/3；OpenVLA 原精度继续使用 GPU 1。
- 每个集合仅在对应原精度 500 回合完整且无错误时开始；OpenVLA Goal/Long 不会在缺少完整基线时进入比较。
- 独立源码及原生源码均受哈希保护；出错停止对应队列，不自动重试。
- 目标为三模型四集合各 500 回合，最多 6000 个 RTN 回合；等待基线最多 48 小时，超时明确标记。

尚未完成：全量 RTN 结果、AWQ 激活校准与缩放搜索、HSA 选择与残差工件链路、固定输入显存/延迟专测。当前没有宣称最终 48 单元目标完成。
