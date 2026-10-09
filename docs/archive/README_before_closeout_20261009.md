> 历史快照：文内相对链接以原 docs/ 目录为基准，当前状态见 ../README.md。

# embodiq — 任务 1.4 统一评测框架

**现状核查更新：2026-10-09；下文主体保留 2026-10-04 阶段快照。**

当前进展与后续清单请先看 [2026-10-09 核查报告](task14_status_20261009.md)。
按用户要求已完成两次[公开 AWQ 代码适配尝试](awq_public_attempt_20261009.md)：
OpenVLA 在 4/32 条训练校准观测下均未通过现有权重重构门禁，已停止；
AWQ/HSA 正式结果继续留白，后续优先完成已有原精度/RTN 的汇总与交付。
本文件下方阶段快照的四处结论已被实际代码和新结果更新：资源专测已完成
**6/12 配置**；AWQ 全范围尝试尚未通过数值校验；配对检验并非仅一处未校正显著；
最终汇总接入资源专测、HSA 批量调度工件传参及真实 IDM 验证仍有收尾工作。
正式成功率仍为 **24/48 单元、12,000 回合**，AWQ/HSA 正式结果保留空位。
其余文档多为时间线记录，应结合日期和新核查结论阅读。

---

## 1. 任务范围

**1.4 VLA 与 WAM 模型适配及任务验证。** 交付物四件:

1. 统一适配模块(图像/语言指令/机器人状态/动作输出/时序预测五类接口)
2. 使原始精度、RTN、AWQ、HSA 四种模型复用**同一评测入口**
3. 在 LIBERO 四集合(空间关系/目标物体/任务目标理解/长时程)上验证
4. 统计各集合成功率、平均成功率、峰值显存、查询延迟,分析性能-资源权衡

**明确不在本项目范围内**(2026-10-03 确认):1.1 的模块敏感度实验、
1.2 的 8 位量化、1.3 的残差策略消融、1.5 的六维配置机制/使用手册/年度固化/端侧硬件。
**AWQ(1.2)与 HSA(1.3)的算法由对应子任务提供,1.4 只负责接入与评测。**

可视化页面不是交付物。若后续需要,必须是 `results/matrix/matrix.json` 的只读渲染。

---

## 2. 框架状态:设计与流程已端到端验证跑通

| 组件 | 状态 | 证据 |
|---|---|---|
| 统一适配模块(三模型五类接口) | 完成 | `core/types.py` 契约 + `tests/test_contract_and_rollout.py` |
| FastWAM 时序预测开关 | 完成 | `--fastwam-mode {first_frame,idm}`,9 项测试 |
| 统一评测入口(四方法) | **完成并实跑验证** | AWQ/HSA 正负向均实测,见 §4 |
| 量化工件接入契约 | 完成 | `quant/artifact.py`,33 项测试;接入文档 + **精确层名清单**已交付 |
| 回合级检查点 | 完成 | 停止代价从一个任务降到一个回合,19 项测试 |
| 四项指标 | 完成 | 显存口径已修正并在量化运行上验证 |
| 固定输入资源专测 | 完成并跑通 | 200 输入 / 600 次测量 |
| 结果汇总 | 完成 | `tools/report_matrix.py`,单一数据源 |
| 控制器 durability | 完成 | 原子写、崩溃留痕、按任务续跑、干净停止、GPU 停驻、多卡分片 |

**剩下的全是数据生产,不是框架缺口。**

---

## 3. 数据进度

### 3.1 结果矩阵:24/48 完整单元(12000/24000 回合)

原精度与 RTN 两列**已全部闭合**,零错误、零 INVALID。

| 模型 | 方法 | Spatial | Object | Goal | Long | 宏平均 | Δ vs 原精度 |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenVLA | 原精度 | 88.20% | 89.20% | 78.20% | 52.20% | **76.95%** | 基准 |
| OpenVLA | RTN | 82.60% | 87.80% | 75.60% | 49.80% | **73.95%** | **−3.00 pp** |
| VLA-Adapter Pro | 原精度 | 98.00% | 99.40% | 97.40% | 95.40% | **97.55%** | 基准 |
| VLA-Adapter Pro | RTN | 98.60% | 95.20% | 87.60% | 93.60% | **93.75%** | **−3.80 pp** |
| FastWAM | 原精度 | 97.00% | 99.20% | 97.40% | 93.60% | **96.80%** | 基准 |
| FastWAM | RTN | 96.60% | 99.40% | 97.00% | 94.40% | **96.85%** | **+0.05 pp** |
| 三模型 | AWQ / HSA | — | — | — | — | — | 等 1.2/1.3 交付工件 |

**12 个 RTN 单元的 `embodiq_source_sha256` 全部为 `47043e48…`**,与隔离树
`experiments/quant_stage5/src` 的指纹一致——整列逐字节可比(这是 2026-10-04
恢复规范路径而非修改冻结树的目的)。

配对 McNemar 只有一处显著:OpenVLA Spatial −5.60 pp(p=0.0078)。
其余净差值都被单回合翻转噪声淹没,OpenVLA Long 的不一致对高达 109/97
(500 回合里 41% 向某一方向翻转),净差只有 −2.40 pp(p=0.44)。
**报净差而不报不一致对会严重夸大结论强度。**

权威表随时重算:`python tools/report_matrix.py` → `results/matrix/matrix.md`。

**原精度基线已逐回合核验**:12/12 集合、6000/6000 回合,每个回合的 `seed` 与
`initial_state_sha256` 与 `episode_manifest.json` 逐项吻合,零错误。
这是配对 McNemar 成立的前提。输入清单 132 个文件 SHA256 全量重算零不匹配。

### 3.2 补跑已完成

`openvla/rtn/libero_10` 于 2026-10-04 完成(四卡 8 进程,`phase=complete`)。
逐回合核验通过:10 任务 × 50 回合、零错误、每回合 seed 与
`initial_state_sha256` 与 `episode_manifest.json` 吻合。

该单元 `contended=true`(四卡 8 进程),其**闭环**资源数字不可与其余单元比较;
成功率不受影响(rollout 在给定 seed 与初始状态下确定,步数上限是步数而非秒数)。

```bash
python tools/run_rtn_validation.py --status    # 各单元 complete/partial/empty
python tools/run_rtn_validation.py --stop      # 随时停止,代价是进行中的那一个任务
```

### 3.3 资源专测:1/12 配置

`results/resource_bench/vla_adapter_fp16.json`(四集合 × 50 输入 × 3 遍 = 600 次测量)。
与闭环互相印证:`weights` 2.97 GiB、稳态 `peak_alloc` 3.55 GiB 完全一致。

---

## 4. 两个必须知道的指标口径

### 4.1 峰值显存:加载期与稳态是两个数

量化在同进程 BF16 权重加载**之后**进行,`reset_peak_memory_stats` 清零计数器
但不归还内存池,NVML 读的是进程真实占用——于是量化后的进程峰值会把加载期
footprint 读回来。实测后果曾是 RTN 显存"上升"(FastWAM 24.84 → 33.15 GiB),
而稳态分配其实正常下降(23.26 → 15.24 GiB)。

**已修复**(`MemoryProbe.mark_after_load`:记录加载期峰值 → `empty_cache()`
归还池 → 重置计数开始稳态测量),结果 schema 升至 `embodiq/run_result/3`,
新增 `load_peak_*` 与 `cache_released_bytes`。HSA 实测验证:

| | 加载期 | 稳态 |
|---|---:|---:|
| `peak_alloc` | 3.98 GiB | **1.86 GiB** |
| `nvml_peak` | 5.12 GiB | **3.70 GiB** |

归还缓存 2.60 GiB。对比 fp16 的 5.68 GiB,量化后下降 35%。

**注意:现有 23 个单元均为 `schema/2`,其 `nvml_peak`/`peak_reserved` 被加载期
污染,不可用于量化前后的峰值对比;读 `peak_alloc`。** `report_matrix.py`
按每个单元自己的 schema 版本输出对应说明。

### 4.2 查询延迟:只有 `per_env_step_s` 跨模型可比

三模型查询密度为 1 : 1/8 : 1/10(OpenVLA 每步重规划、VLA-Adapter 每 8 步、
FastWAM 每 10 步)。`query_s` 只能同模型内比较。报单一个"推理延迟"会差 10 倍。

另:`libero_10` 的 RTN 单元因四卡 8 进程并行被标 `contended=true`,其**闭环**
资源数字不可与其余单元比较;成功率不受影响(rollout 在给定 seed 与初始状态下
确定,步数上限是步数而非秒数)。主表资源数字取自 §3.3 的独占 GPU 专测。

---

## 5. 常用命令

```bash
# 评测入口(四方法同一入口)
python -m embodiq.entry.evaluate --model {openvla,vla_adapter,fastwam} \
  --checkpoint <ckpt> --suite libero_spatial --protocol unified.v3 \
  --quant {fp16,rtn,awq,hsa} [--quant-artifact <path>] [--hsa-rho 0.01] \
  [--fastwam-mode {first_frame,idm}] --trials 50 --gpu 0 --out <dir>

# 结果矩阵(唯一数据源)
python tools/report_matrix.py          # -> matrix.json / matrix.md / cells.csv

# 交给 1.2/1.3 的精确层名清单
python tools/export_scope_manifest.py  # -> artifacts/scope_manifest.json
python tools/report_matrix.py --source awq=results/validation_awq   # 开发用,会标注非标准来源

# 原精度基线逐回合核验
python tools/report_baseline.py

# 资源专测
python tools/run_resource_bench.py --make-inputs                    # 一次性,生成 200 条固定输入
python tools/run_resource_bench.py --model vla_adapter --quant rtn --gpu 2

# RTN 控制器
python tools/run_rtn_validation.py --status
python tools/run_rtn_validation.py --stop
python tools/run_rtn_validation.py --background --resume --models openvla \
  --gpus "openvla=0,1,2,3,0,1,2,3" --min-free-gib 36

# 测试(无需 checkpoint 的门禁)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q tests/
# 110 passed, 9 skipped — 跳过项需 checkpoint 和/或 CUDA
# 其中 test_quant_artifact 33、test_controller_status 22、test_fastwam_temporal_mode 9
```

环境:三个独立 conda env(`openvla` / `vlaadapter` / `fastwam`),
均在 MuJoCo 3.3.2 + robosuite 1.4.1。路径 `/data/jyc/space/Embodied AI`
为规范路径,`/data/jyc/space/Embodied` 为符号链接,两者均可用。

---

## 6. 还需要做什么

| 项 | 阻塞于 | 预计 |
|---|---|---|
| AWQ / HSA 各 12 单元 | **1.2/1.3 交付工件** | 通路已验证,层名清单已交付,交来即可跑 |
| 11 个资源专测配置 | **四卡现已空闲,可开始** | 每配置约 10 min × 4 checkpoint |
| 三轴权衡分析与图 | 以上数据 | 约半天 |

**未提交风险**:工作区有大量未提交改动,最后一次提交为 2026-09-02 `9846c1a`。
结果 JSON 记录 `embodiq_source_sha256`,但该代码无法从 git 恢复——
协议 §9 的"能够重跑并从原始日志生成报告"按现状不满足。建议尽快提交并打 tag。

---

## 7. 文档索引

### 当前有效

| 文档 | 内容 |
|---|---|
| **本文件** | 现状、指标口径、命令、剩余工作 |
| `quant_artifact_contract.md` | **交给 1.2/1.3 的接入契约**:格式、校验、每模型层数、验收门禁 |
| `experiment_protocol.md` | 实验协议(目标、矩阵、指标定义、异常规则、验收条件);见文末修订 |
| `interface_contract.md` | 三模型五类接口的逐项绑定与指标定义依据 |
| `stage_a_contract.md` | `Policy` 契约与无 checkpoint 门禁 |

### 时间线记录(早期章节可能已被推翻)

| 文档 | 内容 | 注意 |
|---|---|---|
| `task14_gap_analysis.md` | 2026-10-03/04 的完整盘点与执行记录(§1–§10) | §1–§3 的问题已在 §7–§10 修复;以本文件为准 |
| `stage4_unified_baseline.md` | 原精度基线流水线 | 已追加 10-03 核验更正 |
| `stage5_rtn_validation.md` | RTN 验证 | 已追加 10-03 核验更正 |
| `stage6_awq_preparation.md` | AWQ 校准准备 | 已追加 10-04 更正 |
| `baseline_review_and_quant_next.md` | 基线复核与中断恢复 | 已追加 10-04 更正 |
| `stage2_progress.md` | 权重盘点与早期适配 | 纯历史 |
| `stage3_integration.md`、`stage3_fastwam_adapter.md`、`stage3_gpu_validation.md` | FastWAM 接入与 GPU 验证 | 纯历史 |
| `stage_b_preview.md`、`stage_b_fp16_baseline.md` | VLA-Adapter 原生协议基线(非 unified.v3) | 纯历史;其延迟不得进入正式主表 |
