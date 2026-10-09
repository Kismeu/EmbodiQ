# 任务 1.4 目标核对与剩余工作清单（2026-10-03 盘点）

> **这是一份按日期追加的时间线记录,不是现状。**
> 现状看 [`README.md`](README.md)。
>
> 本文件前半部分描述的是 2026-10-03 盘点时发现的问题,其中多数已在后续章节修复。
> 已过期的数字都在原处用 `（2026-10-04 矫正：…）` 就地标注——原值保留以维持记录的
> 时间真实性,括号内为当前值。结构性的推翻见下表:
>
> | 早期结论 | 现状 | 更正在 |
> |---|---|---|
> | **文中任何进度数字**(§2.1、§2.2、§5、§7.5、§8.6 的 22/48、11450 回合等) | 23/48 单元、11500 回合;`openvla/rtn/libero_10` 跑中 | `README.md` §3,或随时重算 `tools/report_matrix.py` |
> | §3.1 峰值显存口径错误,不能进报告 | 已修复并在量化运行上验证 | §7.4、§8.3 |
> | §3.2 建立 `Embodied AI` 符号链接 | 已反转:`Embodied AI` 为真实目录 | §9.1 |
> | §3.3 scope 不对称的处理建议 | §3.3 内已标注 10-03 修正(旧建议作废) | §3.3 |
> | §5 完成度估算(23/48、HSA 0%) | 框架已完整;AWQ/HSA 待上游交付工件 | `README.md` §2 |
> | §6 列出的 1.1–1.5 缺口 | 范围已收回 1.4,这些缺口移出计划 | §7 开头 |


本文件只记录本次直接读取的事实与由此得出的判断，不升级历史文档中的声称为已验证结论。

## 0. 本次核查方式

直接读取：`results/baseline_unified_v3_20260919/full`、`results/stage5_rtn_full`、
`results/stage5_rtn_pilot`、`calib/*/status.json`、`artifacts/awq_*`、
`experiments/quant_stage5|6/src`、`src/embodiq`、`docs/*.md`，以及 `nvidia-smi`、
`ps`、`git` 只读输出。逐回合指标由脚本重算，未采用文档中的旧快照数字。

---

## 1. 目标是否准确

**结论：实验协议（`docs/experiment_protocol.md`）与任务 1.4 的表述一致，总体目标是准确的。** 逐条对照：

| 任务 1.4 要求 | 协议/代码中的落实 | 判定 |
|---|---|---|
| 三个代表性模型（VLA-Adapter / OpenVLA / FastWAM） | `MODEL_CONTRACTS` 三族，三套独立 conda 环境 | 一致 |
| 统一适配模块（图像/语言/状态/动作/时序五类接口） | `core/types.py` 的 `Observation`/`ActionChunk` 契约 + 三个 `backends/*/policy.py` | 基本一致，**时序预测接口有缺口，见 2.3** |
| 四种精度条件复用同一评测入口 | `embodiq.entry.evaluate` 单入口，`--quant` 选择方法 | **仅 fp16/rtn 真正接通，awq/hsa 被 `api.py` 显式拒绝** |
| LIBERO 四集合（空间/物体/目标理解/长时程） | `libero_spatial`/`libero_object`/`libero_goal`/`libero_10` | 一致 |
| 统计成功率、平均成功率、峰值显存、查询延迟 | `metrics.success_rate` / `latency` / `memory` 已逐任务落盘 | 字段齐全，**显存口径有缺陷，见 3.1** |
| 分析性能与资源的平衡关系 | 协议 §8 定义了资源专测方案 | **方案已写、工具未实现** |

需要提请确认的三处目标边界（不是偏离，是口径需要显式写死）：

1. **"量化适配"的模块范围按模型各自定义，不是跨模型统一的模块清单。**
   当前 `run_adapter.py:159` 硬编码：OpenVLA 只量化 `llm`；VLA-Adapter 量化
   `llm + vision_dino + vision_siglip + projector + action_head`；FastWAM 量化
   `video_dit + action_dit`（UMT5 文本编码器 4.63 B 参数保持 BF16）。
   协议 §6 要求的是"每模型一份清单、三种方法共用"，所以现状是**合规**的，
   但 48 单元汇总表必须同时给出 `quantized_param_fraction` 与 `compression_in_scope`，
   否则跨模型的显存数字不可比。实测：OpenVLA 87.4%、FastWAM 55.5% 参数被量化。
2. **"平均成功率"= 四集合等权宏平均**，不是回合加权。已在 `interface_contract.md` 写定。
3. **"查询延迟"只有 `per_env_step_s` 跨模型可比**（查询密度 1 : 1/8 : 1/10）。
   `query_s.mean` 只能同模型内比较。该口径已在代码中区分，报告时不能混用。

---

## 2. 已完成的部分（已核算）

### 2.1 原精度基线：**12/12 单元完整，6000 回合，零错误**

文档 `stage4_unified_baseline.md` / `stage5_rtn_validation.md` 里"OpenVLA Goal 仍在运行、
10/12 完整"的记录**已过期**。实际落盘结果：

| 模型 | Spatial | Object | Goal | Long | 四集合宏平均 |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 88.2% | 89.2% | 78.2% | 52.2% | **76.95%** |
| VLA-Adapter Pro | 98.0% | 99.4% | 97.4% | 95.4% | **97.55%** |
| FastWAM (uncond, first-frame) | 97.0% | 99.2% | 97.4% | 93.6% | **96.80%** |

每单元均为 10 任务 × 50 回合 = 500 回合，`n_error = n_invalid = n_policy_error = 0`，
`contended = false`。配对键齐全：每回合记录 `seed`、`initial_state_id`、
`initial_state_sha256`，已抽样核对 fp16 与 RTN 同一回合的初始状态哈希一致。
**`baseline_summary.md` 需要用 `tools/report_baseline.py` 重新生成。**

### 2.2 RTN：**10.9/12 单元**（2026-10-04 矫正：11/12 完整；`openvla/libero_10` 跑中）

| 模型 | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| OpenVLA | 82.6% (500) | 87.8% (500) | **77.6%（450/500，缺 task 9）** | **缺失（0/500）** |
| OpenVLA（10-04 矫正） | 82.60% | 87.80% | **75.60%（10/10 已补齐）** | 跑中 |
| VLA-Adapter | 98.6% (500) | 95.2% (500) | 87.6% (500) | 93.6% (500) |
| FastWAM | 96.6% (500) | 99.4% (500) | 97.0% (500) | 94.4% (500) |

配对 McNemar（精确二项，同一初始状态）：

| 模型 | 集合 | n | 退化 b | 改善 c | Δ(pp) | p |
|---|---|---:|---:|---:|---:|---:|
| FastWAM | 四集合 | 500×4 | 2–11 | 3–15 | −0.4 ~ +0.8 | 0.56–1.00 |
| OpenVLA | spatial | 500 | 66 | 38 | **−5.60** | **0.0078** |
| OpenVLA | object | 500 | 47 | 40 | −1.40 | 0.52 |
| OpenVLA | goal | 450 | 58 | 53 | −1.11 | 0.70 |
| OpenVLA | goal（10-04 补齐后） | 500 | 74 | 61 | **−2.60** | 0.30 |
| VLA-Adapter | goal | 500 | 59 | 10 | **−9.80** | **<0.0001** |
| VLA-Adapter | object | 500 | 24 | 3 | **−4.20** | **<0.0001** |
| VLA-Adapter | long | 500 | 24 | 15 | −1.80 | 0.20 |
| VLA-Adapter | spatial | 500 | 4 | 7 | +0.60 | 0.55 |

宏平均变化：FastWAM +0.05 pp、OpenVLA −2.70 pp（3 集合）、VLA-Adapter −3.80 pp。（2026-10-04 矫正：OpenVLA 三集合 −3.20 pp）
注意 OpenVLA 的 b/c 同时很大（66/38、47/40、58/53），说明单回合翻转噪声远大于净差值，
这一点必须在分析中说明，不能只报净差。

### 2.3 统一适配层：已验证，但时序预测接口未覆盖

`Observation`/`ActionChunk` 契约 + 构造期校验（相机集合、proprio 维度、
`num_valid ≤ K`）已落地，`32/10` 的 FastWAM 分离语义有 mock rollout 测试覆盖。
**缺口**：任务 1.4 明确把"时序预测"列为需要统一的五类接口之一，而当前
FastWAM 走的是 uncond first-frame 路径，`optional_idm` 权重虽已下载
（`libero_optional_idm_2cam224.pt`）但评测入口无开关，视频 DiT 的未来帧去噪在
主矩阵中从未执行。需要决定是补一个 `--fastwam-mode {first_frame,idm}` 开关
并跑补充行，还是在报告中显式声明主矩阵固定 first-frame、时序接口仅做适配验证。

### 2.4 AWQ：准备链路完成，部署链路未完成

- 校准数据：40 个官方 HDF5（33.78 GB，固定 revision，逐文件 SHA256）已下载完成；
  `calib/observations_training_v1/status.json` 为 `complete`，**1280 校准 + 320 开发观测**
  （40 任务 × 32/8），demo 0–7 / 8–9 不重叠，共 1600 个 NPZ 实际落盘。
- 实现：`experiments/quant_stage6/src/embodiq/quant/` 下有 `calibration.py`（激活采集 +
  有界 reservoir）、`awq_search.py`（逐层输入缩放网格搜索）、`awq_blocks.py`
  （**block 输出 MSE 目标 + 逐组对称裁剪**）、`awq_artifact.py`（fail-closed 工件加载，
  校验权重哈希/scale 形状/裁剪形状）。
- 试验产物：`artifacts/awq_scale_pilot_v1/` 完整（VLA-Adapter 第一个语言块 7 个 Linear，
  alpha 0.20–0.40，开发集重建 MSE 全部低于 RTN 候选）。
- **`artifacts/awq_block_pilot_v1/` 是空目录**：`tools/run_awq_block_pilot.py` 在
  2026-09-20 17:14 创建目录后未产出任何文件，无日志，等于未执行成功。
- **`quant/api.py:32` 仍然 `raise NotImplementedError` 拒绝 awq/hsa**，CLI 的
  `QUANT_PRESETS` 只有 `fp16/rtn/rtn-llm/rtn-all`，FastWAM 更是在 `run_adapter.py:130`
  主动拒绝非 fp16/rtn。所以 AWQ 目前**不存在可用的评测入口**。

### 2.5 HSA：只有数据结构，没有算法

`HSASpec`（granularity/criterion/rho/layer_sens/…）与 `QuantLinear` 的
`residual_idx`/`residual_val` 执行路径存在，`LayerRecord` 预留了
`residual_units/residual_bytes/index_bytes`。但**全仓库没有任何敏感度计算、
残差通道选择、HSA 工件保存/加载的实现**（`grep` 仅命中 spec/qlinear/swap 三个文件的
声明与注释）。协议 §6.1 约定的 `quantize(roots, spec, calibration, context) ->
QuantizationArtifact` 签名尚未存在。HSA 是 48 单元中完全空白的 12 个单元。

---

## 3. 必须先解决的三个阻塞问题

### 3.1 【P0・口径错误】峰值显存在量化后反而变大，当前数字不能进报告

实测（各模型跨集合取最大值，GiB）：

| 模型 | 方法 | weights | peak_alloc | peak_reserved | nvml_peak |
|---|---|---:|---:|---:|---:|
| OpenVLA | fp16 | 14.09 | 14.42 | 14.50 | 15.85 |
| OpenVLA | rtn | **5.23** | **5.57** | 15.84 | **16.88** |
| VLA-Adapter | fp16 | 2.97 | 3.55 | 4.34 | 5.68 |
| VLA-Adapter | rtn | **1.27** | **1.85** | 4.66 | **6.00** |
| FastWAM | fp16 | 23.13 | 23.26 | 23.51 | 24.84 |
| FastWAM | rtn | **15.11** | **15.24** | 31.82 | **33.15** |

`weights_bytes` 与 `peak_alloc_bytes` 下降完全符合预期（2.5x / 1.9x / 1.5x），
但 `peak_reserved` 与 `nvml_peak` 不降反升（FastWAM +8.3 GiB）。原因明确：
量化在**同一进程内、BF16 权重加载之后**进行，caching allocator 的 reserved 段
保留了 BF16 加载期的峰值，NVML 进程峰值同样包含加载+打包瞬时量。
照现状写报告会得出"量化使峰值显存上升"的错误结论。

修复（按成本递增，建议至少做前两项）：
1. 量化完成后 `torch.cuda.empty_cache()` + `reset_peak_memory_stats()`，
   并把"加载期峰值"与"稳态推理峰值"分两列报告（协议 §8 已要求区分，代码未做）。
2. 离线量化：打包一次后 `torch.save` 量化 checkpoint，测量进程直接以 int4 形态加载，
   NVML 峰值才真正代表部署占用。这同时是 AWQ/HSA 工件链路所需，不是额外开销。
3. 固定输入资源专测（协议 §8：40 任务 × 5 观测 = 200 条，预热 20、重复 3 遍），
   独占 GPU，与闭环显存分开报告。

### 3.2 【P0・环境损坏】项目目录被改名，所有绝对路径失效，当前无法运行

`/data/jyc/space/Embodied AI` 已不存在，项目现位于 `/data/jyc/space/Embodied`。
`results/stage5_rtn_full/controller.log` 的末尾异常即由此导致：

```
FileNotFoundError: .../Embodied AI/embodiq/results/stage5_rtn_full/status.tmp
```

控制器 PID 715299 因此死亡，`status.json` 把 OpenVLA Goal 永久留在 `running`，
Goal task 9 在第 6 回合被截断，OpenVLA Long 从未入队——这正好解释 2.2 的两处缺口。
`ps` 确认当前没有任何相关进程存活；工作在 2026-09-20 17:14 之后完全停滞。

仍指向旧路径的文件（6 处代码 + 9 处文档/清单）：
`src/embodiq/entry/run_adapter.py:29`（`EMB_ROOT`）、`src/embodiq/ctl/doctor.py:19`、
`env/common.sh:3`、`env/protect_upstream.sh:10`、`env/download_fastwam.sh:6`、
`tests/test_parity_adapter.py:26`，以及 `experiments/quant_stage5|6` 的同名副本。
`results/baseline_unified_v3_20260919/input_manifest.json` 的 132 条记录中有 8 条
路径含旧名（内容哈希本身仍有效，只是路径需重定位）。

修复方式二选一，必须显式记录：
- **做符号链接** `ln -s Embodied "Embodied AI"`：零代码改动，源码冻结哈希不变，
  已有 11 个 RTN 单元与 12 个基线单元可无争议复用。**推荐**，先用它把缺口跑完。
- **改为 `EMB_ROOT` 环境变量驱动**：更干净，但改动 `run_adapter.py` 会改变
  `embodiq_source_sha256`，必须新建协议版本号并重新冻结，已有结果的复用需要
  走 AST 等价 + 输入清单复核流程（`baseline_review_and_quant_next.md` 里做过一次）。

### 3.3 【P1・公平性决策】三种方法必须共用同一模块范围，现在就要定死

RTN 已经按 2.1 的清单跑完 11 个单元。AWQ/HSA 必须沿用**完全相同**的 scope，
否则 48 单元表就是不同范围最优结果的拼接（协议 §6 明确禁止）。两处需要用户裁决：

- OpenVLA 只量化 `llm`，而 VLA-Adapter 连 `vision_dino/vision_siglip/projector/action_head`
  一起量化——两者是同一套 Prismatic 代码，这个不对称没有技术原因。

  **2026-10-03 修正**：`results/rtn-llm_20260902_2304` 与 `results/rtn-all_20260902_2304`
  （VLA-Adapter、native 协议、schema/1、各 2000 回合）已经是一次 scope 消融，
  宏平均 fp16 97.7% / rtn-llm 96.4% / rtn-all 93.2%。即在 LLM 之外多量化
  vision+projector+action_head 额外付出约 **3.2 pp**。所以这个不对称不是记录问题，
  而是一个会让 VLA-Adapter 的 RTN/AWQ/HSA 三行都背上额外损失、而 OpenVLA 不背的混淆因素。
  原先"倾向 A（保留现状 + 仅声明）"的建议据此作废。

  新建议：主矩阵仍保留按模型定义的清单——VLA-Adapter 的 LLM 只有 357 M/1.47 B 参数，
  llm-only 几乎没有压缩收益，不是可用的主设置——但必须做两件事：
  (a) 48 单元表强制同列 `quantized_param_fraction` 与 `compression_in_scope`，让不对称可见；
  (b) 把 llm-only 变体在 unified.v3 下补成 1.1/1.2 的显式补充行，用数据量化这个混淆因素而非隐藏它。
  附带结论：rtn-all 相对 rtn-llm 的 3.2 pp 正是 HSA 残差补偿要回收的 headroom，
  VLA-Adapter Goal 在 unified.v3 下 −9.8 pp 是全矩阵最大损失单元，应作为 HSA 的首选验证场景。
- FastWAM 的 UMT5 文本编码器（4.63 B，占 RTN 后 15.11 GiB 权重的约 61%）不在 scope 内。
  按 `interface_contract.md`，它在 rollout 中执行 0 次、量化省 VRAM 但省 0 延迟。
  决定是维持排除（并在显存分析中明确指出"峰值由一个推理期不执行的模块主导"），
  还是加一个"文本编码器量化"的补充行单独报告。

---

## 4. 剩余步骤（建议顺序）

### 阶段 S0：恢复可运行状态（约 0.5 天）
1. 建立 `Embodied AI` 符号链接（或完成 `EMB_ROOT` 改造 + 重新冻结），
   以 `tests/` 的无 checkpoint 门禁 + 三环境回归（FastWAM 28、OpenVLA 22、
   stage5 15、stage6 12 项）验证。
2. 复核 `input_manifest.json` 的 132 个输入哈希在新路径下仍然通过。
3. 重新生成 `baseline_summary.md/json`，并把 §2.1/§2.2 的真实进度回写到
   `stage4_unified_baseline.md`、`stage5_rtn_validation.md`
   （这两份文档当前的进度描述已过期，会误导后续判断）。

### 阶段 S1：补齐 RTN 缺口（约 1–2 天机时）
4. 重跑 OpenVLA Goal task 9（现有 450 回合的 9 个任务可复用，需确认续跑工具
   不覆盖已完成结果），再跑 OpenVLA Long 全 10 任务 500 回合。
5. 修复 `run_rtn_validation.py` 的 `atomic_json`：目标目录不存在时报错而不是
   写临时文件失败；控制器异常必须落盘状态而不是静默死亡。
6. RTN 24 单元（fp16 12 + rtn 12）配对汇总 + McNemar，形成第一张可用的半张主表。

### 阶段 S2：显存/延迟口径修复（约 1–2 天）
7. 量化后 `empty_cache` + `reset_peak_memory_stats`，结果 schema 升到
   `embodiq/run_result/3`，新增"加载期峰值 / 稳态推理峰值"两列。
8. 实现离线量化 checkpoint 的保存与加载（`save_quantized` / `load_quantized`，
   带 method/implementation/spec_hash/源 checkpoint 哈希校验）。
   这一项是 AWQ/HSA 工件链路的前置，不是只为显存服务。
9. 实现 `tools/run_resource_bench.py`：协议 §8 的 200 条固定输入、预热 20、
   重复 3 遍、独占 GPU、冷启动与稳态分列。**注意当前 GPU 0/1/3 被他人作业占用
   （12.5/20.7/12.8 GiB、利用率 100%/7%/67%），仅 GPU 2 空闲**，独占窗口要排期。

### 阶段 S3：AWQ 打通到评测入口（约 3–5 天 + 机时）
10. 放开 `quant/api.py` 的方法闸门：改为"必须提供经校验的校准工件才允许 awq/hsa"，
    而不是无条件 `NotImplementedError`；同时保留"无工件不得退回 RTN"的 fail-closed 行为。
11. 把 `awq_blocks`/`awq_search`/`awq_artifact` 从"单块试验"扩展到**全 scope**：
    对 2.1 清单内的每个 block 跑一次缩放搜索（q/k 跳过裁剪，按 `search_clip` 注释要求），
    产出单一 AWQ 工件文件（含 per-layer scale、clip_max、层清单、校准清单哈希、
    源 checkpoint 哈希、`implementation=awq_block_v1`）。
12. 三模型各做一次：工件保存/加载一致性检查 → 固定输入数值检查（与搜索期等价）
    → Spatial task 0 两回合闭环冒烟 → 真实 packed int4 推理确认（不是伪量化）。
13. 跑 AWQ 的 12 个单元 × 500 回合。FastWAM 需要先解除 `run_adapter.py:130` 的硬拒绝，
    并确认其 scope 工件按 `video_dit/action_dit` 别名去重（stage5 已修过一次别名问题）。
14. 把 stage6 的 AWQ 代码合回 `src/embodiq`（主干目前完全没有 `calibration.py`/
    `awq_*.py`），并重新冻结源码哈希。

### 阶段 S4：HSA 实现并接入（约 5–8 天 + 机时，最大未知量）
15. 按协议 §6.1 定义 `quantize(roots, spec, calibration, context) -> QuantizationArtifact`，
    注册 `method=hsa, implementation=local_residual_v1`。
16. 实现缺失的三块算法：层敏感度（`layer_sens=leave_one_in`）、
    残差通道选择（`criterion=hsa`、`granularity=channel`、`rho=0.01`、
    `max_per_layer_rho=0.08`、`align_k=8`）、提取残差后重量化主分支
    （`requantize_after_extraction=True`）。
17. 工件格式与加载校验、`LayerRecord` 的 `residual_units/residual_bytes/index_bytes`
    实际填充，并计入显存与存储预算（协议 §6 要求残差/索引/scale 全部入账）。
18. 验收四项：保存/加载一致性、固定输入输出检查、三模型闭环冒烟、
    真实显存与计时路径检查。然后跑 HSA 的 12 个单元 × 500 回合。
19. 预算匹配说明：HSA 的基础位宽与 RTN/AWQ 相同但总预算更大（残差+索引），
    若要声称"等预算下更优"，必须另做一次预算对齐实验，不能只固定 w4g128。

### 阶段 S5：汇总与分析（约 2–3 天）
20. `tools/report_matrix.py`：输出 48 单元明细 + 12 行配置汇总（四集合成功率、
    宏平均、资源专测显存/延迟），同时保留闭环资源记录，两个来源不得混列。
21. 配对统计：每个 (模型, 集合, 方法) 对 fp16 做 McNemar，任务内回合对 bootstrap，
    报告零差异与样本量不足的限制。
22. 权衡分析：以"宏平均成功率损失（pp）vs 稳态显存降幅 vs `per_env_step_s` 变化"
    三轴呈现。**当前 RTN 数据已足以预判一个关键结论：int4 packed tinygemm 在
    batch=1、短序列下是延迟负收益**——OpenVLA 0.204→0.314 s/query（+54%）、
    FastWAM 0.355→0.398（+12%）、VLA-Adapter 0.067→0.090（+34%）。
    这必须作为真实结果如实报告，并与协议 §8 "编译与低位算子不兼容时单列后端差异、
    不得声称纯算法加速"的要求对齐。
23. 完成条件自检：48 单元无缺失、12 配置均有真实资源数据、可从原始日志重跑复现。
    缺 HSA 或缺任一模型即报告"未完成"，不得用局部实验替代。

---

## 5. 当前完成度估算

| 维度 | 状态 |
|---|---|
| 统一适配层 | 已完成（时序预测开关待定） |
| 统一评测入口 | fp16/rtn 已接通；awq/hsa 入口不存在 |
| 48 成功率单元 | **23 / 48**（fp16 12 + rtn 10.9） |
| 校准数据 | 已完成（1280 + 320 观测） |
| AWQ 算法 | 搜索/裁剪/工件加载已实现；全 scope 部署与评测未做 |
| HSA 算法 | 仅数据结构，核心算法 0% |
| 资源专测 | 方案已写、工具未实现、显存口径有缺陷 |
| 配对统计与汇总工具 | `mcnemar_delta` 存在但无调用方；无矩阵汇总工具 |

最短关键路径：**S0 符号链接 → S1 补 RTN 两个缺口 → S2 显存口径 + 离线量化工件
→ S3 AWQ 全 scope → S4 HSA → S5 汇总**。其中 S4（HSA 算法实现）是唯一需要
从零写算法的阶段，也是进度风险的主要来源。

---

## 6. 置于 1.1–1.5 全局后的修正（2026-10-03 补）

收到完整任务书后重新定位。**1.4 本身的工作与目标一致，但它在全局里被提前执行了**：
1.4 是 1.2（RTN/AWQ）与 1.3（HSA）成果的集成验证口，AWQ/HSA 不存在时 48 单元
不可能闭合。此外有三项"喂给 1.4"的交付物在原 1.4 计划中完全缺失。

### 6.1 交付物映射与完成度

| 条目 | 要求要点 | 状态 |
|---|---|---|
| 1.1 | 模块梳理、统一环境、原精度基线 | 已完成 |
| 1.1 | **模块级前置实验分析量化敏感度** | **缺失**（见 6.2） |
| 1.2 | RTN 实现 | 已完成（W4） |
| 1.2 | AWQ 实现 | 搜索/裁剪/工件加载已实现，全 scope 部署与评测未做 |
| 1.2 | **支持 8 位与 4 位** | **W8 缺失**（见 6.3） |
| 1.2 | **可设置量化范围/分组大小/跳过模块** | 能力在库层存在，CLI 未暴露（见 6.5） |
| 1.3 | HSA 算法（AWQ 主干 + 高精度残差） | 仅数据结构，核心算法 0%；**硬依赖 AWQ 先完成** |
| 1.3 | **残差选择策略消融** | **缺失**（`HSASpec.criterion` 五个取值未实现亦未实验） |
| 1.4 | 统一适配模块 + 相同评测入口 | 适配层完成；awq/hsa 入口不存在 |
| 1.4 | 48 成功率单元 | 23/48 |
| 1.4 | 四项指标统计 | 字段齐全，显存口径有缺陷（§3.1） |
| 1.5 | 综合评价体系 | 指标定义完成，资源专测工具未实现 |
| 1.5 | **统一配置机制（六个维度）** | **缺失**，当前为硬编码 preset |
| 1.5 | 工具链（适配/校准/量化/残差/评测/汇总） | 前三项部分具备，残差与汇总缺失 |
| 1.5 | **年度版本固化：配置文件/部署说明/使用手册/测试报告** | **全部缺失**，`docs/` 现为研究日志而非手册 |

### 6.2 1.1 的模块敏感度实验：缺失，但代价很低

该实验本应前置于 scope 选择，实际 scope 由代码审读决定（`walker.py` 的
`HARD_SKIP` 理由写在注释里）。补做的成本远低于预期，因为三件前置条件已就绪：

- `core/fidelity.py`（`ActionFidelity` + `headroom_verdict`）已实现**逐步动作保真度**
  指标，并在文件头给出功效模拟：100 回合的成功率门禁在真实 3 pp 下降下约 90% 概率
  误判为"无差异"，而同一批数据在 step 级有 400 倍信息量，且**无需 rollout**——
  在同一批留出观测上跑 FP 与量化版本比较动作即可。
- 320 条开发集观测（40 任务 × 8）已渲染落盘，与校准集 demo 编号不重叠。
- `results/rtn-{llm,all}_20260902_2304` 已提供一次真实 scope 消融（见 §3.3 修正）。

建议设计：对每个候选模块/scope 做"量化 → 320 条开发观测 → 与 FP 动作块比较
（L1 / 余弦 / 逐维偏差）"，只对保真度判定为 FAIL 的候选用 `protocol.screening`
（10 回合/任务）做闭环确认。这正是 `fidelity.py` 文件头主张的"便宜否决 + 昂贵确认"
方向，逻辑方向是成立的（动作接近 ⇒ 成功率接近，反向不成立）。

### 6.3 1.2 的 8 位量化：当前不存在，不能用 int8 回退冒充

`pack.py` 只有 `quantize_int8_per_channel`，用途是**不可分组层的回退**
（`QuantSpec.fallback="int8_per_channel"`），且 tinygemm 打包核只有 int4 路径。
所以"支持 8 位权重量化"需要真实工作：分组 int8 路径 + 对应算子（或明确以
`Int8Linear` 的反量化浮点计算路径交付并如实标识实现类别）。协议 §2 已把额外位宽
划为补充实验，与 1.2 的"支持"是可以对齐的，但必须实现且有结果，不能缺席。
注意 `SimQuantLinear` 是伪量化（无显存收益），协议 §6 要求区分三类实现，
不得用它充当部署指标。

### 6.4 1.3 的硬依赖

1.3 原文是"在 **AWQ 低比特主干基础上**研发 HSA"。这不是排期偏好而是定义：
HSA 主分支必须是 AWQ 量化的同一对象，`HSASpec` 的注释也印证了这一点
（`residual_impl="gather"` 的理由正是"预排列输入通道会移动 g=128 分组边界，
使 HSA 主分支不再与被比较的 AWQ 行是同一对象"）。因此 S3（AWQ 全 scope）
必须先于 S4（HSA）完成，中间不能并行抢跑。

好消息是最后一公里的接线很窄：`swap.apply_report` 已接受 `input_scales` 与
`residuals` 两个入参并回填 `residual_units/residual_bytes/index_bytes`，
`QuantLinear.from_linear` 已支持 `input_scale` 与 `residual_idx/val` 共存。
缺的是：(a) 产出 residuals 的选择算法，(b) `make_quant_apply` 目前调用
`apply_report` 时未透传这两个入参。

### 6.5 1.5 的统一配置机制：应在 AWQ/HSA 开跑之前完成

1.5 要求六个维度可配：模型类型、量化方法、量化位宽、分组大小、量化范围、
HSA 残差预算。现状：`run_adapter.py:38` 的 `QUANT_PRESETS` 是字面量，
`:159` 按模型硬编码 scope 字典，group_size 固定 128，无 `--w-bit`、
无 `--scopes`、无 `--skip`、无 HSA 残差预算参数。

这项必须排在 S3 之前：AWQ 12 单元 + HSA 12 单元 + 消融若先跑后改配置，
所有结果的 `quant_meta` 都要回填重写。改在前面反而更省。

### 6.6 可视化页面：不是交付物

1.5 的年度固化清单是**源代码、配置文件、部署说明、使用手册、测试报告**，
"结果汇总"是工具链的一个功能项。1.1–1.5 全文没有平台、界面或可视化要求。
对应产物是 `tools/report_matrix.py`（48 单元表 + 12 行汇总 + 三轴权衡图）
与上述文档，不是 Web 页面。若后续需要汇报用页面，应是同一份 JSON 的静态渲染，
排在最后且不得成为第二数据源。

---

## 7. 范围决定与执行记录（2026-10-03）

**范围锁定为 1.4。** 1.1 的模块敏感度实验、1.2 的 8 位量化、1.3 的残差策略消融、
1.5 的六维配置机制/使用手册/年度固化/端侧硬件，全部移出本项目计划。
AWQ（1.2）与 HSA（1.3）的算法由对应子任务提供，**1.4 只负责接入与评测**。
这把 1.4 的交付物固定为四件：统一适配模块、四种方法复用同一评测入口、
LIBERO 四集合验证、四项指标统计与权衡分析。

可视化页面不在 1.4 交付物内；底层框架完成后再考虑，且必须是 `matrix.json`
的只读渲染，不得成为第二数据源。

### 7.1 S0 恢复可运行（完成）

- 建立 `/data/jyc/space/Embodied AI -> Embodied` 符号链接。6 处硬编码绝对路径
  无需改动，源码冻结哈希不变，已完成单元可无争议复用。
- 三棵源码树测试通过：主干 32、quant_stage5 37、quant_stage6 49（跳过项需 checkpoint/CUDA）。
- **原精度基线逐回合核验通过**：`tools/report_baseline.py` 全部断言通过，
  12/12 集合、6000/6000 回合，每回合 `seed` 与 `initial_state_sha256` 与
  `episode_manifest.json` 逐项吻合。这是配对 McNemar 成立的前提。
- **输入清单 132 个文件 SHA256 全量重算零不匹配**（91.6 GiB，1085 s）。
- `stage4_unified_baseline.md`、`stage5_rtn_validation.md` 的过期进度已更正
  （保留历史记录，追加核验节）。

### 7.2 主干代码合并（完成）

主干 `src/` 原本是三棵树里最旧的一棵，缺三项且其中两项影响正确性：

| 缺失 | 影响 |
|---|---|
| `walker.py` 的 FastWAM 别名规则（stage5 修复） | 按旧规则 `mot.mixtures.*` 匹配，FastWAM 会选错模块 |
| `api.py` 的方法闸门 | **`--quant awq` 会静默执行 RTN** |
| `qlinear.py` 的 `clip_max` | AWQ 逐组截断无法执行 |

已将 `quant_stage6` 的 `api/qlinear/walker/awq_artifact/awq_blocks/awq_search/calibration`
与 `entry/run_adapter.py` 合入主干，主干现与 stage6 逐字节一致后再叠加下述改动。
`run_adapter.py` 的源码哈希改为相对 `__file__` 计算，顺带去掉一处旧路径依赖。

### 7.3 统一评测入口：四种方法接通（完成）

这是 1.4 的核心条文。新增 `src/embodiq/quant/artifact.py` 作为
**1.2/1.3 与评测方之间的契约surface**，接入文档见
`docs/quant_artifact_contract.md`，契约测试 `tests/test_quant_artifact.py` 33 项通过。

设计为 fail-closed，四类错误各有对应校验，全部在替换任何模块**之前**完成：

| 风险 | 校验 |
|---|---|
| 换标签的 RTN | awq 工件必须有 `input_scale`；hsa 工件必须有残差；awq 带残差即拒绝 |
| 模块范围不一致 | 工件层集合必须**恰好等于**该模型计划的 int4 层集合 |
| 换了 checkpoint | 每层 `weight`/`bias` SHA256 与实际加载张量比对 |
| 半量化 | 校验全通过后才构造替换；拒绝时模型保持原样（有测试） |

配套改动：

- `run_adapter.py`：`MODEL_SCOPES` 常量把每模型范围收到一处，rtn/awq/hsa 共用，
  使"按方法分叉范围"在结构上不可能；新增 `--quant-artifact`、`--hsa-rho`；
  参数校验提前到解析期（错误调用不再付出加载 20 GiB checkpoint 的代价）；
  解除 FastWAM 对非 fp16/rtn 的硬拒绝——真正的守卫是工件自身的模型感知范围检查。
- `swap.py`：`apply_report` 新增 `clips` 入参并透传 `clip_max`；
  带残差的层改用 `INT4_RESIDUAL_REL_ERR_BAND`（HSA 精确恢复部分通道，
  重建误差按设计低于 RTN band，下界会误伤）。
- `api.py`：`make_quant_apply(spec, scopes, artifact=...)`，无工件的 awq/hsa
  直接报错，不退回 RTN；工件元数据经 `report.artifact` 进入结果 JSON。

CLI 实测行为：`fp16`/`rtn` 照常；`awq`/`hsa` 无工件报错；`rtn` 给工件报错；
FastWAM 可走 `awq`/`hsa`。

### 7.4 峰值显存口径修复（完成）

根因：量化在同进程 BF16 权重加载**之后**进行，`reset_peak_memory_stats` 清零
计数器但不归还内存池，NVML 读的是进程真实占用，于是下一次采样立刻把加载期
footprint 读回来。实测后果是 RTN 的进程峰值反升（FastWAM 24.84 → 33.15 GiB），
而稳态分配其实正常下降（23.26 → 15.24 GiB）。

修复在 `MemoryProbe.mark_after_load()` 单点完成：先记录加载期峰值，
`gc.collect()` + `empty_cache()` 把池归还驱动，再重置峰值计数开始稳态测量。
`MemoryReport` 新增 `load_peak_alloc/reserved/nvml_peak_bytes` 与
`cache_released_bytes`——加载期峰值本身是协议 §8 要求报告的真实数字，不是丢弃项。
结果 schema 升至 `embodiq/run_result/3`，历史 schema/2 文件不改写。

### 7.5 结果汇总工具（完成）

`tools/report_matrix.py` → `results/matrix/matrix.json` + `matrix.md`，
是下游（报告表、权衡图、未来的可视化页面）的**唯一数据源**，下游不重算。
强制项：单元 `complete` 需 task 0..9 + 500 回合 + 零错误；宏平均仅在四集合
全完整时给出；配对差值走 `core.protocol.mcnemar_delta`；显存口径按每个单元
自己的 schema 版本给出对应说明（当前 23 个单元为 schema/2，其峰值不可用于
量化前后对比）。

当前状态：**22/48 完整单元，11450/24000 回合，零完整性问题**。（2026-10-04 矫正：23/48、11500/24000；权威值随时重算 `tools/report_matrix.py`）
（22 而非 23：`openvla/rtn/libero_goal` 为 450/500。**2026-10-04 矫正**：该集合已补齐 10/10，这个差异不再存在；现为 23/48。）

### 7.6 剩余

| 步骤 | 内容 | 依赖 |
|---|---|---|
| S1 | 补 OpenVLA RTN 的 Goal task 9 与 Long 全 10 任务 → 24/48 | GPU，当前仅 GPU 2 空闲 |
| S1b | 修 `run_rtn_validation.py` 的 `atomic_json`：控制器异常须落盘状态 | 无 |
| S2b | `tools/run_resource_bench.py`：协议 §8 的 200 条固定输入、预热 20、重复 3 遍、独占 GPU | GPU 独占窗口 |
| S4 | 收到 1.2/1.3 工件后跑 AWQ/HSA 各 12 单元 | 等上游 |
| S5 | 三轴权衡分析与图 | S1/S2b/S4 |
| — | FastWAM `--fastwam-mode` 开关（1.4 的"时序预测接口"，仅接口验证） | 无 |

**未提交风险**：工作区有 40+ 个未跟踪/修改文件，最后一次提交是 2026-09-02 的
`9846c1a`。结果 JSON 记了 `embodiq_source_sha256`，但那份代码无法从 git 恢复，
协议 §9 的"能够重跑并从原始日志生成报告"按现状不满足。建议在 S1 之前提交并打 tag。

---

## 8. 统一评测框架：流程端到端验证通过（2026-10-03）

本节记录框架设计的**实跑验证**，不是设计意图。每一项都有可复现的命令与产物。

### 8.1 验证用工件（明确不是方法结果）

AWQ/HSA 的算法由 1.2/1.3 提供，尚未交付。为验证通路，`tools/make_validation_artifact.py`
产出**契约合规但不是方法结果**的工件：输入缩放仅由权重幅度导出
（无激活采集、无缩放搜索），残差按权重幅度取 top-k
（对应 `HSASpec.criterion="magnitude"` 消融臂，无激活敏感度、无层级敏感度排序）。

每个工件携带 `validation_only: True`、`implementation="pipeline_validation_{awq,hsa}_v0"`
与 `not_a_method_result` 说明；运行结果写入 `results/validation_{awq,hsa}/`，
**不在 `report_matrix.py` 读取的 stage 目录内**。由此得到的任何数字都是关于管道的，
不是关于量化方法的。

残差语义是正确实现的：残差位于**缩放后权重域**
（`dequantized()` 先 `w[:, idx] += res_val` 再乘 `s_inv`），
故所选通道精确恢复，HSA 执行路径是真被走通而非仅被填充。

VLA-Adapter Spatial 工件：568 层（与 RTN 的 int4 层数吻合），
HSA 版 40.0 MiB / 6352 个残差通道，AWQ 版 3.0 MiB。

### 8.2 端到端跑通结果

```
python -m embodiq.entry.evaluate --model vla_adapter \
  --checkpoint /data/jyc/data/VLA-Adapter/LIBERO-Spatial-Pro \
  --suite libero_spatial --protocol unified.v3 \
  --quant hsa --quant-artifact artifacts/validation/hsa_vla_adapter_spatial.pt \
  --hsa-rho 0.01 --trials 2 --tasks 0 --gpu 3 \
  --out results/validation_hsa/vla_adapter
```

| 环节 | 结果 |
|---|---|
| 工件加载与全量校验 | 568 层、568 输入缩放、568 残差，对真实 checkpoint 逐层权重哈希通过 |
| 量化应用 | 568 int4 + 28 int8 = **596 模块替换**，`spec_id=hsa-w4g128-rho1`，整模型 2.64x |
| swap 重建校验 | 通过（含残差层改用 `INT4_RESIDUAL_REL_ERR_BAND`） |
| 闭环 rollout | **2/2 成功，零错误** |
| 结果 JSON | `schema/3`，带 `quant.artifact` 全套可追溯字段 |
| AWQ 分支 | 568 缩放 / 0 残差 / 开销 1.4 MiB，**2/2 成功** |
| 负向校验（真实通路） | `--quant hsa` 配 AWQ 工件 → `ArtifactError: artifact method 'awq' != requested 'hsa'`，退出码 1 |

### 8.3 显存口径修复在量化运行上的验证

这是 §7.4 那项修复的实证。HSA 运行（VLA-Adapter Spatial）：

| | 加载期 | 稳态 |
|---|---:|---:|
| `peak_alloc` | 3.98 GiB | **1.86 GiB** |
| `peak_reserved` | 4.64 GiB | 2.66 GiB |
| `nvml_peak` | 5.12 GiB | **3.70 GiB** |

`cache_released_bytes = 2.60 GiB`。对比：fp16 闭环 `nvml_peak` 5.68 GiB，
修复前 RTN 读出 6.00 GiB（上升）。修复后量化运行为 **3.70 GiB（下降 35%）**。
加载期峰值受原精度约束这一事实被如实分列，而非丢弃。

方法开销如实入账：HSA 残差 18.3 MiB + 索引 0.02 MiB + 缩放 1.38 MiB = 19.7 MiB；
AWQ 仅缩放 1.4 MiB。"额外显存有界"因此是可证伪的。

### 8.4 资源专测端到端跑通

`tools/run_resource_bench.py --make-inputs` → `results/resource_bench/inputs_v1.json`：
**200 条固定输入**（40 任务 × 5），逐条 SHA256，清单哈希 `38f9ecc516f3`，
按文件名排序选取、无随机性。

VLA-Adapter fp16 实跑：四集合各 50 输入、20 预热 + 3 遍测量 = 每集合 150 次、
共 600 次测量。与闭环互相印证（`weights` 2.97 GiB、稳态 `peak_alloc` 3.55 GiB 完全一致），
`nvml` 4.12 低于闭环 5.68，因本工具进程内无 MuJoCo 渲染，是更干净的策略侧占用。

冷启动分离的必要性由数据给出：**首次查询 1830 ms，稳态 65 ms，相差 26 倍**。

占用守卫实测：对被他人作业占用的 GPU 直接拒绝并点名 PID 与占用量，
退出码 2；`--allow-contention` 可做诊断运行并在结果中标记。

输入来源的限制如实随结果传递：观测重渲染自训练演示，
按构造均为成功轨迹，不满足协议 §8"不得只挑成功轨迹"。
本工具只测资源，而三模型的单次查询代价按构造与输入内容无关
（OpenVLA 固定 7 token、VLA-Adapter 单次前向、FastWAM 固定 10 步 Euler），
故对显存/延迟成立；但不得用于任何成功率结论。

### 8.5 端到端跑出的两个真实缺陷

CPU 契约测试抓不到它们——`QuantLinear.from_linear` 需要 CUDA tinygemm。

1. **`qlinear.py` 设备不匹配**：工件张量来自 `torch.load(map_location="cpu")`，
   权重已在 GPU，而 `input_scale` 在使用处只做了 `.to(torch.float32)` 未改设备
   → `RuntimeError: ... cuda:0 and cpu`。已改为在 `from_linear` 顶部一次性把
   `input_scale`/`clip_max`/`residual_idx`/`residual_val` 统一搬到权重所在设备。
2. **`run_adapter.py` 源码哈希仍依赖旧绝对路径**：`path.relative_to(EMB_ROOT)`
   而 `rglob`/`resolve()` 返回符号链接解析后的真实路径，
   **符号链接对字符串路径比较不起作用**。已改为 `EMB_ROOT` 由 `__file__` 推导、
   哈希键相对 `src/embodiq`，并新增 `embodiq_source_hash_basis` 字段——
   2026-10-03 之前的结果以不同基准计算，不可与新值直接比较。

### 8.6 汇总工具验证

`report_matrix.py --source hsa=... --source awq=...` 四方法同表渲染通过：
AWQ/HSA 单元显示为 `100.00%*(2/500)`（`*` 标明为进度非结果）、无宏平均、
显存读修复后数值、并在表头输出"非标准来源……不得作为任务结果引用"警示。
完整单元计数不受影响（仍 22/48（2026-10-04 矫正：现 23/48）），缺失 task id 逐一列出。

正式矩阵（无 override）：**22/48 完整单元，11450/24000 回合，零完整性问题**。（2026-10-04 矫正：23/48、11500/24000）

### 8.7 框架状态

设计与流程已验证跑通的部分：

- 统一适配模块（三模型五类接口，含 FastWAM 时序预测开关）
- 统一评测入口（四方法共用，AWQ/HSA 经 fail-closed 工件契约接入，正负向均实测）
- 四项指标（成功率 / 宏平均 / 峰值显存 / 查询延迟），显存口径已修正并验证
- 固定输入资源专测（200 输入、冷启动分离、占用守卫）
- 结果汇总（单一数据源、未完成不伪装、配对 McNemar）
- 控制器durability（原子状态写、崩溃留痕、按 task 续跑）

仍待外部输入或 GPU 窗口：

| 项 | 阻塞于 |
|---|---|
| OpenVLA RTN 的 Goal task 9 与 Long 全 10 任务 | GPU 窗口 |
| AWQ/HSA 的 24 个正式单元 | 1.2/1.3 交付工件 |
| 三模型四方法的资源专测（12 配置） | GPU 独占窗口 |
| 三轴权衡分析与图 | 以上数据 |

---

## 9. 问题修复与 RTN 补跑启动（2026-10-04）

### 9.1 路径:符号链接反转,而不是改冻结树

stage5 的 `run_adapter.py:200` 与主干同源地用 `path.relative_to(EMB_ROOT)` 配一个
旧路径字面量,而 `rglob`/`resolve()` 返回符号链接解析后的真实路径——
**符号链接对字符串路径比较不起作用**,补跑会在加载模型后崩。

关键约束:已完成的 11 个 RTN 单元记录 `embodiq_source_sha256 = 47043e48…`,
而**改动 stage5 的任何一个字节都会改变这个值**,使第 12 个单元与其余 11 个
不可比(协议 §9)。因此没有修改 stage5,而是反转符号链接:

```
/data/jyc/space/Embodied AI   （真实目录,恢复为冻结时的规范路径）
/data/jyc/space/Embodied -> Embodied AI   （符号链接,新名字继续可用）
```

验证:stage5 的 run_adapter 口径重算得 `47043e48…`,与 11 个已完成单元
**逐字节一致**。两个路径均可用,操作完全可逆。

### 9.2 冻结检查:按实际执行的代码分范围

控制器随后停在 `audited_source()` 的"源码漂移"检查上。该检查把
`embodiq/src` 与三个模型仓库算进同一个摘要,但 **RTN 运行从
`experiments/quant_stage5/src` 加载,根本不 import `src/`**——我对主干的修改
触发了一个按构造就是误报的检查。

改为分范围,并用模型仓库自己的 git 做独立且更强的验证:

- `audited_source(scope="full")`:原行为不变(fp16 续跑工具仍用它)。
- `audited_source(scope="native")`:组合摘要不符时，不是直接停，而是用 git 证明
  原生代码未漂移——四个仓库均在冻结 HEAD、除基线冻结时即已存在的两个 OpenVLA
  文件外无改动；新增 `native_digest()` 仅覆盖模型仓库，固定为
  `c6e5a306…`，此后可独立检测原生漂移。
- `main_src_drift: True` 与两个摘要值一并写入状态,**如实记录而非隐藏**。

实测:四仓库 head 全部匹配冻结版本、未审计漂移为空;`scope="full"` 仍正确拒绝。

### 9.3 控制器:可比性实测、GPU 停驻、单模型补跑

- **可比性由实测决定**:控制器读取已完成单元记录的 `embodiq_source_sha256`,
  与当前隔离树指纹比对,不一致直接拒绝启动。原先此处是硬编码断言。
- **GPU 停驻**:新增 `--min-free-gib`(默认 20)与 `--gpu-wait-hours`,
  用 `nvidia-smi` 查空闲(不建 CUDA 上下文),不足则每 2 分钟重试。
  协议 D6 要求"运行前检查、尊重现有作业"——不抢占他人作业。
- **单模型补跑**:新增 `--models` 与 `--gpu` 覆盖,只跑 openvla。

### 9.4 其余路径修复

| 位置 | 问题 | 修复 |
|---|---|---|
| `run_adapter.py` `CKPT_ROOT` | 指向从未存在的 `EMB_ROOT/checkpoints/vla-adapter`,默认值永远失败 | 改为 `/data/jyc/data/VLA-Adapter`,可用 `EMBODIQ_CKPT_ROOT` 覆盖 |
| `ctl/doctor.py` | `EMB_ROOT` 旧字面量默认值 | 由 `__file__` 推导,保留 env 覆盖 |
| `tests/test_parity_adapter.py` | checkpoint 路径指向不存在的目录 | 指向真实位置;修好后跳过项由 12 降至 9 |

### 9.5 补跑状态

`tools/run_rtn_validation.py --background --resume --models openvla --min-free-gib 20`
PID 3335886。

- spatial / object 标记 `reused`,**未重跑那 1000 个回合**
- GPU 0 空出 28.6 GiB 后自动启动
- `libero_goal` 只跑缺失的 `task 9`;之后 `libero_10` 全 10 任务
- 旧状态保留为 `status.superseded.20261004_124106.json`

按历史实测耗时估算(fp16 libero_10 为 91 min/task,RTN/fp16 延迟比 1.52,
RTN goal 实测 51 min/task):goal task 9 约 51 min + libero_10 约 23 h ≈ **24 小时**。（2026-10-04 矫正：该估算基于单卡；goal 已完成，libero_10 改为四卡 8 进程，实测 6.2 h，见 §10.5）

完成后 RTN 列闭合,矩阵达 **24/48**。

### 9.6 回归

主干 **104 passed / 9 skipped**(含契约 33、控制器 16、FastWAM 时序 9)。（2026-10-04 矫正：110 passed / 9 skipped；控制器 22）

---

## 10. 断点续跑、干净停止与四卡并行（2026-10-04）

### 10.1 续跑粒度:任务级,不是回合级

实测确认:`run_adapter.py` 在每个任务开始前检查
`gpu{N}_task{T}_results.json`,存在则用 `core/resume.py` 的 `validate_resume`
校验配置(policy/quant/protocol/git/versions/回合数/无错误)一致后跳过。
但 `TaskResult.write()` **每个任务只在全部回合结束时写一次**。

因此:

- **已完成的任务永不重跑**,再次 `--resume` 会跳过。
- **停止代价 = 进行中的那一个任务**。openvla libero_10 约 137 min/任务,
  libero_goal 约 51 min/任务。

回合级续跑无法用于当前 RTN 补跑:它从 `experiments/quant_stage5/src` 加载,
改主干 `src/` 对它无效,改 stage5 又会破坏 `47043e48` 的可比性(§9.1)。
该能力只能加给后续从主干运行的 AWQ/HSA。

### 10.2 干净停止

旧控制器没有信号处理器:SIGTERM 不抛异常,所以 `except BaseException` 路径
不会执行,状态冻结在 `running`、子进程继续占着 17 GiB 显存。
2026-10-04 停止旧控制器时实测复现了这一点(`phase: running`、需手动 kill 子进程)。

`tools/run_rtn_validation.py` 不在任何冻结哈希内(哈希只覆盖 stage5 的
`embodiq` 包),故可安全加入:

- `install_stop_handlers`:SIGTERM/SIGINT → 终止全部在跑 worker
  (先 terminate、60 s 后 kill)→ 写入 `phase=stopped`、`stopped_by`、
  `stopped_at`,并把在跑的 job 标为 `stopped_while_running`;重复信号被忽略。
- `stop_note` 明确写出续跑粒度是任务级,让停止代价不必靠猜。
- GPU 停驻循环改用 `STOPPING.wait(120)`,停止请求立即生效而非等满 2 分钟。
- `--stop`:向状态文件记录的 PID 发 SIGTERM 并等待确认,无需手工找 PID。
- `--status`:打印各单元 complete/partial/empty、续跑会跑哪些 task、控制器是否存活。

5 项针对停止逻辑的测试(状态记录、子进程终止、顽固子进程被 kill、
重复信号幂等、stop_note 口径),控制器测试合计 21 项（2026-10-04 矫正：22 项，§10.6 的停止/崩溃区分补了 1 项）。

### 10.3 四卡按任务分片

单模型补跑原先只用 1 卡(控制器按模型并行)。新增 `--gpus openvla=0,1,2,3`
按任务分片:

```
gpu0 <- tasks [0, 4, 8]    gpu2 <- tasks [2, 6]
gpu1 <- tasks [1, 5, 9]    gpu3 <- tasks [3, 7]
```

正确性依据:任务之间完全独立——各自的环境、各自的 50 个初始状态、
种子由 `(task_id, episode_idx)` 推导,所以**哪张卡跑哪个任务不影响结果**。
一卡一 worker 使每个进程独占设备,与已完成单元的测量条件相同
(此前三模型也是一卡一进程),故逐任务的延迟与显存数字仍可比。
结果文件名含 gpu id,`cell_tasks` 按 JSON 里的 `task_id` 判定,混合前缀无妨。

容量检查对分片用到的每张卡逐一执行,不足则整体停驻。

### 10.4 当前状态

- `libero_goal` 已补齐 **10/10**(task 9 于四卡启动前落盘)
- `libero_10` 四卡并行中,PID 3459114,四个 worker 各约 17 GiB
- 最长分片 3 个任务 × 约 137 min ≈ **约 7 小时**(原单卡约 23 h)
- 完成后 RTN 列闭合,矩阵达 **24/48**

停止:`python tools/run_rtn_validation.py --stop`
查看:`python tools/run_rtn_validation.py --status`
续跑:`python tools/run_rtn_validation.py --background --resume --models openvla --gpus "openvla=0,1,2,3"`

### 10.5 每卡两进程:实测收益与代价

GPU 利用率在一卡一进程下均值约 65%(空档来自 MuJoCo 渲染与环境步进在 CPU),
故加到每卡 2 进程(共 8 worker)。资源核对:96 核(负载 14.6,每 worker 约 2.3 核)、
376 GB 内存(每 worker 约 4 GB)、每卡 34.8/46 GiB。利用率随后升至 94–100%。

**实测吞吐,不是估计:**

| 配置 | 总吞吐 | 单 worker | 关键路径 | 墙钟 |
|---|---:|---:|---|---:|
| 4 worker(一卡一进程) | 1.29 回合/min | 3.10 min/回合 | 3 任务 = 150 回合 | 7.8 h |
| 8 worker(一卡两进程) | 1.70 回合/min | 4.71 min/回合 | 2 任务 = 100 回合 | **6.2 h** |

吞吐只提升 **1.32×**,不是 2×——单 worker 速度降到 1.52 倍耗时,说明 GPU 的
实际饱和度高于 65% 那个读数。墙钟收益主要来自关键路径从 3 个任务缩到 2 个
(10 个原子任务分到 8 片:两片 2 任务、六片 1 任务)。
6 个单任务分片约 3.7 h 后陆续结束,其后 s0/s1 各自独占一卡加速收尾。

受显存限制无法再细分:10 片需要某些卡上 3 进程 = 52 GiB > 46 GiB。

**代价(已接受并记录):** `libero_10` 全部单元被标 `contended=true`
(日志实测:`GPU shared with 1 other process(es) using 15.81 GiB --
resource measurements are not headline results`),其**闭环**延迟/显存不可与
其余 11 个 RTN 单元(一卡一进程)比较。可接受的理由:

1. 主表资源数字来自 `run_resource_bench.py` 的独占 GPU 专测,已覆盖四集合,
   闭环资源记录为次要(协议 §9 要求保留但不混用两个来源)。
2. **成功率完全不受影响**:rollout 在给定 seed 与初始状态下确定,
   步数上限是步数而非秒数,变慢不会多出超时。

### 10.6 分片引入的两个缺陷(已修)

1. **日志路径冲突**:同卡两个分片的日志名只含 GPU 号,两个进程以 `mode="w"`
   打开同一路径,在各自偏移写入 → 两份日志均不可用。结果 JSON 不受影响
   (按 task 分文件)。已在日志名加入分片序号 `_s{index}`,状态记录每片的日志名。
2. **`entry_log` 先用后赋值**:`UnboundLocalError`,只起了 1 个进程便崩溃,
   并留下一个孤儿子进程。已修正定义顺序,并在校验中断言定义早于使用。

两者都是加分片当天实测暴露的,各损失约 10 分钟。

---

## 11. RTN 列闭合（2026-10-04）

`openvla/rtn/libero_10` 完成，控制器 `phase=complete`，四集合全部 10/10。
矩阵达 **24/48 完整单元、12000/24000 回合**。

### 11.1 逐回合核验

| 检查 | 结果 |
|---|---|
| 任务文件 / task_id 覆盖 | 10 个 / 0..9 完整 |
| 回合数 | 500/500，`episode_idx` 完整覆盖 0..49 |
| 错误 | `n_error=0`、`n_invalid=0`、`n_policy_error=0` |
| 逐回合 `seed` | 与 `episode_manifest.json` 全部吻合 |
| 逐回合 `initial_state_sha256` | 与 `episode_manifest.json` 全部吻合 |
| `successes` 与逐回合状态 | 一致 |
| 分片 | GPU 0/1/2/3，`contended=true`，`schema/2` |

### 11.2 可比性已达成

**12 个 RTN 单元（120 个任务文件）的 `embodiq_source_sha256` 全部为
`47043e48e17a246d…`**，与隔离树 `experiments/quant_stage5/src` 的指纹一致。

这正是 §9.1 恢复规范路径、而不是修改冻结树的目的：改动 stage5 的任何一个字节
都会改变该哈希，使补跑的单元与先前完成的 11 个不可比（协议 §9）。现在整列
逐字节同源。同理整列均为 `schema/2`——stage5 不含显存修复，口径一致优先于
单元格正确，`report_matrix.py` 按 schema 输出对应说明。

### 11.3 OpenVLA RTN 四集合结果

| 集合 | 原精度 | RTN | n | 退化 | 改善 | Δ pp | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spatial | 88.20% | 82.60% | 500 | 66 | 38 | **−5.60** | **0.0078** |
| Object | 89.20% | 87.80% | 500 | 47 | 40 | −1.40 | 0.52 |
| Goal | 78.20% | 75.60% | 500 | 74 | 61 | −2.60 | 0.30 |
| Long | 52.20% | 49.80% | 500 | 109 | 97 | −2.40 | 0.44 |
| **宏平均** | **76.95%** | **73.95%** | | | | **−3.00** | |

**四集合中只有 Spatial 达到显著。** 其余三集合的不一致对数量远大于净差：
Long 的 109/97 意味着 500 个回合里 41% 向某一方向翻转，而净差只有 −2.40 pp。
这与 §2.2 的观察一致，并且在补齐后更极端。**只报净差会严重夸大结论强度**，
分析中必须同时给出不一致对。

另一处值得记录：Goal 在 9 任务部分值下是 77.56%（Δ −1.11 pp），补齐第 10 个
任务后变为 75.60%（Δ −2.60 pp）。部分单元的成功率系统性偏乐观，这正是
`report_matrix.py` 给未完整单元加 `*` 标记、并拒绝对未完整集合计算宏平均的理由。

### 11.4 下一步

四卡已全部空闲（各 18 MiB）。框架侧最后一项需要 GPU 的工作是 11 个资源专测
配置（`tools/run_resource_bench.py`，每配置约 10 min × 4 checkpoint，需独占）。
AWQ/HSA 的 24 单元仍等 1.2/1.3 交付工件；接入文档与
`artifacts/scope_manifest.json` 的精确层名清单均已就绪。
