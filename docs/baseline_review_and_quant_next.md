# 基线结果复核与下一步

## 2026-10-04 更正（本节为最新状态，下文保留历史记录）

下文的进度与"下一顺序"均已过期:

- **原精度基线 12/12 集合、6000/6000 回合全部完成并逐回合核验**
  (下文称"1000/6000 回合、2/12 单元")。
- **RTN 11/12 集合完成**,`openvla/libero_10` 跑中。
- 下文"量化下一阶段的代码盘点"所列缺口已处理:方法闸门已实现为工件门控
  (`quant/artifact.py`),`make_quant_apply` 已透传 `input_scales`/`residuals`,
  FastWAM 的量化 CLI 硬拒绝已解除(真正的守卫是工件自身的模型感知范围检查)。
- 下文"下一顺序"中的"接入真实 AWQ 校准""接入可替换 HSA 实现"已按范围决定
  转为 1.2/1.3 的交付物;1.4 只负责接入与评测,契约见 `quant_artifact_contract.md`。

本文件记录的中断根因审计(注释变更的 AST 等价性验证)仍然有效,
且其 `audited_source()` 已在 2026-10-04 扩展为按范围检查——
`scope="full"` 保持原行为,`scope="native"` 用模型仓库自身的 git 独立验证原生代码
未漂移,以免 `embodiq/src` 的改动阻塞一个从隔离树执行的补跑。详见
`task14_gap_analysis.md` §9.2。

后续快照：VLA-Adapter/FastWAM 四集合已完成，平均 97.55%/96.80%；OpenVLA Goal 仍在运行。已在独立副本开始 RTN 验证，最新报告见 `stage5_rtn_validation.md`，本文件其余内容保留此前恢复阶段记录。

## 已验证结果

对 `baseline_unified_v3_20260919/full` 逐任务 JSON 复核：

| 模型 | 集合 | 任务/回合覆盖 | 成功 | 成功率 | 错误 |
|---|---|---|---|---|---|
| OpenVLA | Spatial | 10 任务、500 回合 | 441 | 88.2% | 0 |
| VLA-Adapter Pro | Spatial | 10 任务、500 回合 | 490 | 98.0% | 0 |
| FastWAM | 全部 | 尚无全量结果 | — | — | — |

OpenVLA/VLA-Adapter 其余三个集合均未执行。已完成 1000/6000 基线回合、2/12 原精度集合单元，不能计算完整四集合平均，也不能称基线已全部完成。（2026-10-04 矫正：原精度已 12/12、6000/6000 全部完成并逐回合核验，宏平均 OpenVLA 76.95%、VLA-Adapter 97.55%、FastWAM 96.80%）

已完成的记录均对应 unified.v3、seed=42；逐回合初始状态哈希与 episode_manifest 一致。旧原生协议结果不拼入本表。

## 中断原因与证据

原控制器在 2026-09-19 14:41:22 以 `Source/config changed after pipeline freeze` 停止。唯一定位到的文件为 `OpenVLA/prismatic/models/vlas/openvla.py`，修改时间 10:57:25，差异为新增中文注释。

核验方法：使用固定 commit `c8f03f48af692657d3060c19588038c7220e9af9` 的该文件，在哈希计算中替代当前文件，其他源码字节保持原样。重建摘要与原冻结摘要完全相等；该文件修改前后 `ast.dump(..., include_attributes=False)` 完全一致。不存在本次审计范围以外的源码差异证据。

- 原冻结摘要：`34f6b34914ef2eea2d04c4d0933d83631257904bfb8d3a5faafcee0e2e058647`
- 含注释后的摘要：`03f8f183dfd7ec72a9a77ae2faefd1e1354b7eef7f78d823816ceafd61b0fb18`

保留用户注释和原始失败记录。恢复时只承认这一项已审计的注释变化，不关闭源码冻结检查，不自动接受未来代码变化。

## 已启动的续跑

工具：`tools/resume_unified_baselines.py`；启动时 PID 4147223。

最新状态：`results/baseline_unified_v3_20260919/recovery/status.json`。原根目录 status.json 保留原失败历史，不再代表最新续跑状态。

续跑先复核三个环境版本、全部输入文件 SHA256、初始状态文件；再验证已有结果的任务/回合覆盖与配对清单。完整集合复用，缺失任务才启动。OpenVLA 使用 GPU 1，VLA-Adapter 使用 GPU 2，FastWAM 使用 GPU 3；全部离线。失败不自动重试，不覆盖已完成结果。

已复核全部 132 个输入文件，哈希均通过；续跑已进入 full 阶段，两个 Spatial 集合标记 reused，OpenVLA Object、VLA-Adapter Object、FastWAM Spatial 三个作业已实际启动。此处不把“已启动”写成“剩余回合已完成”；以 recovery/status.json 和结果文件为准。

## 量化下一阶段的代码盘点

静态检查 `quant/api.py`、`swap.py`、`spec.py`、`qlinear.py` 得到：

- RTN：存在模块扫描、替换和低位 Linear 链路；VLA-Adapter 有旧协议结果，但三模型 unified.v3 的 RTN 主矩阵尚无结果。
- AWQ：底层模块支持传入 input_scale；当前 make_quant_apply 未提供激活采集/缩放搜索，也未把校准结果传给 apply_report。
- HSA：已有 HSASpec 和高精度 residual 张量执行接口；尚缺完整的敏感度计算、残差选择和工件加载链路。
- 直接构造 method=awq/hsa 不能证明实际采用相应算法，必须先实现明确的校准产物校验，再开放 CLI。
- FastWAM 的量化 CLI 仍主动拒绝未验证预设，避免套用其他模型的 LLM scope。

下一顺序：补齐公共原精度基线 → 固定各模型量化模块范围与校准集 → 验证 RTN → 接入真实 AWQ 校准 → 接入可替换 HSA 实现 → 执行配对闭环与资源专测。

基线运行期间不编辑被冻结的 src/模型源码；后续量化开发须使用独立副本或等基线结束，防止再次因源码变更中断。当前复核/续跑工具位于 tools，未改变评测模型与动作逻辑。
