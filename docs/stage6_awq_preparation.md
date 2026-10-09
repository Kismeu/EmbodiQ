# AWQ 校准准备与缩放搜索试验（2026-09-20）

## 2026-10-04 更正（本节为最新状态，下文保留历史记录）

下文"下一步"的路线已被范围决定取代:**AWQ 算法由 1.2 提供,1.4 只负责接入与评测**。
本阶段产出的校准数据与搜索实现仍然有效,但"扩到全 scope 并部署"不再是 1.4 的工作。

已变化的部分:

- **接入通路已建成并验证**:`src/embodiq/quant/artifact.py` 定义
  `embodiq/quant-artifact/1` 格式与 fail-closed 校验(33 项测试),
  交付方文档见 `quant_artifact_contract.md`。`--quant awq --quant-artifact <path>`
  已实测跑通闭环(VLA-Adapter Spatial,568 层全覆盖,2/2 成功)。
- **主干已补齐**:本阶段的 `calibration.py`、`awq_search.py`、`awq_blocks.py`、
  `awq_artifact.py` 已从 `experiments/quant_stage6/src` 合入 `src/embodiq/quant/`;
  同时补上了主干缺失的 FastWAM 别名规则与 `clip_max`，以及原本**完全缺失的
  方法闸门**(按旧主干代码 `--quant awq` 会静默执行 RTN)。
- **`artifacts/awq_block_pilot_v1/` 仍为空目录**:`tools/run_awq_block_pilot.py`
  在 2026-09-20 17:14 创建目录后未产出任何文件。该试验未执行成功,不作为结论依据。
- `artifacts/awq_scale_pilot_v1/` 的产物仍然有效,但仅覆盖 VLA-Adapter
  第一个语言块的 7 个 Linear,是局部重建指标,不可转译为闭环成功率结论。
- 校准数据已全部就绪:1280 校准 + 320 开发观测(40 任务 × 32/8),
  40 个官方 HDF5 逐文件 SHA256 已记录。

下文第 6 节"下一步"中与 1.4 相关的部分已完成;其余(全 scope 搜索、裁剪适配)
属于 1.2。

## 工作边界

原精度基线与 RTN 尚有 OpenVLA 集合在运行。其源代码均未修改，哈希复核通过。本阶段代码位于独立 `experiments/quant_stage6/src`，既不修改原 src，也不修改 RTN 使用的 quant_stage5。

本轮完成校准数据准备链路和真实激活缩放搜索试验；没有把试验产物当作已部署 AWQ 模型，没有向正式比较表填入 AWQ 成功率。

## 训练数据

- 官方本地下载工具指向 `yifengzhu-hf/LIBERO-datasets`。
- 固定 revision：`f13aa24a3da8c43c7225569f28c562979fa0e35a`。
- 只下载 Spatial/Object/Goal/Long 共 40 个 HDF5 文件，合计 33,784,856,577 字节；不下载其他 90 任务。
- `tools/download_calibration_demos.py` 使用 hf-mirror.com，清除代理变量、HTTP trust_env=False，禁用隐式 token/Xet，保存每个文件 SHA256。
- 下载状态：`calib/libero_training_demos/status.json`，启动 PID 2368788。

只从训练演示读取当前模拟器状态与场景 XML，不读取评测轨迹，也不依据最终任务成功率选样/调参。由于是同一 LIBERO benchmark，训练和评测可包含相同任务，不能称为任务隔离或零样本泛化实验。

## 观测生成与划分

`tools/prepare_calibration_observations.py` 使用 MuJoCo 3.3.2、256×256 双相机，从每个演示的存储状态重新渲染原始观测。恢复对应场景 XML，修复官方文件中作者机器绝对路径及旧 chiliocosm 包名，所有资产仅指向本地文件。

固定每任务 demo_0..7 为校准、demo_8..9 为开发验证，每条轨迹均匀取 4 帧。轨迹编号不重叠；每任务 32/8 观测。该划分是本项目校准/开发划分，不冒充官方 train/test 划分。

样本清单记录数据 revision、源 HDF5 SHA256、demo/frame 编号、原始状态与 XML 摘要、输出 NPZ 摘要。NPZ 只包含两相机和当前机器人状态，不含动作/奖励/成功标签。

首个 Spatial task 0 已生成 40 观测，位于 `calib/observations_spatial_task0_v1`；校准/开发 demo ID 分别为 0..7/8..9，互不重叠。相机 shape、有限数值和非空图像检查通过。

全任务准备控制器 `tools/prepare_all_calibration.py` 已后台启动（PID 2387416），状态 `calib/observations_training_v1/status.json`。依次渲染下载完成的文件，等待其余文件最多 24 小时，失败保留日志；全部完成时应有 1280 个校准、320 个开发观测。当前不能把目标数量当成已完成数量。

## 激活采集与搜索

新增 `quant/calibration.py`：对指定模块收集所有调用的 mean absolute activation，同时用固定种子的有界 reservoir 保存输出重建样本。每次调用按 token 位置均匀选候选，reservoir 在调用候选间抽样，不声称对所有原始 token 严格均匀抽样。缺少激活、非有限输入和错误预算均报错，退出时移除 hooks。

新增 `quant/awq_search.py`，标识 `awq_linear_scale_search_v1`：20 点激活指数网格，原始 RTN 对应 alpha=0 候选，目标为单个 Linear 的输出 MSE；考虑本地 packed 格式 BF16 scale/zero 的舍入。当前不包含权重裁剪，也不执行 attention/MLP block 级重建目标，不能冒充官方完整 AWQ recipe。

参考：[AWQ 官方缩放实现](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)、[官方裁剪实现](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_clip.py)。架构适配差异必须在后续正式方法定义中说明。

12 项准备/防误标签测试通过，覆盖采样上限、确定性、异常后清理、缺失激活、零输入通道、输入维度以及 RTN 候选不被搜索丢弃。

## 真实模型试验

`tools/calibrate_awq_pilot.py` 在 GPU 3 加载 VLA-Adapter Spatial-Pro 原精度，用上述 32 个训练观测和 8 个开发观测验证第一语言块的 7 个 Linear。

- 每层校准触发 32 次，统计 20,256 行输入，保存 64 行有界重建样本。
- alpha 选择：q=0.20、k=0.30、v=0.35、o=0.25、gate=0.20、up=0.35、down=0.40。
- 7 层开发集重建 MSE 均低于其 RTN 候选。这只是局部重建指标，不能转译为闭环成功率结论。
- 产物：`artifacts/awq_scale_pilot_v1/report.json`、`input_scales.pt`、`activation_samples.pt`，均绑定观测清单和模型权重摘要。
- 报告明确标记 `search_pilot_not_deployed_awq`、`block_objective=false`、`clipping=false`。

下一步：校验全部训练观测 → 完成可比较的 AWQ block/裁剪适配与产物加载校验 → 真实 packed 推理数值检查 → 三模型闭环冒烟 → 配对正式评测。HSA 尚未实现选择/残差工件完整链路，后续仍保留独立可替换实现接口。

## AWQ 适配尝试记录（2026-10-09）

已使用完整训练校准观测对 VLA-Adapter 做两次隔离尝试：

- 全 scope：568 个 INT4 层，视觉 qkv/MLP 层重构误差最高约 0.93，严格校验拒绝；
- language-only：168 个 INT4 层，`language_model.model.layers.3.mlp.up_proj` 重构误差约 1.07，严格校验拒绝。

两次工件均保存在 `artifacts/awq_experimental/`，不进入 `artifacts/awq_public/`，也不计入正式矩阵。结论是当前 per-Linear 搜索不能替代公开 AWQ 的 block-level scale/clipping 优化。后续只有接入真正的 block-level AWQ 并通过统一校验后，才允许重新启用 AWQ 正式单元；在此之前正式结果只统计 FP16/RTN。

## 官方 block 搜索适配复核（2026-10-09，本次执行）

按用户“AWQ 利用现有代码尝试，失败则留白；HSA 留白”的要求，固定 Han Lab
官方代码版本，在真实 OpenVLA Spatial checkpoint 上完成 4 条与 32 条训练观测
的两次适配尝试。搜索和 packed INT4 执行可运行，但首个语言 block 的 q/k/v
权重重构误差均超过项目现有 0.35 门禁，未进入闭环。未修改门禁、未产生正式 AWQ
结果，当前停止该路线。详细数值、适配差异、证据限制和复跑命令见
[AWQ 开源适配尝试结果](awq_public_attempt_20261009.md)。
