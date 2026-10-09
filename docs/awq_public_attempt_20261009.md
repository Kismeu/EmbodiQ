# AWQ 开源适配尝试结果（2026-10-09）

**结论：本次 OpenVLA 适配未通过项目现有数值门禁，AWQ 正式结果继续留白；HSA 继续等待上游。** 两次真实 GPU 尝试均已结束，没有启动正式 50 回合/任务的评测，也没有将试验结果加入正式矩阵。

## 尝试内容

- 使用 [MIT Han Lab 官方 llm-awq](https://github.com/mit-han-lab/llm-awq)，固定 commit `d6e797a42b9ef7778de8ee2352116e0f48a78d61`，本地保存在 `experiments/awq_public_attempt_20261009/upstream`，保留原始许可证。
- 模型为真实 `openvla-7b-finetuned-libero-spatial`，运行于 OpenVLA 环境、L40 GPU 0。CUDA 分配和推理可用；本次 NVML/nvidia-smi 监控异常，不报告正式显存或延迟结论。
- 使用现有 Spatial task 0 官方训练演示重渲染观测，逐文件校验 SHA256。第一次取 4 条，第二次扩到该任务已有的全部 32 条校准观测；均未使用开发集或评测成功率调参。
- 直接复用官方 `auto_scale_block` / `auto_clip_layer` 搜索函数，替换文本入口为 OpenVLA 的真实图像与语言前向输入；只采集 prefill，避免用非空 KV cache 重放 attention。
- 官方搜索模块导入依赖未安装的 `awq_inference_engine`。本次只提取并加载其搜索函数，未伪造扩展、未安装或变更模型环境。推理沿用 embodiq 已有 packed INT4 后端。
- 为与现有 RTN 一致，搜索的伪量化函数绑定到本项目 tinygemm 的 BF16 scale/float-offset 约定；缩放保留在 Linear 输入侧，不折叠到相邻层。**因此这是公开 AWQ 搜索对本项目后端的适配，不是官方完整配置的原样复现。**

原计划依次检查 32 个语言 block、224 个 Linear；若数值通过，再进入动作与闭环验证。两次都在第 0 个 block 的 7 层检查后停止。候选搜索前后的原始权重均做了逐元素一致性检查。

## 数值结果

阈值沿用 `src/embodiq/quant/swap.py`：INT4 权重相对 Frobenius 重构误差上限 `0.35`，没有为 AWQ 放宽。另检查 packed 输出相对其反量化参考输出的误差，上限 `0.05`。

| 第一语言 block 的层 | 4 条校准：权重相对误差 | 32 条校准：权重相对误差 | 32 条校准：packed 一致性误差 | 32 条校准：相对原精度输出误差 |
|---|---:|---:|---:|---:|
| attention q_proj | 6.4021 | 6.4221 | 0.002019 | 0.023593 |
| attention k_proj | 5.7957 | 5.8313 | 0.001963 | 0.022110 |
| attention v_proj | 0.9950 | 0.9950 | 0.002117 | 0.043315 |
| attention o_proj | 0.1877 | 0.1873 | 通过 | 见原始报告 |
| MLP gate_proj | 0.1069 | 0.1067 | 通过 | 见原始报告 |
| MLP up_proj | 0.1054 | 0.1052 | 通过 | 见原始报告 |
| MLP down_proj | 0.1364 | 0.1351 | 通过 | 见原始报告 |

两次 q/k/v 都超过权重门禁；全部 7 层的 packed 输出一致性检查均通过。增加到 32 条校准观测后，拒绝原因仍然存在。

这里的三种误差不能混用：

1. **权重误差**衡量反量化后的有效权重与原权重的偏差，是当前统一替换器的硬性门禁。
2. **packed 一致性误差**衡量真实 INT4 算子与其反量化参考的差异，用于检查执行路径；通过不意味着任务性能无损。
3. **校准输出误差**衡量已有校准输入上的输出偏差，只是局部诊断，既不是独立开发集验证，也不是动作/闭环成功率。

当前证据提示：激活加权的搜索目标与统一的无激活权重重构门禁可能不一致。不能把这次门禁拒绝直接解释为 AWQ 算法无效，或三模型均无法使用 AWQ。若将来重新启动这条路线，应先检查适合激活感知量化的验收口径，并做独立开发集与块输出验证；本轮按“失败则留白”的要求到此停止，不绕过现有检查继续填表。

## 产物与复跑

新增运行脚本：`tools/attempt_public_awq_openvla.py`。每次运行记录公开源码版本/哈希、运行脚本哈希、校准清单哈希、逐层指标和异常；试验目录中的 `runner_snapshot.py` 保存该次实际脚本。

| 路径（项目根目录下） | 内容 |
|---|---|
| `experiments/awq_public_attempt_20261009/openvla_spatial_attempt1/report.json` | 4 条校准的失败报告 |
| `experiments/awq_public_attempt_20261009/openvla_spatial_attempt1.log` | 第一次完整日志 |
| `experiments/awq_public_attempt_20261009/openvla_spatial_attempt2/report.json` | 32 条校准的复核报告 |
| `experiments/awq_public_attempt_20261009/openvla_spatial_attempt2.log` | 第二次完整日志 |
| `experiments/awq_public_attempt_20261009/openvla_spatial_attempt2/candidate_layers.pt` | 被拒候选的层缩放/裁剪诊断，独立格式 `embodiq/rejected-awq-diagnostics/1`，不是可部署工件 |
| 两次试验目录内的 `traceback.txt`、`runner_snapshot.py` | 错误堆栈及源代码快照 |

复跑第二次配置时使用新的输出目录：

```bash
/data/jyc/tools/anaconda3/envs/openvla/bin/python -u \
  tools/attempt_public_awq_openvla.py \
  --upstream experiments/awq_public_attempt_20261009/upstream \
  --out experiments/awq_public_attempt_20261009/openvla_spatial_new_attempt \
  --gpu 0 --blocks 32 --calibration 32
```

本次未生成可部署 `artifact.pt`。`artifacts/awq_public`、正式 AWQ/HSA 结果目录以及历史原精度/RTN 结果均未写入。正式数据仍为原精度 + RTN 的 24/48 单元、12,000 回合。无需为了填补缺项继续投入 AWQ/HSA 数据生产。
