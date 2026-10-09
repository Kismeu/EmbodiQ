# FastWAM 统一适配（2026-09-18）

## 范围

接入已验证的 `libero_uncond_2cam224.pt` first-frame 路径，默认不编译、不生成未来视频；复用官方 `eval_libero_single._predict_action_chunk`，不重写多相机处理、proprio 归一化、采样、动作反归一化或夹爪转换。Optional-IDM 其他路径不在本轮支持范围。

## 实现

- `src/embodiq/backends/fastwam/policy.py`：加载原生 Hydra 配置、本地 Wan 组件/任务权重/统计；预测 32×7 动作，num_valid=10；回合 reset 更新原生采样 seed。
- `src/embodiq/backends/fastwam/env.py`：使用原生环境创建/空动作，初始状态显式读取本地 LIBERO 资产，复用统一任务/回合循环。
- 共用入口 `python -m embodiq.entry.evaluate` 现可选择 openvla、vla_adapter、fastwam。各模型使用自己的 conda 环境及原生源代码，输出同一结果格式。
- FastWAM 必须传 checkpoint 文件和 dataset-stats。额外记录 stats SHA256、稳定配置 SHA256、采样步数、sigma_shift、编译开关和图像尺寸。配置摘要排除 Hydra 的时间输出目录。
- 本轮 FastWAM CLI 只开放原精度；量化预设明确报未验证，不把其他模型的 LLM scope 静默套到 FastWAM。Policy 已留 post-load quant_apply 接口供后续接入。

## 测试与修复

1. CPU/配置/接口回归：25 passed（4.53 秒），覆盖原生配置、32/10 时序、动作值不重复变换、错误长度、种子重置、配置摘要、初始状态与原生读取完全相同，以及现有合约/异常/续跑测试。
2. GPU 输出一致性：`test_parity_fastwam.py`，1 passed（189.22 秒）。三组固定原始观测，逐组对比原生与适配器完整 32×7 动作，逐元素一致，num_valid=10。显式启用 EMBODIQ_FASTWAM_GPU_TEST=1，模型/组件均离线加载。
3. 首次统一闭环在读取 LIBERO 初始状态时失败：PyTorch 2.6 默认 weights_only=True 不接受资产内的 NumPy 数组。已在 FastWAMEnvFactory 内按原生路径设 weights_only=False，仅用于已知本地 LIBERO 初始状态，加入路径边界及数组检查；未全局更改 torch.load。
4. 修复显存监控/争用检测将 CUDA 逻辑编号直接当 NVML 物理编号的问题。使用两套已安装 PyTorch 均提供的编号解析函数；实测 CUDA_VISIBLE_DEVICES=2 时 logical 0 -> NVML 2。此前映射错误的资源结果不追认。
5. 修复后统一闭环在 GPU 2 重跑完成，进程退出码 0。`results/stage3_fastwam_unified_smoke_v2/libero_spatial/gpu2_task0_results.json` 记录 2/2 成功、错误数 0；本次配置为 Spatial task 0、两个初始状态、seed=42、预测 32 步/执行 10 步。小样本结果仅验收适配功能，不是集合成功率结论。

本轮所有测试进程已结束。当前三个模型都具备同一入口/回合循环/结果格式下的原精度运行证据；OpenVLA 和 FastWAM 还各通过了原生/适配器固定观测输出一致性验证，VLA-Adapter 先前已有对应验证。

## 可复现命令

在 embodiq 目录执行，GPU 2 应有足够可用显存：

```bash
env CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true \
  /data/jyc/tools/anaconda3/envs/fastwam/bin/python -m embodiq.entry.evaluate \
  --model fastwam \
  --checkpoint '/data/jyc/space/Embodied AI/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt' \
  --dataset-stats '/data/jyc/space/Embodied AI/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json' \
  --suite libero_spatial --quant fp16 --protocol official.fastwam --seed 42 \
  --tasks 0 --trials 2 --gpu 2 --out results/stage3_fastwam_unified_smoke_v2
```

`fp16` 是旧 CLI 的原精度标签，实际 BF16。显式 seed=42 与先前 FastWAM 原生冒烟保持一致，并写入 protocol；不是声称项目已有 official.fastwam 默认 seed=0 与原生训练配置 seed=42 相同。

## 尚未覆盖

本轮仅接入与功能验证，不是正式 48 单元评测。MuJoCo 版本统一、四集合公共基线、模型错误计分口径、资源专测和三种量化完整接入仍待完成。GPU 运行时其他作业曾改变显存占用，记录保留；最终主表需固定独占测量条件，并进一步核验峰值采样和冷启动/预热边界。
