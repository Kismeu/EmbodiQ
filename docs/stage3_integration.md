# 统一评测入口接入进度（2026-09-18）

2026-09-19：开始统一 MuJoCo 3.3.2 与 unified.v2 公共协议原精度基线，见 `stage4_unified_baseline.md`。本批保留独立结果目录与版本，不覆写此前接入验证。

当前最新：FastWAM 已接入共用入口，GPU 动作一致性测试通过，统一入口 Spatial 两回合 2/2 成功、错误数 0；三个模型原精度接入验证均已有运行证据。见 `stage3_fastwam_adapter.md`。下文为各轮历史实施记录；完整量化主矩阵尚未执行。

最新更新：所需下载均已完成（转换版仓库失败由官方原始组件替代）；OpenVLA GPU 一致性测试通过、统一入口 Object 冒烟 3/4 成功，FastWAM 原生闭环 1/1 成功。详见 `stage3_gpu_validation.md`。以下下载量和显存不足描述保留为前一轮实施快照，不代表当前状态。

## 当前交付

- 新增 `embodiq.backends.openvla.policy.OpenVLAPolicy`，复用原生图像处理、提示词/动作解码和夹爪转换，返回统一 `(1, 7)` 动作块。
- 新增共用入口 `python -m embodiq.entry.evaluate --model ...`。OpenVLA/VLA-Adapter 使用同一任务循环、回合执行器与结果格式；旧 `run_adapter` 命令仍兼容。
- `--checkpoint` 指定单套权重；VLA-Adapter 也可使用 `--checkpoint-root`；`--repo-root` 可指定原生源码。
- 每个进程只导入一个原生仓库，拒绝已经加载的同名 experiments/prismatic 包来自另一个仓库，避免两个模型的代码串用。
- 断点续跑检查模型、权重路径、量化设置、协议、版本、源码摘要及回合完整性；不兼容结果要求使用新目录，不覆盖旧结果。
- 新增源码 SHA256 记录，量化配置同步到 policy.quant_id；出现错误回合时入口返回非零退出码。

## 验证边界

共用入口 help、py_compile 与 git diff --check 通过。CPU 合约、异常处理、续跑检查共 19 passed（11.98 秒，test_resume_identity.py、test_contract_and_rollout.py、test_rollout_error_surfacing.py）。OpenVLA 新适配器尚未通过 GPU 输出一致性及闭环验证：检查时四张 GPU 仅空闲约 2.8–3.6 GiB，不满足 BF16 模型加载需要。此前 8/10 Spatial 结果来自原生入口，不能当作新适配器通过证据。FastWAM 尚未接入共用入口。

已有 result/rollout 对错误回合的成功率分母口径仍与 experiment_protocol.md 不完全一致；本轮通过非零退出码与禁止复用错误结果阻止其被当作完整实验，正式主表前仍需统一模型失败/基础设施失败分类。现有显存监控的 CUDA 逻辑编号到 NVML 物理编号映射也需要在独占资源测试前修正和验证。

## GPU 可用后的下一条命令

在 embodiq 目录、openvla 环境中，以一个 Object 回合验证共用入口：

```bash
env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/jyc/tools/anaconda3/envs/openvla/bin/python \
  -m embodiq.entry.evaluate --model openvla \
  --checkpoint /data/jyc/models/openvla-7b-finetuned-libero-object \
  --suite libero_object --quant fp16 --protocol official.openvla \
  --tasks 0 --trials 1 --out results/stage3_openvla_object_smoke
```

通过后再做固定观测的原生/适配器输出一致性，扩展四集合冒烟；量化与最终主矩阵不由一次成功冒烟替代。`fp16` 为兼容旧 CLI 的原精度标签，实际加载 BF16。

## 后台无代理下载

脚本：`tools/download_missing_models.py`。清除名称包含 proxy 的环境变量，HTTP session.trust_env=False，禁止隐式 token 和 Xet/额外传输后端，固定 HF_ENDPOINT=https://hf-mirror.com，无官方端点回退。镜像提供的文件重定向也使用同一无代理 HTTP session。

| 工作进程（启动时 PID） | 内容 | 状态文件 |
|---|---|---|
| 1286944 | tokenizer、OpenVLA Goal、OpenVLA Long 队列 | `artifacts/mirror_download/status.json` |
| 1290228 | Wan 原始 VAE 与 UMT5 文本编码器 | `artifacts/mirror_download_wan_original/status.json` |

tokenizer 已下载；Goal 和 Wan 大文件已产生实际下载数据，尚未完成；Long 排在 Goal 后。检查时 Goal 目录占用约 1.3 GiB、Wan 目录约 591 MiB，包含未完成临时文件，不代表可加载。后台最多每任务尝试 3 次；不会无限重试。进程持有锁防止同一队列重复运行，记录每个来源 revision；最终 downloaded 后仍需独立检查文件、哈希和加载兼容性。

原转换权重仓库 `DiffSynth-Studio/Wan-Series-Converted-Safetensors` 在镜像返回 401，失败记录保留。第二进程下载 Wan 官方仓库原始文件作为替代，后续 FastWAM 必须使用 `model.redirect_common_files=false`，并设 `DIFFSYNTH_MODEL_BASE_PATH` 指向 FastWAM/checkpoints；不能继续按转换版路径寻找文件。原生配置已启用 skip_dit_load_from_pretrain，配合现有完整任务 checkpoint，暂不下载额外视频 DiT 预训练分片。

下载在独立会话后台持续运行；GPU 检查未启动排队推理作业，也未停止已有用户进程。查看状态文件和同目录 worker.log 可判断完成/失败，不把后台启动成功当作模型下载完成。
