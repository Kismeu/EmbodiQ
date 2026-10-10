# 阶段 2：原精度基线与进度核查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: 实验准备与局部验证
- Origin Date: 2026-09-17
- Updated Date: 2026-09-18（收尾确认；实验日志日期为 2026-09-17）
- Verification Status: PARTIAL（分项证据如下，非三模型全量验证完成）
- Version Label: stage2_audit_v2

后续进度见 `stage3_integration.md`：用户已授权后台无代理镜像下载缺失文件，OpenVLA 共用入口适配已开始；本文件保留阶段 2 检查时的证据快照。

## 目标与结论边界

目标保持为三模型 × 原精度/RTN/AWQ/HSA × 四个 LIBERO 集合，并比较成功率、显存和延迟。当前完成资源盘点和部分运行检查，缺权重依赖的检查按用户指示暂缓；不能宣称阶段 2 全部完成。

## 权重与环境清单

| 模型 | 已找到权重 | 核验 | 缺项 |
|---|---|---|---|
| OpenVLA | `/data/jyc/models/openvla-7b-finetuned-libero-spatial` 和 `...-object` | 每套四个 safetensors 分片齐全、索引引用文件存在，982 个 tensor key 可读取，配套统计存在，分片完整 SHA256 已生成 | 指定本地范围未找到 Goal/Long 权重；Object 尚未执行闭环 |
| VLA-Adapter Pro | `/data/jyc/data/VLA-Adapter/LIBERO-{Spatial,Object,Goal,Long}-Pro` | 每套主权重可打开，982 个 tensor key；统计、动作头与状态投影存在；四套已有冻结清单校验 PASS；主权重/动作头/状态投影完整 SHA256 已生成 | 已新增 `--checkpoint-root`，可传实际路径，保留旧默认；公共协议基线待建立 |
| FastWAM | `FastWAM/checkpoints/fastwam_release/libero_{uncond,optional_idm}_2cam224.pt` 及 stats | uncond 使用 CPU mmap、weights_only=True 成功读取，顶层为 mot/step/torch_dtype/proprio_encoder | 在项目/模型/数据/HF 缓存检查范围未找到 Wan2.2 VAE、UMT5 文本权重及 tokenizer；optional-IDM 未反序列化核验 |

VLA-Adapter 旧清单对大权重仅记录文件大小，对小文件记录内容哈希。PASS 表示符合旧清单，不是完整大权重 SHA256 校验。safetensors 头部读取不等于完成每个张量的数据完整性验证。

另已对 22 个权重文件计算完整 SHA256，保存在 `docs/stage2_weights.sha256`：VLA-Adapter 四套主权重、四个动作头、四个状态投影；OpenVLA Spatial/Object 共八分片；FastWAM 两份任务权重。这是当前文件的内容快照，不是与上游可信摘要比对后的认证；处理器/配置等尚需纳入最终冻结清单。

| Conda 环境 | PyTorch | Transformers | NumPy | MuJoCo | robosuite |
|---|---|---|---|---|---|
| openvla | 2.2.0+cu121 | 4.40.1 | 1.26.4 | 3.3.2 | 1.4.1 |
| vlaadapter | 2.2.0+cu121 | 4.40.1 | 1.26.4 | 3.3.2 | 1.4.1 |
| fastwam | 2.6.0 | 4.50.3 | 2.2.6 | 3.1.6 | 1.4.1 |

环境均位于 `/data/jyc/tools/anaconda3/envs/`。FastWAM 当前环境与仓库 pin 存在多处版本偏差，不能视为可复现环境已锁定。

FastWAM 初次 `import fastwam.runtime` 因缺 boto3 失败；本次按仓库版本经清华 PyPI 安装 boto3==1.35.99 及 botocore/jmespath/s3transfer 后，导入通过。未批量升级 PyTorch/MuJoCo 或其他已有依赖。运行源码内置 Wan 加载器，不需要因为未安装 diffsynth 包就判定缺依赖。

## 历史结果复核

直接读取 `results/fp16_baseline_20260902_2125/*/*_results.json` 共 40 份文件：每集合 task_id=0..9，记录 500 回合，error_episodes 均为空。

| 集合 | 成功/回合 | 成功率 |
|---|---|---|
| Spatial | 495/500 | 99.0% |
| Object | 497/500 | 99.4% |
| Goal | 486/500 | 97.2% |
| Long | 476/500 | 95.2% |
| 宏平均 | 合计 1954/2000 | 97.70% |

目录名虽为 fp16，policy.torch_dtype 实际记录 bfloat16。40 份记录全部 contended=true，其延迟不能进入正式主表。属于旧原生协议结果，当前公共协议基线尚未建立；权重历史路径与现位置不同且缺完整内容哈希，不能声称完全重建历史 provenance。

另有两套完整 VLA-Adapter RTN 历史记录（各 40 文件/2000 回合）：

| 方法 | Spatial | Object | Goal | Long | 宏平均 |
|---|---|---|---|---|---|
| rtn-llm | 99.2% | 98.6% | 95.2% | 92.6% | 96.40% |
| rtn-all | 99.0% | 93.2% | 90.2% | 90.4% | 93.20% |

以上为任务 JSON 聚合复核，不是本次重跑；RTN 各有 30/40 份文件标为 contended=true。两个 scope 不得混成同一个 RTN 主实验版本；正式量化范围仍待锁定。

OpenVLA 旧 Spatial 日志截止任务 0 第 23 回合开始，只有 22 个已完成回合，19 成功；不能把该日志当成完整集合基线。

## 本次验证

1. `tests/test_contract_and_rollout.py` 与 `tests/test_rollout_error_surfacing.py`：11 passed，13.54 秒。为 CPU/mock 接口检查。
2. `tests/test_parity_adapter.py`：3 passed，140.94 秒。使用实际 Spatial-Pro 权重与已有 fixture，在 GPU 0 上验证逐元素动作一致、动作块形状及夹爪约定。checkpoint 通过 EMBODIQ_ADAPTER_CKPT 指向真实路径；HF/Transformers 均离线。随后通过统一入口另跑 Spatial task 0 的 1 个回合，1/1 成功、error_episodes 为空，结果在 `results/stage2_audit/adapter/libero_spatial/gpu0_task0_results.json`；GPU 实际为 CUDA_VISIBLE_DEVICES=1，文件中的 gpu0 是逻辑编号，不是物理编号。新加的 --checkpoint-root 参数经此运行验证，并通过 py_compile 和 git diff --check。
3. FastWAM：runtime 导入通过、uncond checkpoint 安全反序列化通过。缺少 VAE/文本组件，未触发会自动下载依赖的原生评测入口，闭环未完成。

硬件：沙箱内 nvidia-smi 不可见；沙箱外读取正常，4 张 L40 均存在其他作业，空闲约 28–29 GiB。本次仅功能验证，所有新 GPU 测试的耗时均不作资源基准。未停止其他作业。

## 全项目当前进度

| 项目 | 状态 |
|---|---|
| 目标与协议文档 | 已建立；方法范围明确，技术参数尚未全部冻结 |
| 三模型源码 | 已存在，HEAD 见 experiment_protocol.md |
| VLA-Adapter 原精度 | 历史全量记录已聚合核对；本次真实权重一致性及 1 回合闭环通过 |
| FastWAM 原精度 | 任务权重已有；组件缺失，环境尚不一致，闭环待补 |
| 统一模型适配 | VLA-Adapter 有实现；openvla/fastwam 后端仅空 __init__.py，不能算接入完成 |
| RTN | VLA-Adapter 有历史闭环记录，其他模型未验证 |
| AWQ/HSA | 有 spec/低位模块/输入缩放及残差接口；本次未发现完整校准搜索与统一调用链证据，不能算已完成 |
| 正式公共协议主矩阵 | 0/48 单元完成验收；历史原生数据不等同公共协议主矩阵 |

## 下一步与下载约束

先完成无需下载的路径配置和模型接入工作，复用已通过的 VLA-Adapter 检查。补齐 OpenVLA Goal/Long 与 FastWAM VAE/文本组件后，方可完成三个模型四集合原精度验证。缺 checkpoint 可暂缓；需要下载时使用镜像并明确禁用用户代理，不得消耗用户 VPN 流量。正式显存/延迟另等独占 GPU 条件。

### 本次 VLA-Adapter 闭环复现命令

在 embodiq 目录执行（需可用 GPU）：

```bash
env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/jyc/tools/anaconda3/envs/vlaadapter/bin/python \
  -m embodiq.entry.run_adapter --suite libero_spatial --gpu 0 \
  --quant fp16 --trials 1 --tasks 0 \
  --checkpoint-root /data/jyc/data/VLA-Adapter \
  --out results/stage2_audit/adapter
```

重跑应指定新的输出目录，保留本次记录。`--quant fp16` 是现有 CLI 原精度标签，实际 BF16 以结果元数据为准。

本轮不以“继续等下载”冒充完成，也不把 FastWAM 的 runtime import 当成成功运行模型。
