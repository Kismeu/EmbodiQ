# 公共协议原精度基线（2026-09-19）

## 2026-10-03 核验更正（本节为最新状态，下文保留历史记录）

原精度基线**已全部完成并通过逐回合核验**：12/12 集合、6000/6000 回合。
`tools/report_baseline.py` 重跑通过全部断言——每集合 task_id 0..9、每任务 50 回合、
`error_episodes` 为空、逐回合 `seed` 与 `initial_state_sha256` 与
`episode_manifest.json` 完全一致。

| 模型 | Spatial | Object | Goal | Long | 四集合宏平均 |
|---|---:|---:|---:|---:|---:|
| openvla | 88.20% | 89.20% | 78.20% | 52.20% | 76.95% |
| vla_adapter | 98.00% | 99.40% | 97.40% | 95.40% | 97.55% |
| fastwam | 97.00% | 99.20% | 97.40% | 93.60% | 96.80% |

下文"进入 hashing_inputs""尚未宣布 6000 回合完成"等描述属于 2026-09-19 的历史状态，已过期。

后续复核：原进程因注释导致的源码字节变化停止，仅完成 OpenVLA Spatial 441/500、VLA-Adapter Spatial 490/500。已核对 AST 等价并启动带输入复核的续跑，最新状态在 `results/baseline_unified_v3_20260919/recovery/status.json`。详情见 `baseline_review_and_quant_next.md`，下文启动状态保留为历史记录。

## 运行范围

本轮仅建立原精度基线：OpenVLA、VLA-Adapter Pro、FastWAM uncond first-frame，各四个 LIBERO 集合、每集合 10 任务、每任务 50 回合，共 6000 回合、12 个集合结果单元。RTN/AWQ/HSA 尚不在本轮队列。不得将这 12 个原精度单元解释为最终 48 单元矩阵已完成。

启动前回归：FastWAM 环境 28 passed，OpenVLA 环境 22 passed。FastWAM MuJoCo 从 3.1.6 升级为 3.3.2，经清华镜像禁用代理安装；未修改其他模型框架版本。

## 固定协议 unified.v3

| 项目 | 设置 |
|---|---|
| 仿真 | MuJoCo 3.3.2、robosuite 1.4.1，共用本地 LIBERO 代码与资产 |
| 集合 | Spatial、Object、Goal、Long（libero_10） |
| 每任务回合 | 50 个不同初始状态，禁止不够时循环重复 |
| 初始状态 | 三模型使用同一读取函数、同一文件和编号顺序 |
| 随机性 | seed=42；每回合重置环境、Python、NumPy、PyTorch RNG |
| 最大执行步数 | 400 / 400 / 400 / 700；不含 30 步稳定等待 |
| 渲染 | 256×256；各模型原生相机与预处理保留 |
| 预测/执行长度 | OpenVLA 1/1、VLA-Adapter 8/8、FastWAM 32/10 |
| 原精度 | BF16；旧 CLI 名称 fp16 仅为兼容标签 |
| FastWAM | sigma_shift=5.0，10 次采样步，不编译，不生成未来视频 |
| 终止 | 每执行一步检查成功；达到上限计超时失败 |
| 错误 | 模型错误计失败并记录；环境错误标无效；OOM/中断上抛，禁止静默吞掉 |

结果 schema 更新为 embodiq/run_result/2，历史 schema/1 文件保留不覆盖。错误回合仍令该作业 degraded/非零退出，必须检查后处理，不能把删去错误样本后的成功率当主结果。记录增加查询 P95。显存采样峰值在预热后重置，但正式资源比较还需独立固定输入专测。

## 后台流水线

入口：`tools/run_unified_baselines.py --background`。

运行根目录：`results/baseline_unified_v3_20260919`。

1. 检查三个环境的仿真版本，导出环境/代码摘要和 40 任务 × 50 初始状态清单。
2. 每模型 Spatial task 0 运行 2 回合。只有三个作业均退出成功、回合齐全、无错误时才进入下一阶段。该门槛检查工程完整性，不以小样本成功率预设方法优劣。
3. 对所用权重、处理器、配置、Wan 组件逐文件计算 SHA256，完整后冻结输入清单。
4. 每模型依次执行四集合全量原精度基线。每个任务完成后写入结果 JSON；每个回合在日志中输出状态。
5. 失败不自动重试；对应模型队列停止，已完成结果保留。其余独立模型可继续。全部完成后状态才变为 complete。

GPU 分配：OpenVLA 使用物理 GPU 2；VLA-Adapter/FastWAM 在物理 GPU 3 通过锁串行运行。避免占用检查时 GPU 0/1 的其他作业。各进程显式设置 CUDA_VISIBLE_DEVICES 与 MUJOCO_EGL_DEVICE_ID，推理启用 Hugging Face/Transformers 离线及 DIFFSYNTH_SKIP_DOWNLOAD。

代码摘要在每个作业开始前复核；运行期间不得编辑模型/适配器源码或配置。需要修改时应先检查运行队列并创建新版本，不混入本批结果。

## 进度与证据文件

- `status.json`：控制器 PID、阶段、各作业 PID/命令/退出码/完成回合数。
- `controller.log`、`logs/*.log`：启动、模型加载和逐回合日志。
- `environment_manifest.json`：三个环境包版本、代码 SHA256 与 GPU 分配。
- `episode_manifest.json`：40 个任务及 2000 个共享 episode 配置（在三模型上复用，共 6000 回合）。
- `input_manifest.json`：权重与辅助输入的内容哈希；只有 complete=true 表示哈希阶段完成。
- `full/<model>/<suite>/*_results.json`：全量基线结果。

v3 控制器 PID 3542332，环境/初始状态清单已导出。三个模型小规模检查均已完成：各 2/2 成功，合计 6/6、零错误；三个结果的 seed=42、MuJoCo=3.3.2 及逐回合初始状态哈希与清单一致。当前进入 hashing_inputs，完整权重哈希后自动进入全量。尚未宣布 6000 回合完成，以 status.json 与实际结果为准。旧 v2 控制器 PID 3526708 已失败结束。

## 后续验收

必须核对每集合 task_id=0..9、每任务 50 个 episode、无无效/错误回合，以及三模型相同 episode 的初始状态哈希一致。之后才能形成四集合成功率与宏平均基线表。

本批同时记录闭环显存/延迟，但不会替代固定输入、独占 GPU、明确冷启动/稳态口径的正式资源专测。HSA 的算法定义、校准与量化范围尚未全部冻结，不因原精度流水线已启动而标记最终实验协议全部完成。
