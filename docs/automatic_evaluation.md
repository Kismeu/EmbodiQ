# 一键式评测（本机 CLI 第一版）

当前开放 OpenVLA、VLA-Adapter、FastWAM，以及原精度基线和 RTN。
一次提交自动执行预检、闭环评测、资源专测、报告；AWQ/HSA 返回待接入错误，不会启动或回退。
这是本机顺序执行器，尚无 Web 界面或跨机器调度。

## 一条命令开始

从 embodiq 仓库根目录运行，控制器不加载模型，各阶段分别进入对应模型环境。

```bash
# 三模型 × 原精度/RTN × LIBERO 四集合（12,000 回合）；GPU 索引按本机情况填写
PYTHONPATH=src python -m embodiq.ctl.cli launch configs/stage_fp16_rtn.json --gpu 3
```

此命令前台运行，立即打印唯一运行目录。无需再逐条调用 `execute`、资源专测或报告脚本。
RTN 在模型加载阶段执行，沿用已验证的量化范围和 W4/group128 设置。
每个模型/方法完成所选集合的闭环后，再进行固定输入资源专测；最终生成本次运行的报告。
实际按阶段列表顺序执行，当前只有一张 GPU、一个工作进程，不会并发写同一份状态。

正式档位每集合 10 个任务 × 50 回合。选择全部三模型、两方法、四集合时，
计划为 24 个闭环阶段、6 个资源阶段、1 个报告阶段。
执行时间包括模型加载及大量闭环操作，不承诺秒级完成。

```bash
# 查看可用能力
PYTHONPATH=src python -m embodiq.ctl.cli capabilities

# 只检查配置和显示展开计划，不创建运行、不占 GPU
PYTHONPATH=src python -m embodiq.ctl.cli launch configs/stage_fp16_rtn.json --dry-run

# 选择一个模型、两方法、一个任务集合
PYTHONPATH=src python -m embodiq.ctl.cli launch configs/stage_fp16_rtn.json \
  --models vla_adapter --methods fp16 rtn --suites libero_spatial --gpu 3

# 小规模真实流程验证：每个所选集合仅 task 0、1 回合；资源每集合 1 条输入、2 次预热、1 次测量
PYTHONPATH=src python -m embodiq.ctl.cli launch configs/stage_fp16_rtn.json \
  --suites libero_spatial --profile smoke --gpu 3
```

`smoke` 明确标为开发验证，不进入正式矩阵。它保留任务原有动作步上限，不用缩短任务伪造成功率。
只有正式档位四个集合全部完整时才给出四集合宏平均；子集报告称为“所选集合平均”。
若只选择 RTN，则原精度差值留白，不自动混入历史基线。

## 状态、停止与恢复

将命令输出的运行目录替换下例 `RUN_DIR`。路径含空格时加引号。

```bash
PYTHONPATH=src python -m embodiq.ctl.cli status "RUN_DIR"
PYTHONPATH=src python -m embodiq.ctl.cli stop "RUN_DIR"
PYTHONPATH=src python -m embodiq.ctl.cli resume-run "RUN_DIR" --gpu 3
PYTHONPATH=src python -m embodiq.ctl.cli report --run "RUN_DIR"
```

- GPU 忙或被另一个 embodiq 运行锁定时，默认立即标为 `blocked` 并返回非零，保留运行目录。
  `--wait-seconds 600` 可等待最多 600 秒，期间响应停止请求。
- 控制器运行期间发现其他计算进程占用同一 GPU，会停止自己启动的工作进程并阻塞；
  不终止其他用户进程。独占要求适用于当前全部 GPU 阶段，避免资源测量受争用影响。
- `stop` 写入停止请求；控制器会终止当前工作进程组、停止后续阶段。Ctrl-C/SIGTERM 同样处理。
  超过 10 秒未退出会升级终止信号。停止不意味着当前回合已保存，恢复依赖上一个回合检查点。
- `resume-run` 校验输入和已完成阶段产物，跳过完成阶段；中断的闭环会交由评测入口继续回合断点，
  中断的资源阶段会重新测量该模型/方法的全部所选集合。
- 执行错误、阻塞和停止分别保留 `failed`、`blocked`、`stopped`，不计为完成。
  任一阶段失败时停止后续阶段；修复环境问题后恢复同一运行，改变模型或代码则创建新运行。
- 工作进程继承运行锁与 GPU 锁。控制器被 SIGKILL 时，仍活着的工作进程会持锁，防止重复启动。
  这种情况下控制器不能处理停止标志；需等待残留工作进程结束或人工核对后清理，再恢复。

`status` 显示逐阶段状态、尝试次数、日志路径和错误原因。
完成运行的 `report --run` 检查保存的产物哈希并返回报告路径；未完成运行可尝试生成诊断报告，
缺失阶段会被列为准入问题，不会给出有效正式平均值。

## 输入配置与可追溯性

沿用 `configs/stage_fp16_rtn.json`；可通过 `--models`、`--methods`、`--suites` 选择子集。
自定义权重在配置中设置 `checkpoint` 或 `checkpoint_root`，FastWAM 还需匹配的 `dataset_stats`。
闭环和资源专测使用同一份解析后的逐集合模型路径。
`EMB_ROOT`、`DATA_ROOT`、`OPENVLA_ROOT` 可调整默认根目录；
模型 Python 环境可通过 `EMBODIQ_PYTHON_OPENVLA`、`EMBODIQ_PYTHON_VLA_ADAPTER`、
`EMBODIQ_PYTHON_FASTWAM` 指定，并在创建运行时固定。

每个运行包含：

```text
plan.json                   解析后的配置、路径、解释器与阶段依赖
run.json                    动态状态、尝试次数、产物哈希
input_identity.json         权重/配置、相关源码、输入及环境身份
episode_manifest.json       冻结回合清单副本
logs/                       每阶段日志
results/{method}/{model}/   本次闭环结果与回合断点
resource_bench/              本次固定输入专测与清单副本
report/report.json          机器可读汇总与证据链接
report/report.md            人可读对比
report/cells.csv             逐集合指标
report/tradeoffs.png         性能与资源对比图（另有 SVG）
```

启动和恢复会重算使用中的权重、归一化统计、相关源码、资源观测与初始状态文件的 SHA256，
并记录三模型中所选环境的包版本。因此首次启动有校验耗时。
已完成阶段的产物被修改时拒绝复用；计划被编辑时要求另建运行。
外部权重、源码和数据应在运行期间保持不变；仿真资产仍依赖已部署环境，这不是完整容器化复现包。

报告仅扫描当前运行目录，不读取历史原精度/RTN 目录，也不修改 `results/matrix/` 正式历史表。
正式档位复用既有准入检查，并验证新运行与冻结初始状态匹配。
资源专测另外核对计划哈希、模型路径、精度、动作块、FastWAM 参数、固定输入和测量次数。

显存分别保留加载期与稳态，延迟包括查询和动作步摊销推理耗时；后者不等于真机端到端响应时间。
当前资源输入来自成功演示的开发观测，不代表失败状态分布。

本轮真实 smoke 验证发现：单次查询可能短于旧默认 100 ms 采样间隔，使重置后的 NVML 峰值为 0。
准入检查会拒绝这种结果。现已在采样重置和报告边界补充同步读取，并让资源专测实际使用 100 Hz
稳态采样；加载期频率单独记录。NVML 不可用时仍保留缺失值并拒绝准入，不用 allocator 数值替代。

历史资源文件的 `nvml_sampler_hz=100` 是当时脚本声明的目标，而旧 `MemoryProbe` 实际默认 10 Hz；
这批历史峰值没有因本次修复而重新测量。正式历史矩阵保持原数据，新运行的资源结果独立保存。
工作进程同时固定模型 CUDA 与仿真 EGL 的物理 GPU，沿用本地离线权重/缓存，避免加载时访问模型源。

## 范围

`launch/resume-run` 是自动流程；旧 `run/execute/resume` 仍保留为原有单任务管理入口，
不能用旧 `execute` 操作新自动运行。第一版不提供任意新模型自动适配、AWQ/HSA 算法、
多 GPU 并行、RLark/RoboColiseum 接入或 Web 服务。
后续界面可以直接复用本次计划、状态和报告契约。

## 2026-10-09 实测验收

三模型 × 原精度/RTN，选择 Spatial 的 smoke 档位，通过一次 `launch` 完成
6 个真实闭环回合、6 个资源阶段及报告，共 13 个阶段。六回合均成功，
但每条件只有一个回合，不用于估计正式成功率或宣称模型性能改善。
六组专测均无争用、NVML 非零；每组只测一次查询，不能作为稳定延迟统计。

完成后实际调用 `resume-run`，重新核对输入和产物，13 个阶段的尝试次数全部保持 1，
证明没有重复运行已完成阶段。单元/集成测试另覆盖运行中停止、工作进程组清理、失败重试、
GPU 忙阻塞、继承锁防重复运行、篡改输入/产物拒绝，以及资源加载自定义权重路径。

- 主测试集：208 passed、23 skipped；FastWAM 环境定向检查：19 passed。存在重叠，不相加。
- 四集合正式配置：24 闭环 + 6 资源 + 1 报告的计划与路径预检通过，未重复执行 12,000 回合。
- 新报告：[`report.md`](../results/pipeline_validation_20261009/20261009_220944_ab8a066ab2/report/report.md)。
- 验收记录与日志：[`acceptance.json`](../results/framework_closeout_20261009/pipeline_v1/acceptance.json)。
- 版本固化目录：`releases/task14_pipeline_v1_20261009/`，源码与证据快照附逐文件和压缩包 SHA256。

首轮真实流程在 VLA-Adapter 原精度资源阶段被零 NVML 峰值门禁拒绝，该失败记录也保留。
修复测量后创建新运行，没有修改旧输入身份或强行复用旧阶段。

## 本地网页入口

在同一执行层之上已增加 `embodiq web`，支持任务选择、提交、状态/日志、停止/续跑及结果下载。参见[Web/API 工作台](web_workbench.md)。
