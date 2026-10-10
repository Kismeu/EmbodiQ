# 一键式正式评测

当前开放 OpenVLA、VLA-Adapter、FastWAM，以及原精度基线和 RTN。
一次提交自动执行预检、闭环评测、资源专测、报告；AWQ/HSA 返回待接入错误，不会启动或回退。
网页与命令行复用本机顺序执行器；网页入口见[Web/API 工作台](web_workbench.md)。

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
embodiq capabilities

# 只展开正式计划，不创建运行、不占用 GPU
embodiq launch configs/stage_fp16_rtn.json --dry-run

# 一个模型、两方法、一个集合，共 1,000 回合
embodiq launch configs/stage_fp16_rtn.json \
  --models vla_adapter --methods fp16 rtn --suites libero_spatial --gpu 0
```

四个集合全部完成时才给出四集合平均成功率；子集报告为“所选集合平均”。若只选择 RTN，则相对原始精度的差值留白，不自动混入历史基线。

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

当前资源专测实际使用 100 Hz 采样，并在测量边界同步读取显存；NVML 不可用或峰值无效时拒绝报告准入。

历史资源文件的 `nvml_sampler_hz=100` 是当时脚本声明的目标，而旧 `MemoryProbe` 实际默认 10 Hz；
这批历史峰值没有因本次修复而重新测量。正式历史矩阵保持原数据，新运行的资源结果独立保存。
工作进程同时固定模型 CUDA 与仿真 EGL 的物理 GPU，沿用本地离线权重/缓存，避免加载时访问模型源。

## 范围与验证

`launch/resume-run` 为完整自动流程；`run/execute/resume` 保留为单任务管理入口，不能交叉操作两种运行目录。平台提供本地网页和 API，不包含任意模型自动适配、AWQ/HSA 正式算法、多 GPU 并行或跨机器调度。

当前所有新评测固定使用正式协议：每集合 10 项任务、每任务 50 回合。配置、命令行和网页都拒绝其他评测档位。旧记录及原始实验文件保留在本地，不会被改写为正式结果。

自动化测试覆盖计划规模、任务执行、正式结果准入、停止与恢复、输入及产物篡改拒绝。测试中的 CPU 模型替身不产生实际模型性能数据。

已有正式历史数据覆盖 24 个原始精度/RTN 组合，共 12,000 回合。当前只读交互演示由这些数据汇总生成；更新网页和文档不会重新执行这批实验。
