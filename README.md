# EmbodiQ

面向具身模型的量化适配与机器人任务评测工作台。选择模型、量化方法和 LIBERO 任务集合，自动执行任务评测、速度与显存测量，并生成可追溯的对比报告。

[只读交互演示](demo/index.html) · [演示视频](docs/media/embodiq-demo.mp4) · [安装与配置](docs/getting_started.md) · [工作台说明](docs/web_workbench.md) · [命令行流程](docs/automatic_evaluation.md)

![评测结果界面](docs/media/results.png)

## 当前支持

| 项目 | 状态 |
|---|---|
| 模型 | OpenVLA、VLA-Adapter、FastWAM |
| 可执行方法 | 原始精度（实际 BF16）、四位量化 RTN（128 个权重一组） |
| 待接入方法 | AWQ、HSA；不生成占位分数，不自动退回其他方法 |
| 任务集合 | LIBERO 空间关系、目标物体、任务目标、长时程操作 |
| 执行入口 | 中文 Web/API 与统一 CLI |
| 运行管理 | GPU 独占检查、状态与日志、停止、断点续跑 |
| 输出 | 任务成功率、集合平均、查询耗时、运行/加载峰值显存；JSON、Markdown、CSV、图片 |

FastWAM 的自动对比固定使用 first-frame 模式。量化算法代码中保留部分 AWQ/HSA 接口及开发实现，不表示其已完成正式验证。

## 无需显卡，先看演示

在仓库根目录运行：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

浏览器打开 `http://127.0.0.1:8000/demo/`。也可以直接打开 `demo/index.html`。

演示复用真实工作台界面，可选择配置、预览评测规模、浏览历史实验、切换任务集合、下载演示数据表。展示数据来自完成且通过结果文件校验的**快速检查**；不会执行模型，也不能替代正式性能结果。直接点击 GitHub 中的 HTML 文件只会查看源码，需要下载仓库后打开或通过静态服务器访问。

## 启动真实工作台

建议 Linux、Python 3.10+，从源码安装：

```bash
python -m pip install -e '.[ctl]'
embodiq web --config configs/stage_fp16_rtn.json --runs-root results/platform --port 8765
```

打开 `http://127.0.0.1:8765`。页面可先浏览和预览配置；**运行模型前**必须准备各模型独立环境、兼容权重、LIBERO 及固定输入清单，具体见[安装与配置](docs/getting_started.md)。仓库不包含模型权重、训练数据或完整历史实验文件。

环境准备完成后，可用 CLI 执行一次小规模检查：

```bash
embodiq launch configs/stage_fp16_rtn.json \
  --models vla_adapter --methods fp16 rtn \
  --suites libero_spatial --profile smoke --gpu 0
```

正式三模型 × 两方法 × 四集合为 31 个阶段、12,000 回合。`smoke` 对应页面“快速检查”；每个模型/方法/集合仅 1 回合。正式评测使用 `--profile formal`。

## 运行与结果的约定

- 同一个实验使用固定任务协议与输入，比较成功率及资源占用。
- 续跑校验模型、代码、环境、输入和已完成结果；条件改变时需新建实验。
- 仅在正式评测完成全部四个集合后给出四集合平均成功率。
- 查询耗时指一次模型调用，不能等同于机器人的端到端动作时延。
- 运行峰值显存来自显卡监测，加载峰值来自推理框架，两者分开报告。
- 本地服务不包含多用户认证、跨集群调度或任意模型上传。

已有本地正式实验覆盖原始精度/RTN 的 24 个模型—方法—集合组合、12,000 回合。展示用快速检查与正式实验严格分开。历史资源采样口径限制和详细记录见[项目进展说明](docs/README.md)；其中 `results/` 链接指向本地实验材料，未随代码仓库发布。

## 目录

```text
src/embodiq/         统一接口、模型适配、量化、调度、Web/API
configs/            实验配置与待接入方法模板
tests/              协议、量化、结果准入和控制器测试
tools/              资源测量、报告、数据准备与演示工具
env/                已有环境的安装、兼容性与启动脚本
demo/               可离线运行的只读交互演示
docs/               使用说明、设计记录与演示视频
```

`env/` 与部分历史工具保留实验机器上的默认路径，迁移前按文档检查并覆盖；不会自动下载所有模型并完成整套环境配置。

## 验证

```bash
# 全量测试需具备量化及各测试所需依赖
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q tests/
```

本机最近一次完整回归：227 通过、23 跳过；中文界面调整后的 Web/API 专项：19 通过。跳过项不能视为已验证的模型或 GPU 路径。演示视频来自只读展示，不新增正式实验。

## 上游项目

模型与仿真环境分别参见 [OpenVLA](https://github.com/openvla/openvla)、[VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter)、[FastWAM](https://github.com/yuantianyuan01/FastWAM)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。界面信息组织参考 [RLark](https://github.com/RLinf/RLark) 和 [Genie Sim / RoboColiseum](https://github.com/AgibotTech/genie_sim)。关联关系与授权范围见 [第三方说明](THIRD_PARTY.md)。
