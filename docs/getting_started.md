# 安装与配置

当前交付包含可运行的本地工作台和评测执行代码。源码仓库不包含外部模型、仿真资源、演示训练数据或完整实验结果；只读演示可以在不安装模型的情况下使用。

## 1. 安装控制端

建议在 Linux 的 Python 3.10+ 环境，从仓库根目录执行：

```bash
python -m pip install -e '.[ctl]'
embodiq capabilities
embodiq launch configs/stage_fp16_rtn.json --profile formal --dry-run
embodiq web --port 8765
```

可用 `--port 0` 自动选择空闲端口。当前推荐源码可编辑安装：完整流程会读取仓库中的 `configs/` 和 `tools/`，不是脱离源码目录运行的独立 wheel 服务。

## 2. 准备模型与仿真环境

三个模型依赖栈不同，分别准备解释器，参考对应上游的版本要求与 `env/` 中已有脚本。

本机已验证的主要版本：OpenVLA / VLA-Adapter 使用 PyTorch 2.2 与 NumPy 1.26；FastWAM 使用 PyTorch 2.7 与 NumPy 2.2。不要强行安装到同一环境。`env/setup_*.sh` 是既有部署记录，需要先核对本地路径、环境名称和上游版本。

所有模型环境均应能导入源码包：

```bash
/path/to/model-env/bin/python -m pip install -e . --no-deps --no-build-isolation
```

将解释器配置为实际位置：

```bash
export EMBODIQ_PYTHON_OPENVLA=/path/to/openvla/bin/python
export EMBODIQ_PYTHON_VLA_ADAPTER=/path/to/vlaadapter/bin/python
export EMBODIQ_PYTHON_FASTWAM=/path/to/fastwam/bin/python
```

上游源码可放在一个共同目录下，包含 `OpenVLA/`、`VLA-Adapter/`、`FastWAM/` 和 `LIBERO/`。主配置支持路径变量：

```bash
export EMB_ROOT=/path/to/embodied-projects
export DATA_ROOT=/path/to/data
export OPENVLA_ROOT=/path/to/openvla-checkpoints
```

也可以复制 `configs/stage_fp16_rtn.json` 为被 Git 忽略的 `configs/my.local.json`，直接填写模型源码、权重与统计文件路径。控制端支持上述变量；部分历史 `env/` 脚本仍会设置本机默认值，使用前需调整脚本。

OpenVLA/VLA-Adapter 分集合选择兼容 checkpoint；FastWAM 还需要归一化统计文件。LIBERO 需安装 BDDL、初始状态文件、仿真资产和无头渲染依赖；通过 `embodiq doctor --family openvla|adapter|fastwam` 分别检查。

## 3. 固定评测输入

自动流程不仅检查路径存在，还要求输入清单及内容哈希匹配：

- 回合清单：当前流程从 `results/baseline_unified_v3_20260919/episode_manifest.json` 读取任务初始状态与种子身份。
- 资源专测清单：配置字段 `resource.fixed_input_manifest`，默认 `results/resource_bench/inputs_v1.json`，引用经过冻结和校验的观测文件。

这些实验输入没有打包在源码里。迁移时应准备与当前 LIBERO/模型协议一致的清单和观测；回合清单导出逻辑在 `tools/run_unified_baselines.py::export_assets`，观测准备见 `tools/prepare_all_calibration.py`，资源清单见 `tools/run_resource_bench.py --make-inputs`。这些工具中有原实验目录约定，应核对配置后使用。

若输入缺失，任务会明确进入阻塞状态；不能靠创建空 JSON 或替换为演示数据绕过预检。静态 `demo/` 的快照不属于可执行实验输入。

## 4. 提交、查看与恢复正式评测

```bash
embodiq launch configs/my.local.json --models vla_adapter --methods fp16 rtn \
  --suites libero_spatial --profile formal --gpu 0
embodiq status results/platform/<运行编号>
embodiq resume-run results/platform/<运行编号> --gpu 0
```

运行目录以实际命令打印的路径为准；CLI 未传 `--out-root` 时使用配置的输出目录，网页使用 `--runs-root`。

完整用法见[自动评测说明](automatic_evaluation.md)及[网页工作台](web_workbench.md)。原始精度/RTN 是当前可执行集合；AWQ/HSA 等待正式实现或通过验证的工件。

## 5. 只读演示与视频

```bash
python -m http.server 8000 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8000/demo/`。该模式不调用后端，不需要显卡，不允许创建/停止/恢复任务。结果来自真实完成运行的快照，去除了机器路径、文件哈希和原始日志。

可用 `tools/build_showcase.py <已完成正式运行目录> ...` 重新导出演示快照；导出前会验证保存的结果文件。当前随仓库发布的演示来自 `tools/report_matrix.py` 重新准入的正式历史矩阵，通过 `tools/build_showcase.py --matrix <矩阵文件>` 导出；历史汇总不伪造托管阶段进度。`tools/record_showcase.cjs` 使用额外的 Playwright 开发工具录制字幕演示，不是运行平台的必需依赖。
