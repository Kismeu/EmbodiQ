# 使用指南

[返回首页](../README.md) · [评测说明](evaluation.md)

## 1. 浏览演示

下载仓库，在仓库根目录运行：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

浏览器打开 <http://127.0.0.1:8000/demo/>。演示支持配置预览、结果筛选与数据下载，不启动模型。也可以直接打开本地 `demo/index.html`；GitHub 中的 HTML 文件页面只显示源码。

## 2. 准备评测环境

真实评测需要 Linux、Python 3.10+、NVIDIA 显卡，以及以下资源：

| 资源 | 准备内容 |
| :--- | :--- |
| 模型 | [OpenVLA](https://github.com/openvla/openvla)、[VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter)、[FastWAM](https://github.com/yuantianyuan01/FastWAM) 的源码与兼容权重 |
| 运行环境 | 按各模型上游要求分别准备 Python 环境 |
| 仿真环境 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 的任务定义、仿真资产、初始状态与无头渲染依赖 |
| 评测输入 | 固定的回合清单和资源测量观测，用于保证比较条件一致 |

OpenVLA、VLA-Adapter 需使用对应任务集合的权重；FastWAM 还需配套的归一化统计文件。模型权重、仿真数据与完整评测输入不包含在仓库中。

<details>
<summary>部署人员：配置模型路径与评测输入</summary>

复制主配置，并填写当前机器上的实际路径：

```bash
cp configs/stage_fp16_rtn.json configs/my.local.json
```

主要配置项：

| 配置项 | 含义 |
| :--- | :--- |
| `models.*.repo_root` | 模型源码目录 |
| `models.*.checkpoint_root` / `checkpoint` | 权重目录或文件 |
| `models.fastwam.dataset_stats` | FastWAM 归一化统计文件 |
| `resource.fixed_input_manifest` | 资源测量观测清单 |

用下列变量指定各模型的 Python 解释器，替换为实际路径：

```bash
export EMBODIQ_PYTHON_OPENVLA=/path/to/openvla/bin/python
export EMBODIQ_PYTHON_VLA_ADAPTER=/path/to/vlaadapter/bin/python
export EMBODIQ_PYTHON_FASTWAM=/path/to/fastwam/bin/python
```

各模型环境均需能导入本项目，在相应环境中执行：

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

回合清单当前从 `results/baseline_unified_v3_20260919/episode_manifest.json` 读取；资源观测清单由配置指定。请使用与任务协议匹配的有效输入，不可用空文件代替。回合清单导出函数见 `tools/run_unified_baselines.py` 的 `export_assets`；观测准备工具为 `tools/prepare_all_calibration.py`，资源清单由 `tools/run_resource_bench.py --make-inputs` 生成。运行这些工具前需检查其中的模型和数据路径。

`env/` 脚本包含部署环境约定，迁移到新机器时需调整路径。完整模型环境与数据准备仍需手动完成。

</details>

## 3. 启动工作台

在仓库根目录安装控制端并启动服务：

```bash
python -m pip install -e '.[ctl]'
embodiq web --config configs/my.local.json --runs-root results/platform --port 8765
```

浏览器打开 <http://127.0.0.1:8765>。若主配置已适配当前机器，可直接使用 `configs/stage_fp16_rtn.json`。

## 4. 提交与查看评测

1. 在“新建评测”选择模型、量化方法、任务集合及显卡编号。
2. 核对预计回合数后，点击“开始评测”。
3. 在“任务管理”查看状态；需要中断时点击“停止评测”，之后可继续运行。
4. 完成后，在“评测结果”点击“读取结果”，按任务集合查看并下载报告。

显卡忙碌或资源缺失时，页面会提示原因。修改模型、评测输入或程序后，应新建评测。

<details>
<summary>命令行用法</summary>

```bash
# 仅查看完整计划，不启动模型
embodiq launch configs/my.local.json --dry-run

# 完整评测：三模型、原始精度与 RTN、四个任务集合
embodiq launch configs/my.local.json --gpu 0

# 查看状态或恢复评测，将路径替换为命令输出的运行目录
embodiq status /path/to/run
embodiq resume-run /path/to/run --gpu 0
```

</details>
