<h1 align="center">EmbodiQ</h1>
<p align="center"><strong>具身模型量化与任务评测工作台</strong></p>
<p align="center">通过网页或命令行，对比量化前后的任务成功率、推理耗时与显存占用。</p>

<p align="center">
<a href="#安装">安装</a> · <a href="#模型与数据准备">模型与数据</a> · <a href="#配置">配置</a> · <a href="#运行">运行</a> · <a href="#演示">演示</a>
</p>

## Updates

- **[2026-10-10]** Initial release of code.
- ...

## 安装

使用 Linux、Python 3.10+；运行模型评测需要 NVIDIA GPU。

```bash
git clone https://github.com/Kismeu/embodiq.git
cd embodiq

conda create -n embodiq python=3.10 -y
conda activate embodiq
python -m pip install -e '.[ctl]'
```

## 模型与数据准备

按所选模型的官方说明安装推理环境，并下载对应的 LIBERO 权重：

| 模型 | 源码与安装说明 | 可用评测方法 |
| :--- | :--- | :--- |
| OpenVLA | [openvla/openvla](https://github.com/openvla/openvla) | 原始精度、RTN |
| VLA-Adapter | [OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) | 原始精度、RTN |
| FastWAM | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | 原始精度、RTN |

**更多模型、量化方法、仿真测试持续接入中...**

安装 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，准备仿真资产、初始状态与评测输入。具体步骤见[安装与配置](docs/getting_started.md)。

各模型使用独立的 Python 环境。在需要使用的模型环境中安装 EmbodiQ：

```bash
/path/to/model-env/bin/python -m pip install -e . --no-deps --no-build-isolation
```

## 配置

复制配置模板：

```bash
cp configs/stage_fp16_rtn.json configs/my.local.json
```

编辑 `configs/my.local.json`，填写实际路径：

| 配置项 | 内容 |
| :--- | :--- |
| `models.<模型>.repo_root` | 模型源码目录 |
| `models.<模型>.checkpoint_root` 或 `checkpoint` | 权重目录或文件 |
| `models.fastwam.dataset_stats` | FastWAM 的归一化统计文件 |
| `resource.fixed_input_manifest` | 资源测量输入清单 |
| `outputs.root` | 评测结果保存目录 |

指定所选模型的 Python 解释器，替换为对应环境中的实际路径：

```bash
export EMBODIQ_PYTHON_OPENVLA=/path/to/openvla/bin/python
export EMBODIQ_PYTHON_VLA_ADAPTER=/path/to/vlaadapter/bin/python
export EMBODIQ_PYTHON_FASTWAM=/path/to/fastwam/bin/python
```

配置字段说明见 [`configs/`](configs/README.md)。

## 运行

### 网页工作台

```bash
embodiq web --config configs/my.local.json --runs-root results/platform --port 8765
```

打开 <http://127.0.0.1:8765>，选择模型、量化方法、任务集合和显卡，提交评测。完成后可查看成功率折线图、耗时与显存对比，并导出报告、数据表和图表。

### 命令行

```bash
embodiq launch configs/my.local.json \
  --models vla_adapter \
  --methods fp16 rtn \
  --suites libero_spatial \
  --gpu 0
```

按需求替换参数；`--models`、`--methods` 和 `--suites` 均支持多个值。

| 参数 | 可选值 |
| :--- | :--- |
| `--models` | `openvla`、`vla_adapter`、`fastwam` |
| `--methods` | `fp16`（原始精度）、`rtn` |
| `--suites` | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` |
| `--gpu` | 显卡编号 |
| `--out-root` | 输出目录 |

`fp16` 是原始精度的配置名称，模型实际使用 BF16。AWQ、HSA 暂未接入。

查看或恢复任务，将 `RUN_DIR` 替换为启动命令输出的运行目录：

```bash
embodiq status RUN_DIR
embodiq resume-run RUN_DIR --gpu 0
```

更多命令见[命令行用法](docs/automatic_evaluation.md)，网页操作与接口见[工作台说明](docs/web_workbench.md)。

## 演示

[![EmbodiQ：配置评测、查看任务、对比结果](docs/media/embodiq-preview.gif)](docs/media/embodiq-demo.mp4)

<p align="center"><a href="docs/media/embodiq-demo.mp4">观看完整视频</a></p>

无需安装模型即可体验交互演示。在仓库根目录运行：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000/demo/>。演示使用已完成的评测数据，不启动模型。

## 致谢

感谢 OpenVLA、VLA-Adapter、FastWAM 和 LIBERO 提供模型与仿真评测基础。相关授权见[第三方说明](THIRD_PARTY.md)。
