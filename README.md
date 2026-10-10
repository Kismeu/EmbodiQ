<h1 align="center">EmbodiQ</h1>
<p align="center"><strong>具身模型量化与任务评测平台</strong></p>
<p align="center">在同一套机器人操作任务中，对比模型量化前后的任务表现、响应速度与显存占用。</p>

<p align="center">
<a href="#平台演示">观看演示</a> · <a href="#支持范围">支持范围</a> · <a href="docs/getting_started.md">使用指南</a> · <a href="docs/evaluation.md">评测说明与结果</a>
</p>

## 平台演示

[![EmbodiQ 操作演示，点击查看完整视频](docs/media/embodiq-preview.gif)](docs/media/embodiq-demo.mp4)

<p align="center"><a href="docs/media/embodiq-demo.mp4"><strong>观看完整视频 · 约 1 分钟</strong></a></p>

| 配置评测 | 管理任务 | 查看结果 |
| :--- | :--- | :--- |
| 选择模型、量化方法和任务集合 | 查看执行状态，支持停止与继续运行 | 对比成功率、查询耗时和显存，导出报告 |

演示展示已完成的 **12,000 回合**评测结果，可浏览配置、切换任务集合和下载数据。演示模式不会启动模型。

## 支持范围

| 模型 | 原始精度 | RTN 四位量化 |
| :--- | :---: | :---: |
| [OpenVLA](https://github.com/openvla/openvla) | 支持 | 支持 |
| [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) | 支持 | 支持 |
| [FastWAM](https://github.com/yuantianyuan01/FastWAM) | 支持 | 支持 |

原始精度采用 BF16；RTN 将所选权重压缩为 4 位。AWQ、HSA 当前不可用。

评测采用 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 机器人仿真任务，覆盖四类能力：

| 空间关系 | 目标物体 | 任务目标 | 长时程操作 |
| :--- | :--- | :--- | :--- |
| 理解物体之间的位置关系 | 识别并操作指定物体 | 根据指令完成指定目标 | 连续完成多步操作 |

每个任务集合包含 10 项任务，每项运行 50 回合。完整的三模型、两方法、四集合评测共 **12,000 回合**。指标含义及结果见[评测说明](docs/evaluation.md)。

## 开始使用

**体验演示**：下载仓库，在根目录执行以下命令，再打开 <http://127.0.0.1:8000/demo/>。无需显卡，也无需安装模型。

```bash
python -m http.server 8000 --bind 127.0.0.1
```

**运行评测**：按[使用指南](docs/getting_started.md)准备模型、仿真环境和评测输入，再启动本地工作台。

模型权重与仿真数据需另行获取。第三方项目及使用条款见[第三方说明](THIRD_PARTY.md)。
