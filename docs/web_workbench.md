# 中文评测工作台

通过同一个网页选择模型、量化方法和 LIBERO 任务集合，完成正式评测、进度管理与结果对比。当前支持 OpenVLA、VLA-Adapter、FastWAM 的原始精度及 RTN；AWQ、HSA 保留为不可选的待接入项。

## 启动

完成[安装与配置](getting_started.md)后，在仓库根目录运行：

```bash
embodiq web --config configs/stage_fp16_rtn.json --runs-root results/platform --port 8765
```

打开 `http://127.0.0.1:8765`。服务默认仅监听本机，控制器独立于网页进程；关闭浏览器或重启 HTTP 服务不会自动结束已提交任务。

## 使用流程

1. **新建评测**：选择模型、量化方法和任务集合，填写显卡编号与忙碌时等待秒数。
2. **核对规模**：每个“模型 × 方法 × 任务集合”包含 10 项任务、每项 50 回合，共 500 回合。默认选择三模型、两方法及四集合，共 31 个阶段、12,000 回合。
3. **任务管理**：查看状态和日志；停止后可以恢复，已通过校验的完成阶段不会重复运行。
4. **评测结果**：选择已完成实验并读取结果，按集合比较成功率、查询耗时与显存，下载报告、数据表及原始对比图。

GPU 默认编号为 0，请根据机器实际空闲情况选择。显卡忙碌时任务等待或阻塞，不会终止其他实验。FastWAM 固定使用首帧模式。未满足正式评测协议的旧记录不在当前列表展示；原始文件保留在本地。

界面采用“新建评测 / 任务管理 / 评测结果”三页结构。运行编号、原始日志和技术限制可展开查看；图表使用毫秒与 GiB。汇总显存为所选集合峰值的最大值，查询耗时为各集合平均值的均值。

## API

`GET /api/capabilities` 返回模型、方法、任务集合及当前会话令牌。变更请求需携带 `X-Embodiq-Token`，服务同时校验 Host 与 Origin。

| 请求 | 用途 |
|---|---|
| `GET /api/capabilities` | 查询可用能力与会话令牌 |
| `POST /api/preview` | 预览规模，不创建任务 |
| `GET /api/runs` | 列出正式评测记录 |
| `POST /api/runs` | 创建并启动评测 |
| `GET /api/runs/{id}` | 查看运行状态 |
| `GET /api/runs/{id}/logs?stage={阶段编号}` | 查看整体或分阶段日志 |
| `POST /api/runs/{id}/cancel` | 停止评测 |
| `POST /api/runs/{id}/resume` | 恢复评测 |
| `GET /api/runs/{id}/report` | 读取通过完整性检查的报告 |
| `GET /api/runs/{id}/report/{文件名}` | 下载 Markdown、CSV、PNG 或 SVG |

预览或提交请求示例：

```json
{
  "models": ["vla_adapter"],
  "methods": ["fp16", "rtn"],
  "suites": ["libero_spatial"],
  "profile": "formal",
  "gpu": 0,
  "wait_seconds": 0
}
```

`profile` 可省略，固定使用 `formal`。上述请求为 1,000 回合、5 个阶段。恢复请求仅允许指定 `gpu` 与 `wait_seconds`。

报告需运行完成，且计划和各阶段产物的哈希保持一致；不完整或被修改的报告会被拒绝。只有四集合全部完成，才给出四集合平均成功率。查询耗时不等于机器人端到端动作时延，运行与加载峰值显存分别报告。

## 只读演示

`demo/` 使用相同界面，支持本地离线预览与报告下载，不允许提交或修改任务。当前快照来自通过准入检查的正式实验历史矩阵，共 24 个组合、12,000 回合。

从配置页选择模型、方法和任务集合，点击“查看本配置示例”，进入对应的任务摘要；再点击“下一步：对比结果”，查看同一配置的结果。历史汇总没有托管运行的逐阶段进度，演示以任务评测、资源测量和报告摘要说明范围，不构造执行日志。报告保留历史资源采样频率和权重身份记录的限制。当前控制器的完整性校验与历史数据可追溯程度不能混为一谈。

## 开发验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q tests/test_web.py tests/test_pipeline.py
```

这些测试使用 CPU 模型替身验证 HTTP、控制器生命周期、正式报告准入、停止与恢复、独占锁及篡改拒绝，不生成真实模型的性能结论。

界面信息组织参考 RLark 与 Genie Sim / RoboColiseum，关联范围见[第三方说明](../THIRD_PARTY.md)。
