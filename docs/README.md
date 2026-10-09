# embodiq：任务 1.4 统一适配与评测框架

最后更新：2026-10-09，框架收尾完成。AWQ/HSA 按用户决定暂时留白。

## 当前状态

**三模型统一适配、原精度/RTN 正式评测和资源汇总可用，FastWAM IDM 已完成真实运行验证。**
这不表示 AWQ/HSA 算法已完成，也不表示四方法 48 单元全量实验已完成。

| 交付项 | 状态 |
|---|---|
| OpenVLA / VLA-Adapter / FastWAM 图像、语言、状态、动作适配 | 已实现，原精度/RTN 完整闭环验证 |
| FastWAM 时序接口 | 修复 IDM 配置选择；真实未来视频去噪前向通过；Spatial task 0 单回合成功（70 步、7 次查询） |
| 统一评测入口 | 四种方法共用入口；AWQ/HSA 必须提供合法工件 |
| HSA 批量调度 | 工件路径解析、缺失预检和 rho 参数传递已补齐；算法仍待交付 |
| 原精度 / RTN 正式成功率 | 24/48 单元，12,000 回合，零错误/INVALID |
| 固定输入资源专测 | 6/12 配置，每配置 600 次测量；已接入最终 JSON/Markdown/CSV |
| 正式结果准入 | 拒绝开发工件、不同实现/范围/协议混用、错 checkpoint、重复或非冻结初始状态、失配资源结果 |
| 版本固化 | 可校验源码与证据归档，含三环境包版本、上游源码和历史实验源代码 |
| AWQ / HSA 正式结果 | 留白；AWQ 已尝试未通过数值门禁，HSA 待上游 |

结果见 [正式矩阵](../results/matrix/matrix.md)，图见 [性能—资源对比](../results/matrix/tradeoffs.png)。
验证与归档说明见 [收尾报告](framework_closeout_20261009.md)。

这里的“框架收尾”指评测底座和既有实验交付。后续新增的 `launch` 已连接本机顺序调度、
资源专测和按运行生成报告；用法、验证档位与限制见[一键式评测](automatic_evaluation.md)。
旧 `run` 仍只创建计划。现已增加[本地 Web/API 工作台](web_workbench.md)，支持页面提交、停止/续跑、日志和报告；跨集群调度尚未实现。
RLark / RoboColiseum 的前期调研及规划见[平台方案](platform_research_20261009.md)。

## 结果解释

| 模型 | 原精度宏平均 | RTN 宏平均 | 变化 |
|---|---:|---:|---:|
| OpenVLA | 76.95% | 73.95% | −3.00 pp |
| VLA-Adapter Pro | 97.55% | 93.75% | −3.80 pp |
| FastWAM first-frame | 96.80% | 96.85% | +0.05 pp |

当前 RTN 节省稳态显存，但查询更慢。+0.05 pp 不作为性能提升证据。
原精度配置名虽为 fp16，实际模型 dtype 为 BF16。
主矩阵固定 FastWAM first-frame；IDM 属于接口补充验证，不混入四集合成功率。

显存区分加载期与稳态；主表资源只取固定输入专测，缺失时留白。
历史闭环 schema/2 的 NVML/reserved 峰值混有加载期缓存，不能作为部署峰值对比。
查询耗时与动作步摊销耗时均保留；后者不是机器人端到端控制响应时间。
历史专测绑定 checkpoint 路径、量化规格和策略参数，未记录 checkpoint 文件哈希，限制随表保留。
后续一键流程实测还发现历史脚本声明的 100 Hz 与当时实际默认 10 Hz NVML 采样不一致；
新专测已修正采样频率并补充边界读取，历史资源数字未重测，详见[自动评测说明](automatic_evaluation.md)。

## 常用命令

```bash
# 本地网页入口：选择模型、方法和任务后自动执行
PYTHONPATH=src /data/jyc/tools/anaconda3/envs/openvla/bin/python -m embodiq.ctl.cli web --port 8765

# 新的一键入口：三模型、原精度/RTN、四集合；GPU 索引按实际资源填写
PYTHONPATH=src python -m embodiq.ctl.cli launch configs/stage_fp16_rtn.json --gpu 3

# 当前默认只规划原精度 / RTN；不启动待交付方法
python -m embodiq.ctl.cli --help

# 重建正式表；完整性问题返回非零，AWQ/HSA 缺失不补零
python tools/report_matrix.py

# 从同一 matrix.json 生成图片
/data/jyc/tools/anaconda3/envs/openvla/bin/python tools/plot_tradeoffs.py

# 统一入口（任务与回合规模由参数明确指定）
python -m embodiq.entry.evaluate --model openvla \
  --checkpoint /data/jyc/models/openvla-7b-finetuned-libero-spatial \
  --suite libero_spatial --protocol unified.v3 --quant rtn \
  --trials 50 --gpu 0 --out results/new_run/openvla

# 无 GPU 回归
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /data/jyc/tools/anaconda3/envs/openvla/bin/python -m pytest -q tests/
```

批量配置用 `configs/stage_fp16_rtn.json`。`stage_awq.json` 等仅为后续接入模板，
当前没有合格工件，不应启动。三个模型分别使用 openvla / vlaadapter / fastwam 环境。

## 文档索引

- [框架收尾报告](framework_closeout_20261009.md)：变更、真实验证、测试、归档与复跑。
- [平台化调研与方案](platform_research_20261009.md)：RLark / RoboColiseum 可借鉴能力、现状与验收路径。
- [一键式评测](automatic_evaluation.md)：本机自动执行、停止、恢复及本次报告。
- [量化工件契约](quant_artifact_contract.md)：正式工件格式、范围、来源与准入标志。
- [接口契约](interface_contract.md)：图像、语言、状态、动作、时序和计时语义。
- [实验协议](experiment_protocol.md)：原始设计与后续修订；现状以本文件为准。
- [AWQ 开源尝试](awq_public_attempt_20261009.md)：失败证据及适配边界。
- [收尾前盘点](task14_status_20261009.md)：保留当时状态；其缺口已按收尾报告更新。
- [旧入口快照](archive/README_before_closeout_20261009.md)：历史记录，不用于判断现状。
