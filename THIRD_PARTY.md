# 第三方项目与发布范围

EmbodiQ 通过适配器调用 OpenVLA、VLA-Adapter、FastWAM 和 LIBERO。模型权重、仿真数据和独立上游代码仓库不在本次代码发布内容中，获取及使用它们时需遵循各自的许可证和权重条款。

- OpenVLA：模型推理与 Prismatic 接口。
- VLA-Adapter：动作预测接口及本地兼容处理。
- FastWAM：动作序列与时序预测接口。
- LIBERO / robosuite / MuJoCo：机器人操作任务与仿真环境。
- PyTorch：模型运行、张量运算与量化内核。

RLark 与 RoboColiseum / Genie Sim 仅作为任务管理及评测展示的设计参考。本项目未复制其网页品牌资产，也未接入其云端服务。不得将它们报道中的启动速度或仿真—真机相关性当作本项目的实测指标。

`experiments/` 中下载的 AWQ 上游及历史快照不随发布上传。`src/embodiq/quant/` 的开发接口保留自身方法身份和准入检查，不能将待接入方法描述为可用算法。

当前仓库尚未选定面向外部再分发的整体开源许可证；发布到 GitHub 本身不代表重新授权外部项目或模型。许可证选定后可补充根目录 LICENSE。
