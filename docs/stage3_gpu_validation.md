# GPU 验证记录（2026-09-18）

## 下载核查

后台镜像下载已经结束：OpenVLA Goal、Long、Wan tokenizer、原始 VAE 和 UMT5 编码器均标记 downloaded。原转换权重仓库的 401 失败记录仍保留，但所需组件由 Wan 官方原始格式替代，不是当前缺件。FastWAM 使用 `model.redirect_common_files=false`。

OpenVLA Goal/Long 每套四分片齐全，safetensors 可读取，共 982 个 tensor key，配套 dataset_statistics.json 存在。此为结构检查，不等于四集合闭环完成。

所有 GPU 验证启用 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1；FastWAM 另启用 DIFFSYNTH_SKIP_DOWNLOAD=true。未在推理期间下载模型。

## OpenVLA 输出一致性

- 测试：`tests/test_parity_openvla.py`。
- 物理 GPU：1（进程逻辑编号 0）。
- checkpoint：`/data/jyc/models/openvla-7b-finetuned-libero-spatial`。
- 结果：1 passed，271.18 秒；24 条依赖弃用/配置警告。
- 取已有 LIBERO 原始观测 fixture 的前三条，忽略其中 VLA-Adapter 动作标签；分别用 OpenVLA 原生图像处理/动作解码/夹爪转换路径和新适配器计算动作，固定随机种子，验证三个 `(1,7)` 输出逐元素一致。
- 限制：仅三个固定输入、同一 Spatial checkpoint 的数值路径验证；不表示完整任务成功率或全部模型精度版本已验证。

## 闭环运行

GPU 0 有约 0.73 GiB 的其他进程驻留，统一入口已标记 contended=true。本轮为功能验证，其资源数字不进入最终比较主表。

## 本轮完成与剩余工作

完成：所有缺失模型下载；OpenVLA 新适配器固定输入数值一致性及 Object 闭环；FastWAM 本地权重加载和原生闭环。三个模型现在均有原精度运行证据，但证据规模和协议不同。

未完成：FastWAM 接入统一 Policy/runner；三模型公共协议四集合基线；AWQ/HSA 完整接入；12 配置/48 单元正式比较。FastWAM 环境仍是 MuJoCo 3.1.6，另外两个环境为 3.3.2，正式比较前需要统一并重验，不能把本次不同环境的小样本结果混入主表。

本轮启动的三个 GPU 验证进程均已结束；没有留下未交代的推理后台任务。下一步优先将已跑通的 FastWAM 原生推理封装为统一适配器，完成原生与适配器一致性验证，再冻结公共协议。
