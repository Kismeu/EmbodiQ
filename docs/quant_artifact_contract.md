# 量化工件接入契约（AWQ / HSA → 统一评测入口）

面向 1.2（AWQ）与 1.3（HSA）的交付方。1.4 是评测方：算法在你们那边产出，
评测入口只通过本文件定义的格式接收，并在加载前做完全部校验。

- 实现：`src/embodiq/quant/artifact.py`
- 契约测试：`tests/test_quant_artifact.py`（33 项，CPU、无需 checkpoint）
- 格式标识：`embodiq/quant-artifact/1`

## 1. 为什么是 fail-closed

评测入口对 `--quant awq` / `--quant hsa` **强制要求工件**，没有工件直接报错，
不会退回 RTN。原因是一次静默退回会产出一整套"看起来合理、标签是 AWQ、
实际是 RTN"的 48 单元结果，而这种错误在成功率表上看不出来。

以下四类错误各自对应一组校验，全部在替换任何模块**之前**完成；
任一失败则模型保持原样，不会出现半量化状态：

| 风险 | 校验 |
|---|---|
| 换标签的 RTN | AWQ 工件必须至少有一层带 `input_scale`；HSA 工件必须至少有一层带残差 |
| 模块范围不一致 | 工件层集合必须**恰好等于**该模型计划中 `decision == "int4"` 的层集合 |
| 换了 checkpoint | 每层记录 `weight`/`bias` 的 SHA256，与实际加载的张量逐一比对 |
| 半量化 | 全部校验通过后才构造和替换模块 |

## 2. 文件格式

`torch.save` 一个普通 dict，必须能以 `weights_only=True` 加载
（工件是数据，不得可执行）。

```python
{
  "format": "embodiq/quant-artifact/1",
  "method": "awq" | "hsa",              # 必须与运行的 --quant 一致
  "formal": True,                        # 正式结果准入必需；开发工件为 False
  "validation_only": False,              # 接口测试工件必须为 True
  "implementation": "awq_block_v1",      # 自定义标识，写入结果 JSON
  "implementation_version": "1.0",
  "producer_code_sha256": "<hex>",       # 产出该工件的代码哈希
  "model_family": "openvla" | "vla_adapter" | "fastwam",
  "suite": "libero_spatial",             # 可选，建议填
  "w_bit": 4,
  "group_size": 128,
  "source_checkpoint_sha256": "<hex>",   # 源 checkpoint 输入摘要
  "calibration": {
      "manifest_sha256": "<hex>",        # 必填：无校准清单的行不可复现
      "n_calibration": 1280,
      "n_development": 320,
      "source": "libero training demos",
  },
  "scope": ["llm"],                      # 必须等于运行 scope，见第 3 节
  "layers": {
      "<module name>": {
          "weight_sha256": "<hex>",      # 必填
          "bias_sha256": "<hex>" | None, # 必填（无 bias 时为 None）
          # --- AWQ ---
          "input_scale": Tensor[in_features],              # 严格正、有限
          "clip_max":    Tensor[out_features, in//g] | None,  # >= 0、有限
          # --- HSA ---
          "residual_idx": Tensor[k]  (int16/int32/int64),  # 无重复、[0, in)
          "residual_val": Tensor[out_features, k],         # 有限
      },
      ...
  },
}
```

执行侧已就绪：`QuantLinear` 的融合算子
`embodiq.w4a16(x, packed, scales_zeros, s_inv, res_idx, res_val, bias, g, out)`
同时消费输入缩放与残差，`clip_max` 在打包前对缩放域权重做逐组截断。
无需你们提供 kernel。

`input_scale` 按约定**保留在模块内**（`QuantSpec.explicit_input_scale=True`），
不折叠进前一个算子：FastWAM 的 norm 是 `elementwise_affine=False`，
Qwen2.5-0.5B 是 GQA(14/2) 使 `o_proj` 无法折进 `v_proj`，折叠目标基本不存在。

## 3. 每模型的模块范围（必须沿用）

协议 §6 要求 RTN/AWQ/HSA 共用同一份每模型清单，否则 48 单元表变成
不同范围最优结果的拼接。范围定义在 `run_adapter.py` 的 `MODEL_SCOPES`：

| `model_family` | `scope` | 工件须覆盖的层数 |
|---|---|---:|
| `openvla` | `["llm"]` | 224 |
| `vla_adapter` | `["llm","vision_dino","vision_siglip","projector","action_head"]` | 568 |
| `fastwam` | `["video_dit","action_dit"]` | 600 |

**层名清单已导出,不要自己推导。**

```bash
python tools/export_scope_manifest.py          # -> artifacts/scope_manifest.json
```

该文件对每个模型给出:

- `artifact_must_cover`：工件必须覆盖的**精确层名列表**
- `artifact_must_exclude_int8_fallback`：必须排除的 INT8 回退层
- `int4_sha256`：层名列表的哈希,可用来确认双方看的是同一份
- `scope`、`n_int4`、`n_int8_fallback`、`quantized_param_fraction`、`compression_in_scope`

清单由已完成 RTN 运行的逐层报告导出,与评测器在加载时重算的计划同源。
导出时会校验两件事并在 `problems` 中报告:层集合在该模型四个集合间**完全一致**
(已验证,2026-10-04),以及计数与本文档公布的数字一致。
`tests/test_scope_manifest.py` 的 13 项测试把文档、工具与真实运行记录钉在一起——
文档与计划一旦漂移,会在你们交付之前就失败,而不是在工件到达时才被拒。

两点容易踩的：

- **VLA-Adapter 另有 28 个层回退 INT8**（`out_features` 不满足 kernel 对齐，
  如 SigLIP 的 4304）。这些层**不要**放进工件：逐输出通道 INT8 既不接受
  分组输入缩放也不接受分组截断。工件层集合 = 568 个 int4 层。
- **FastWAM 的模块别名**：`named_modules()` 先暴露 `video_expert.*` /
  `action_expert.*`，不是旧规则里的 `mot.mixtures.*`。按前者取名，
  共享专家去重后只应出现一次。

## 4. 方法身份约束

| 方法 | 必须有 | 必须没有 |
|---|---|---|
| `awq` | 至少一层 `input_scale` | 任何残差（带残差即为 HSA） |
| `hsa` | 至少一层残差 | — |

HSA 的主分支按 1.3 的定义应当是 **AWQ 低比特主干**，因此 HSA 工件通常
同时携带 `input_scale` 与残差。只带缩放不带残差的工件会被当作
"AWQ/RTN 主干"拒绝。

HSA 残差预算：每层 `residual_idx.numel() / in_features` 不得超过
`HSASpec.max_per_layer_rho`（默认 0.08）。运行时的全局 rho 由
`--hsa-rho` 指定（默认 0.01）。残差字节、索引字节、缩放字节
**全部计入方法开销**并进入显存列——固定基础位宽不等于相同总预算。

## 5. 结果 JSON 里会记录什么

通过校验后，以下内容写入每个结果文件的 `quant` 字段，用于可追溯性：

```
quant.method                      awq | hsa
quant.implementation              你填的实现标识
quant.implementation_version      你填的版本
quant.scope                       运行范围
quant.artifact.artifact_sha256    工件文件自身哈希
quant.artifact.producer_code_sha256
quant.artifact.source_checkpoint_sha256
quant.artifact.calibration        校准清单哈希与规模
quant.artifact.residual_bytes / index_bytes / input_scale_bytes
quant.artifact.method_overhead_bytes
quant.artifact.n_layers_with_input_scale / _clip / _residual
```

不同实现的结果不汇总成同一行。一轮正式比较须锁定一个实现并覆盖
三模型四集合；其他实现列为独立实验版本。

2026-10-09 收尾：以上限制已落实到 `core/admission.py` 和 `tools/report_matrix.py`。
执行器允许开发工件做诊断，但正式汇总要求 `formal=true`、`validation_only=false`、
真实校准清单与样本数、正确模型/集合/完整 scope；缺少标志也不自动升级为正式。
同一模型/方法的实现版本、策略和协议必须一致；同集合内 checkpoint 与工件哈希必须一致。
元数据将两个准入标志写入结果 JSON。历史测试工件即使被误放入正式目录也会被拒绝。

批量配置同时支持 AWQ/HSA 的 `artifact` 单文件或 `artifact_root` 按模型/集合解析，
HSA 的 `rho` 会传递为 `--hsa-rho`；缺工件在启动模型前阻断。当前默认仍只运行原精度/RTN。

## 6. 交付验收（协议 §6.1）

工件交来后，评测方按顺序执行，任一项不通过则不进入正式评测：

1. **保存/加载一致性**：`QuantArtifact.load().bind()` 全量校验通过。
2. **固定输入输出检查**：在开发集 320 条观测上，量化后动作块与搜索期
   记录一致（`core/fidelity.py` 的 `ActionFidelity`）。
3. **三模型闭环冒烟**：Spatial task 0、2 回合、unified.v3、seed=42，零错误。
4. **真实显存与计时路径检查**：确认走的是 packed int4 算子而非
   `SimQuantLinear` 伪量化（后者无显存收益，不能作为部署指标）。

## 7. 调用方式

```bash
python -m embodiq.entry.evaluate \
  --model vla_adapter --suite libero_goal \
  --checkpoint /data/jyc/data/VLA-Adapter/LIBERO-Goal-Pro \
  --protocol unified.v3 --trials 50 --gpu 2 \
  --quant hsa --quant-artifact /path/to/hsa_vla_adapter_goal.pt \
  --hsa-rho 0.01 \
  --out results/stage7_hsa_full/vla_adapter
```

目录约定（`tools/report_matrix.py` 的 `METHOD_SOURCES` 已按此读取）：

- AWQ → `results/stage6_awq_full/<model>/<suite>/`
- HSA → `results/stage7_hsa_full/<model>/<suite>/`

## 8. 自检清单

交付前请确认：

- [ ] `weights_only=True` 可加载，无自定义类
- [ ] `format` / `method` / `implementation` / `implementation_version` /
      `producer_code_sha256` / `source_checkpoint_sha256` 齐全且非空
- [ ] `calibration.manifest_sha256` 已填
- [ ] `scope` 与第 3 节表格一致
- [ ] 层集合恰好等于该模型的 int4 层（openvla 224 / vla_adapter 568 / fastwam 600）
- [ ] 未包含 VLA-Adapter 的 28 个 INT8 回退层
- [ ] 每层 `weight_sha256` / `bias_sha256` 取自将要评测的那个 checkpoint
- [ ] `input_scale` 严格正、形状 `(in_features,)`
- [ ] `clip_max` 形状 `(out_features, in_features // 128)`、非负
- [ ] `residual_idx` 整数、无重复、在 `[0, in_features)` 内
- [ ] `residual_val` 形状 `(out_features, k)`、有限
- [ ] HSA 每层残差占比不超过 `max_per_layer_rho`
- [ ] AWQ 工件不含残差

## 当前正式工件范围（2026-10-08）

由 `tools/export_scope_manifest.py` 根据当前适配器和量化范围生成：

- OpenVLA：224 个 INT4 层，范围 `llm`；
- VLA-Adapter：568 个 INT4 层、28 个 INT8 fallback 层，范围 `llm + vision_dino + vision_siglip + projector + action_head`；
- FastWAM：600 个 INT4 层，范围 `video_dit + action_dit`。

AWQ/HSA 工件必须覆盖对应模型的全部 INT4 层，不能只提交单个 block 的 pilot。当前 `artifacts/awq_public/vla_adapter_spatial_task0_pilot` 只覆盖第一个语言 block，因此仍属于开发验证材料，不能进入正式矩阵。
