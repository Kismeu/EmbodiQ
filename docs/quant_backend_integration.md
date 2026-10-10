# 量化后端接入说明

本文档定义后续 RTN、AWQ、HSA 代码接入 `embodiq` 时必须遵循的边界。
目标是让量化实现只负责量化，让模型适配器和 LIBERO 评测入口保持不变。

## 运行边界

统一入口的顺序固定为：

```text
构造 adapter
  -> adapter.load() 加载原始 checkpoint
  -> adapter.quantization_roots() 暴露命名模块根
  -> quantizer 在这些根上执行替换或加载工件
  -> adapter.act() 和统一 rollout
```

量化后端不得在 checkpoint 加载前替换模块。packed 模块会改变参数名称和
状态字典结构，提前替换会导致原始 checkpoint 的权重被静默跳过。

## Adapter 侧接口

三个 adapter 都实现：

```python
def quantization_roots(self) -> dict[str, object]:
    ...
```

返回值中的 key 是量化范围和工件层名的一部分：

| 模型 | 根节点 |
|---|---|
| OpenVLA | `model` |
| VLA-Adapter | `model`, `action_head`, `proprio_projector` |
| FastWAM | `model` |

VLA-Adapter 的两个辅助模块不能由量化后端自行从 adapter 属性中查找，必须
通过这个映射传入。未加载模型时调用该方法必须失败；这样可以防止对空对象
生成看似有效的量化报告。

## Quantizer 侧接口

当前评测入口使用一个可调用对象：

```python
QuantApply = Callable[[Mapping[str, object], str], QuantReport]
```

调用方传入 `roots` 和模型族名，后端返回 `QuantReport`。后端需要满足：

1. 只访问传入的根节点，不导入具体模型仓库的 adapter 类；
2. 使用 `QuantSpec` 中的位宽、分组大小、kernel 和 HSA 预算；
3. 为每个 Linear 记录量化、回退或保留原因；
4. 在任何模块替换前完成工件和权重哈希校验；
5. 替换后确认目标范围内没有未处理的 Linear；
6. 把实现名称、版本、工件哈希、校准清单哈希和额外字节写入报告；
7. 不允许 AWQ/HSA 缺少工件时静默退回 RTN。

入口工厂为：

```python
make_quant_apply(spec, scopes=..., artifact=...)
```

后续量化代码可以替换该工厂内部实现，也可以在这里注册新的后端，但不应
修改 `Policy.act()`、`core.rollout` 或 LIBERO 环境逻辑。

## 工件和实验门禁

正式 AWQ/HSA 工件必须通过以下顺序：

```text
weights_only 加载
  -> method / implementation 校验
  -> checkpoint 哈希校验
  -> 精确层集合校验
  -> 每层权重和 bias 哈希校验
  -> scale / clip / residual 形状校验
  -> 固定输入动作保真度
   -> 全量 LIBERO 评测
```

测试工件必须放在 `artifacts/validation/`，并标记为
`validation_only: true`；它们只能验证接入管线，不能进入正式结果矩阵。

## 不需要修改的部分

后续量化实现不需要修改：

- `Observation` / `ActionChunk` 契约；
- 三个模型的 `act()`；
- 统一 rollout 和 episode 恢复；
- LIBERO 成功判定；
- 显存和延迟统计；
- `report_matrix.py` 的结果汇总逻辑。

量化方法完成后，主要工作是生成正式工件、通过门禁，然后使用现有统一入口
完成三模型四集合的评测。
