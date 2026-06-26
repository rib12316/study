# 特性 #3：compressed-tensors —— 一个"统一量化格式"如何描述一切

> 学习阶段：AI Infra 基础储备 / 量化方向（你的主场）
> 对应源码：vllm-project/vllm @ main（2026-06）
> 本讲定位：前两讲你认识了 AWQ（一种具体 W4A16）和 FP8（一种具体 W8A8）。它们各有各的配置格式、各有各的 Config 类。**这一讲回答一个更深层的问题：能不能用一套统一的 JSON schema 描述所有量化方法？** compressed-tensors 就是答案，它是 neuralmagic（LLM-compressor）主导的事实标准。
> 干中学原则：本讲你要**亲手实现一个 compressed-tensors config 解析器**——把 `config.json` 里的 `config_groups` 解析成 `target → scheme` 映射，并实现 target 匹配（精确/正则/类名）。这是 vLLM `_quantization_scheme_map_from_config` + `find_matched_target` 的复刻。

---

## 一、为什么需要"统一格式"？（背景）

你在实验室里一定遇到过这种痛苦：

- AWQ 模型的 `quantize_config.json` 长这样：`{"bits":4, "group_size":128, "zero_point":true}`
- GPTQ 模型的长这样：`{"bits":4, "group_size":128, "desc_act":false, "sym":true}`
- FP8 模型的长这样：`{"quant_method":"fp8", "activation_scheme":"dynamic"}`
- 你自己的量化方法……又是一套。

**每种方法都发明自己的配置字段**，导致：
1. 推理引擎要为每种方法写专门的解析器（第一讲那 20+ 个 Config 类就是这么来的）。
2. 想做"混合精度"（比如 LM head 用 FP8、其它层用 W4A16）？AWQ/GPTQ 的单一格式根本表达不了。
3. 量化工具链（如 llm-compressor）产出的模型，换个引擎就读不懂。

compressed-tensors 的设计哲学：**不要为每种量化算法发明格式，而是发明一种"描述量化配置的元语言"**。它本身不规定量化数学，只规定"如何描述一层用什么策略量化"。一个 compressed-tensors 配置可以是：

```json
{
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],            // 所有 nn.Linear
      "weights": {                       // 权重怎么量化
        "num_bits": 4, "type": "int",
        "symmetric": false, "strategy": "group", "group_size": 128
      },
      "input_activations": null          // 激活不量化 → 这就是 W4A16
    },
    "group_1": {
      "targets": ["re:.*lm_head"],       // 正则匹配 lm_head
      "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"},
      "input_activations": {"num_bits": 8, "type": "float", "dynamic": true}  // W8A8 FP8
    }
  },
  "ignore": ["lm_head.language_model"]   // 这些层完全不量化
}
```

**这一个文件就描述了"大部分层 W4A16、lm_head W8A8、某些层跳过"的混合精度方案**——AWQ/GPTQ 的单一格式根本做不到。这就是"统一格式"的价值。

> 💡 对你的意义：你以后设计自己的量化方法，**优先让它产出 compressed-tensors 格式**而不是自创格式，就能白嫖 vLLM 的 kernel 分发、混合精度支持、TP 兼容，几乎零成本接入生态。

---

## 二、核心数据结构：`target_scheme_map`

compressed-tensors 在 vLLM 里的一切都围绕一个中心数据结构（`compressed_tensors.py:94`）：

```python
# Map from [target -> scheme]
self.target_scheme_map: dict[str, dict] = target_scheme_map
```

它是 `target（目标层名）→ scheme_dict（量化方案）` 的字典。`scheme_dict` 形如：

```python
{
    "weights": QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=128, ...),
    "input_activations": QuantizationArgs(...) | None,
    "output_activations": QuantizationArgs(...) | None,
    "format": "pack-quantized"
}
```

这个 map 是从 `config_groups` 解析来的。解析过程就是你实践任务要复刻的 `_quantization_scheme_map_from_config`（`compressed_tensors.py:297`）。

---

## 三、解析流程：从 `config_groups` 到 `target_scheme_map`

`_quantization_scheme_map_from_config`（`compressed_tensors.py:297`）做的事：

```python
target_scheme_map = {}
for group_name, quant_config in config["config_groups"].items():
    targets = quant_config["targets"]        # ["Linear"] 或 ["re:.*proj"]
    for target in targets:
        target_scheme_map[target] = {
            "weights":         QuantizationArgs.model_validate(quant_config["weights"]),
            "input_activations": ... ,        # 可能 None
            "output_activations": ... ,       # 可能 None
            "format": ...,
        }
```

**关键设计**：扁平化。原始配置是 `config_groups → (targets, scheme)`，解析后变成 `target → scheme`，把"组"这层中间结构消解掉。这样后续查"某一层用什么 scheme"就只需一次 target 匹配，不用遍历组。

> ⚠️ 注意 `targets` 是一个**列表**，一个 scheme 可以应用到多个 target；同一 target 出现在多个组里则后者覆盖前者。

---

## 四、核心难点：target 匹配（layer_name 怎么找到它的 scheme）

这是 compressed-tensors 最精巧的部分，也是你实践任务的核心。给定一个 layer（比如 `model.layers.0.self_attn.q_proj`），怎么知道它属于哪个 target？

答案在 `find_matched_target`（`compressed_tensors.py:113` → `utils.py:113`）。它按**优先级**依次尝试三种匹配：

### 4.1 精确匹配 / 正则匹配（最高优先级）

`_is_equal_or_regex_match`（`utils.py:175`）：

```python
def _is_equal_or_regex_match(value, target, check_contains=False):
    if target.startswith("re:"):           # ① 正则：target 以 "re:" 开头
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:                   # ② 包含匹配（仅用于模块类名）
        if target.lower() in value.lower():
            return True
    elif target == value:                  # ③ 精确相等
        return True
    return False
```

三种 target 写法：
- `"model.layers.0.self_attn.q_proj"` —— 精确匹配某层
- `"re:.*proj"` —— 正则匹配所有 `proj` 结尾的层（`re:` 前缀）
- `"Linear"` —— 模块类名（靠 `check_contains` 走"包含"逻辑）

### 4.2 完整的三级匹配（`find_matched_target`，utils.py:146）

```python
matched_target = (
    _find_first_match(layer_name, targets)                              # ① 先按层名匹配
    or _find_first_match(module.__class__.__name__, targets, True)      # ② 再按模块类名（包含匹配）
    or _match_fused_layer(layer_name, targets, fused_mapping)           # ③ 最后处理融合层
)
```

**三级优先级**：
1. **层名精确/正则匹配**：`layer_name` 对每个 target 做 `_is_equal_or_regex_match`。
2. **模块类名包含匹配**：如果①没中，拿 `module.__class__.__name__`（如 `ReplicatedLinear`/`QKVParallelLinear`）做**包含**匹配。这就是为什么 `"Linear"` 这个 target 能匹配所有 Linear 子类——因为它们的类名都含 "Linear"。
3. **融合层匹配**：处理 `qkv_proj` 这种融合权重（`q_proj`+`k_proj`+`v_proj` 融合而成），见 `_match_fused_layer`。

> 💡 第②级是 vLLM 能用 compressed-tensors 的关键技巧：vLLM 内部的 Linear 不叫 `nn.Linear`，而是叫 `QKVParallelLinear`/`RowParallelLinear` 等。靠**类名包含 "Linear"** 这个宽松规则，一个 `"Linear"` target 就能覆盖 vLLM 所有变体。这是"配置与引擎实现解耦"的工程智慧。

---

## 五、从 scheme 到具体实现：`_get_scheme_from_parts`

拿到 layer 的 `scheme_dict` 后，`get_scheme`（`compressed_tensors.py:805`）调 `_get_scheme_from_parts`（684）把它映射成具体的 `CompressedTensorsScheme` 子类。这是一个**巨大的 if-elif 决策树**：

```python
def _get_scheme_from_parts(self, weight_quant, input_quant, output_quant, format, ...):
    if self._is_nvfp4_format(weight_quant):   return CompressedTensorsW4A4Fp4(...)
    if self._is_mxfp4(weight_quant):          return CompressedTensorsW4A4Mxfp4(...)
    if self._is_mxfp8(weight_quant):          return CompressedTensorsW8A8Mxfp8(...)
    if self._is_fp8_w4a8_sm90(...):           return CompressedTensorsW4A8Fp8(...)
    if self._is_wNa8o8_int(...):              return CompressedTensorsWNA8O8Int(...)
    if self._is_wNa16_group_channel(...):     return CompressedTensorsWNA16(...)   # ← AWQ 的 W4A16 在这
    ...
    if self._is_fp8_w8a8(...):                return CompressedTensorsW8A8Fp8(...) # ← FP8 W8A8 在这
    if self._is_static_tensor_w8a8(...):      return CompressedTensorsW8A8Int8(...)
    raise NotImplementedError(...)
```

`schemes/` 目录里有 11 个 scheme 类，对应不同 (num_bits, type, strategy, 激活是否量化) 组合：

| Scheme 类 | 对应什么量化 | 等价于前两讲的 |
|-----------|-------------|---------------|
| `CompressedTensorsWNA16` | W4A16 group 量化 | AWQ/GPTQ（第二讲） |
| `CompressedTensorsW8A8Fp8` | W8A8 FP8 | FP8（第一讲） |
| `CompressedTensorsW8A8Int8` | W8A8 INT8 | SmoothQuant 风格 |
| `CompressedTensorsW4A8Fp8` | W4A8 FP8 | 新格式 |
| `CompressedTensorsW8A16Fp8` | W8A16 FP8（仅权重量化） | FP8 权重量化 |

**这就是"统一格式"的落地**：一个 compressed-tensors 配置，根据 `(num_bits, type, strategy, input_activations)` 自动分发到 W4A16 / W8A8 / W8A16 等不同实现——而这些实现内部又复用了第一讲第二讲的 kernel 分发（Marlin/Machete/...）。**compressed-tensors 是格式层，scheme 是算法层，kernel 是实现层**，三层解耦。

---

## 六、关键澄清：compressed-tensors 不靠 override "吞并"，而是靠"成为产出标准"

这是我在源码里发现的一个**反直觉**的事实，必须讲清楚。

第一讲讲 override 责任链时，你可能以为 compressed-tensors 会用 `override_quantization_method` 去抢 AWQ/GPTQ 的控制权。**实际不是**：

1. 看 `_verify_quantization` 的 `overrides` 列表（`model.py:983`）：里面有 `auto_gptq`/`gptq`/`auto_awq`/`awq`/`modelopt`... **但就是没有 `compressed-tensors`**。
2. `CompressedTensorsConfig` **没有重写** `override_quantization_method`（用基类默认，返回 None）。

那 compressed-tensors 怎么被选中？——靠**正常工厂路径**：当 checkpoint 的 `quant_method == "compressed-tensors"` 时，第一讲的 `get_quantization_config("compressed-tensors")` 直接返回 `CompressedTensorsConfig`。它的优先级其实**最低**（排在所有 override 之后），但它是"正确声明身份"的 checkpoint 的**默认归宿**。

> 💡 深刻的设计哲学：**真正的"统一"不靠在运行时抢权（那是 AWQ/GPTQ 这种历史格式被迫做的适配），而是靠在产出端（llm-compressor）统一下游格式。** 谁控制了模型产出的格式标准，谁就控制了生态。compressed-tensors 不和 AWQ/GPTQ 在 override 列表里抢，是因为它假定未来的模型**直接以它的格式产出**。这是标准之战的思维，不是兼容之战的思维。

---

## 七、把第三讲和前两讲连起来

| 层次 | 第一讲 | 第二讲 | 第三讲 |
|------|-------|-------|-------|
| 配置格式 | 每方法一种 json | AWQ 的 quantize_config.json | **统一的 config_groups schema** |
| 配置类 | Fp8Config/AutoAWQConfig... | AutoAWQConfig | CompressedTensorsConfig（一个类管所有） |
| 分发维度 | 按 layer 类型 | 按 layer 类型 + 硬件 | 按 **target 匹配** + scheme 决策树 |
| 表达能力 | 单一量化 | 单一 W4A16 | **混合精度**（不同层不同量化） |

**第三讲是前两讲的"抽象升维"**：前两讲是"一种量化怎么跑"，第三讲是"如何用一套语言描述任意量化"。从工程角度看，compressed-tensors 是最接近"量化即配置（quantization-as-config）"理想的设计。

---

## 八、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：config 结构（基础）
1. 打开 `compressed_tensors/compressed_tensors.py:297`。`_quantization_scheme_map_from_config` 遍历 `config_groups` 时，对每个 group 取了哪些 key？（至少写出 4 个）
2. 一个 `config_group` 里 `input_activations` 为 `null` 意味着什么量化类型？（提示：W?A?）。结合 `_get_scheme_from_parts`，这种情况下 weight 是 int 4bit group 量化会落到哪个 scheme？
3. `CompressedTensorsConfig.__init__`（80行）的 `target_scheme_map` 是 `dict[str, dict]`。它的 key 是什么、value 是什么？为什么不用 `dict[str, QuantizationArgs]`（直接存 weights 的 args）？

### 任务 B：target 匹配（核心）
4. 读 `utils.py:175 _is_equal_or_regex_match`。一个 target 写成 `"re:model.layers.*.mlp"`，对 layer_name `"model.layers.5.mlp.gate_proj"` 会匹配吗？为什么？（提示：`re.match` 是从头匹配还是任意位置？）
5. `find_matched_target`（utils.py:146）的三级匹配里，第二级 `_find_first_match(module.__class__.__name__, targets, True)` 的 `True` 参数控制什么？为什么这级要用"包含匹配"而第一级不用？
6. 思考题：如果 vLLM 把内部的 `QKVParallelLinear` 类重命名成 `AttentionProjection`（不含 "Linear"），一个 `"targets": ["Linear"]` 的 compressed-tensors 配置会失效吗？这说明 target 匹配的什么脆弱性？

### 任务 C：scheme 决策（机制）
7. `_get_scheme_from_parts`（684）的 if-elif 顺序里，`_is_wNa8o8_int`（724行）的注释说"Must come before the WNA16 check"。如果不小心把它放到 WNA16 检查**之后**，会发生什么？（提示：WNA8O8 和 WNA16 的判断条件有重叠吗？）
8. `_get_scheme_from_parts` 里 `_is_fp8_w8a8` 匹配成功后，还会先调 `_check_scheme_supported(W8A8Fp8.get_min_capability(), error=False)`（750行）。这个 `error=False` 是什么意思？如果当前 GPU 算力不够，会落到哪个 scheme？
9. 思考题：compressed-tensors 没有重写 `override_quantization_method`（第六节）。但用户能用 `--quantization compressed-tensors` 强制加载一个 AWQ checkpoint 吗？去 `override_quantization_method` 的调用处（model.py:1015-1020）分析：当所有 override 都返回 None 时，最终 `self.quantization` 会变成什么？

---

## 九、干中学实践任务（核心！）

> 在 `practice_compressed_tensors.py` 里实现一个简化版 compressed-tensors config 解析器 + target 匹配器。
> 依赖：仅标准库（`re`）。不需要装 vllm/compressed-tensors/torch。
> 设计哲学：你不读 vLLM 的解析器，而是**重建**它。能解析真实样例配置并正确匹配，才算真懂。

### 实践 1：解析 config_groups → target_scheme_map（热身）
实现 `parse_config_groups(config: dict) -> dict[str, dict]`：
- 输入：第一节示例那样的 `config_groups` 字典
- 输出：扁平化的 `target → scheme_dict` 映射
- 要求：用简化的 `QuantArgs`（dataclass）存 num_bits/type/symmetric/strategy/group_size/dynamic。

验证：解析一个含 2 个 group、每 group 多 target 的配置，检查每个 target 都有正确的 weights scheme。

### 实践 2：target 匹配（核心）
实现 `find_matched_target(layer_name, module_class_name, targets) -> str | None`：
- 复刻第四节的三级匹配：精确/正则匹配 → 类名包含匹配
- 支持 `re:` 前缀的正则 target

验证（关键用例）：
```python
targets = ["model.layers.0.self_attn.q_proj", "re:.*lm_head", "Linear"]
assert find_matched_target("model.layers.0.self_attn.q_proj", "QKVParallelLinear", targets) == "model.layers.0.self_attn.q_proj"  # 精确
assert find_matched_target("model.lm_head", "Linear", targets) == "re:.*lm_head"           # 正则
assert find_matched_target("model.layers.1.mlp.gate_proj", "ReplicatedLinear", targets) == "Linear"  # 类名包含
assert find_matched_target("model.norm", "LayerNorm", targets) is None                     # 不匹配
```

### 实践 3：端到端——给一层找出它的 scheme（整合）
实现 `get_scheme_for_layer(layer_name, module_class, config) -> dict | None`：
- 先 `parse_config_groups` 得到 target_scheme_map
- 再 `find_matched_target` 找到匹配的 target
- 返回对应的 scheme_dict（不匹配返回 None，表示该层不量化）

验证：构造一个混合精度配置（Linear→W4A16，lm_head→W8A8，ignore norm），验证不同层能拿到正确 scheme，norm 拿到 None。

> 💡 这三个实践是层层递进的，做完你就亲手跑通了一遍 compressed-tensors 的核心解析逻辑。实践 2 的正则匹配是**最大的易错点**——`re.match` 从字符串**开头**匹配，但**不锚定结尾**（不是 `re.search`，也不是 `re.fullmatch`）。所以 `"re:.*lm_head"` 会匹配 `"model.layers.0.lm_head_extra"`（`.*` 吃掉前缀后 `lm_head` 匹配，尾巴 `_extra` 不影响结果）。这个细节 vLLM 源码（utils.py:186）就是这么写的，要忠实复刻。我在准备用例时一开始也踩了这个坑，特意留了⑤⑥两个 case 给你体验。

---

## 十、任务答卷区

> 代码阅读任务 A/B/C 答案写这里。实践代码放 `practice_compressed_tensors.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_compressed_tensors.py 运行结果）

---

## 十一、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 第一次见到 config_groups 时，你做量化的研究里有没有类似的"分层配置"思路？② target 匹配的三级优先级，哪级最让你觉得巧妙？③ 实现 re: 正则匹配时，re.match 和 re.search 的区别你踩了吗？

（完成实践后填写）

---

## 十二、个人复盘感悟（留给你写）

> 你是量化方向研究生，建议角度：① compressed-tensors "不抢权、做标准"的策略，你怎么看量化生态的标准之争？对你发论文/开源自己的量化方法有什么启发（要不要直接产出 ct 格式）？② "量化即配置"（quantization-as-config）这个抽象层次，比 AWQ/GPTQ 的"一种方法一个类"高明在哪？③ 你做过的量化研究里，混合精度（不同层不同量化）的需求大吗？ct 的 config_groups 是否满足你的研究需要？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。完成后告诉我，下一讲建议：① **继续量化主线收尾**——讲 `QuantizationArgs` 的 pydantic 校验 + `scalar_type`（vLLM 怎么统一描述 uint4/int8/fp8/e4m3 等所有数值类型）；② 或**换领域**——vLLM 的招牌特性 **PagedAttention / Continuous Batching**（这才是 vLLM 出名的根本原因，面试必问）。
