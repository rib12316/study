# 特性 #4：scalar_type —— vLLM 如何用一种类型描述所有数值（uint4/int8/fp8/e4m3/带偏置的量化）

> 学习阶段：AI Infra 基础储备 / 量化方向（你的主场，量化主线收尾）
> 对应源码：`vllm/scalar_type.py` + `vllm/model_executor/layers/quantization/utils/quant_utils.py`
> 本讲定位：前三讲你见了 `uint4`、`float8_e4m3fn`、`uint4b8` 这些"奇怪的类型名"。**这一讲回答：vLLM 怎么用一种统一的数据结构，把整数、浮点、带偏置的量化类型全部描述清楚？** 这背后是一个非常优雅的设计——`ScalarType`。理解它，你就理解了 vLLM 量化类型系统的灵魂，也能解释"GPTQ 的 4bit 到底存的是 0~15 还是 -8~7"。
> 干中学原则：本讲你要**亲手实现一个 mini ScalarType**，并用它驱动 pack/unpack，验证 GPTQ 的 bias 机制。

---

## 一、为什么 torch.dtype 不够用？（背景）

你做量化，最熟悉的数值类型大概是：
- `torch.int8`（8bit 有符号整数，-128~127）
- `torch.uint8`（8bit 无符号整数，0~255）
- `torch.float8_e4m3fn`（FP8，1符号+4指数+3尾数）
- `torch.bfloat16`、`torch.float16`...

但 PyTorch 的 dtype 有两个致命缺陷：

### 缺陷1：不支持 sub-byte（子字节）类型
PyTorch **没有 `torch.int4` 或 `torch.uint4`**！最小就是 8bit。但量化的灵魂就是 4bit/2bit。第二讲你看到的 INT4 权重，在 torch 里只能用 `int32` 装 8 个 4bit 值来"伪装"——torch 根本不知道这是个 4bit 类型。

### 缺陷2：表达不了"带偏置的整数"
GPTQ 的 4bit 量化，存储的是无符号整数 `0~15`，但**实际代表的值是 `-8~7`**（带符号的）。这种"存储值 = 实际值 + bias"的关系，`torch.uint8` 这种纯类型根本表达不了。

vLLM 为此发明了 `ScalarType`——一个能描述**任意位宽、任意指数/尾数划分、任意符号性、任意偏置**的统一类型。它是 vLLM 量化类型系统的基石。

> 💡 对你的意义：你以后设计任何量化格式，数值类型描述都是第一步。`ScalarType` 是一个可以直接抄的、工业级的子字节类型系统设计。

---

## 二、核心：ScalarType 的四字段模型

`vllm/scalar_type.py:22`

```python
@dataclass(frozen=True)
class ScalarType:
    exponent: int        # 指数位（浮点用；整数则为 0）
    mantissa: int        # 尾数位（浮点）或"有效位"（整数，不含符号位）
    signed: bool         # 是否有符号位
    bias: int            # 偏置：stored_value = actual_value + bias（量化用）
```

**就这四个字段，能描述一切**。关键推导（`scalar_type.py:167`）：

```python
@property
def size_bits(self) -> int:
    return self.exponent + self.mantissa + int(self.signed)
```

- **整数类型**：`exponent=0`，`size_bits = mantissa + signed`。
- **浮点类型**：`exponent>0`，`size_bits = exponent + mantissa + signed`。
- **是否浮点**：`is_floating_point()` 等价于 `exponent != 0`。

### 2.1 用四字段重建你熟悉的类型

| 类型 | exponent | mantissa | signed | bias | size_bits | 来源 |
|------|----------|----------|--------|------|-----------|------|
| `uint4` | 0 | 4 | False | 0 | 4 | `scalar_types.uint4` |
| `int4` | 0 | 3 | True | 0 | 4 | `scalar_types.int4`（注意 mantissa=3，因为有1符号位） |
| `int8` | 0 | 7 | True | 0 | 8 | `scalar_types.int8` |
| `float8_e4m3fn` | 4 | 3 | True | 0 | 8 | FP8，有限值无 NaN |
| `float8_e5m2` | 5 | 2 | True | 0 | 8 | FP8，IEEE754 |
| **`uint4b8`** | 0 | 4 | False | **8** | 4 | **GPTQ 4bit！** 存 0~15，实际 -8~7 |
| `uint8b128` | 0 | 8 | False | **128** | 8 | GPTQ 8bit：存 0~255，实际 -128~127 |

> ⚠️ 注意 `int4` 的 mantissa 是 **3** 不是 4！因为 `size_bits = mantissa + signed = 3 + 1 = 4`。这四个字段的关系要刻进脑子。

### 2.2 bias 的精妙：`uint4b8` 是 GPTQ 的灵魂

`scalar_types.uint4b8 = ScalarType.uint(4, 8)`（`scalar_type.py:350`）

这表示：**存储一个 4bit 无符号数（0~15），但它代表的实际值是 `stored - 8`，即 -8~7**。

为什么 GPTQ 要这么做？因为：
1. GPU 的 packed int32 解包出来的天然是**无符号**的（第二讲你 unpack 出来都是 0~15）。
2. 但权重量化后是对称分布的负正数（-8~7）。
3. 用 `bias=8`，就让"存储的无符号 0~15"和"实际的带符号 -8~7"优雅对应：`actual = stored - bias`。

这就是为什么第二讲里 GPTQ 的 `qzeros` 要"减1存储"——本质就是 bias 机制。`ScalarType` 把这个约定显式编码进了类型本身，而不是散落在各处注释里。

> 💡 `min()` 和 `max()` 自动考虑 bias（`scalar_type.py:170-182`）：`uint4b8.min() = 0 - 8 = -8`，`uint4b8.max() = 15 - 8 = 7`。类型自带值域，这是 dtype 做不到的。

---

## 三、ScalarType 驱动 pack/unpack（连接第二讲）

第二讲你手写 AWQ 的 pack/unpack 时，硬编码了 `PACK_FACTOR=8`、`MASK=0xF`。现在用 ScalarType，这些**全部从类型派生**。看 vLLM 的通用 pack 函数（`quant_utils.py:461`）：

```python
def pack_quantized_values_into_int32(w_q, wtype: ScalarType, packed_dim=0):
    pack_factor = 32 // wtype.size_bits      # ← 从类型派生！uint4 → 8，int8 → 4
    mask = (1 << wtype.size_bits) - 1         # ← 从类型派生！uint4 → 0xF
    ...
    for i in range(pack_factor):
        res |= (w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i
```

**这是第二讲的升华**：第二讲你针对 AWQ 4bit 写死了常量；这里**同一个函数，传入不同的 ScalarType，就能 pack 任何位宽**。`uint4`→每int32装8个，`int8`→每int32装4个，`uint2b2`→每int32装16个。**类型即配置，函数即通用逻辑**——这就是类型抽象的威力。

unpack（`quant_utils.py:483`）对称：
```python
pack_factor = 32 // wtype.size_bits
mask = (1 << wtype.size_bits) - 1
res[..., i::pack_factor] = (w_q_perm >> wtype.size_bits * i) & mask
```

---

## 四、ScalarType 的 ID 编码（跨 Python/C++ 边界）

一个隐藏但重要的设计（`scalar_type.py:136`）：ScalarType 有个 `id` 属性，把四个字段打包成一个 int64：

```python
@functools.cached_property
def id(self) -> int:
    val = 0; offset = 0
    def or_and_advance(member, bit_width):
        nonlocal val, offset
        val = val | (int(member) & ((1 << bit_width) - 1)) << offset
        offset += bit_width
    or_and_advance(self.exponent, 8)        # 8 bits
    or_and_advance(self.mantissa, 8)        # 8 bits
    or_and_advance(self.signed, 1)          # 1 bit
    or_and_advance(self.bias, 32)           # 32 bits
    or_and_advance(self._finite_values_only, 1)
    or_and_advance(self.nan_repr.value, 8)
    return val
```

为什么？因为 **PyTorch 的 custom op 只能传基本类型（int/float/tensor）**，不能传自定义 Python 对象。ScalarType 要传给 C++ kernel（`csrc/core/scalar_type.hpp`），就先编码成 int64，对面再解码。**Python 和 C++ 各有一份 ScalarType 实现，靠 id 格式保持同步**（`scalar_type.py:19-21` 注释强调"keep in sync"）。

> 💡 这是高性能库的常见模式：Python 端用 dataclass 做高层抽象，跨 FFI 边界时序列化成扁平的 int。你以后写自定义 CUDA kernel 接 Python 时会反复用到这个套路。

---

## 五、ScalarType 在量化系统里的角色（串联前三讲）

回到第一讲的分层：`Config → Method → kernel`。ScalarType 横跨这三层：

1. **Config 层**：每个量化方法的 Config 持有一个 `quant_type: ScalarType`（如 `AutoAWQConfig.quant_type = scalar_types.uint4`，`auto_awq.py:209`）。
2. **Method 层**：`create_weights` 用 `quant_type.size_bits` 算 `pack_factor`，决定权重的 packed shape。
3. **Kernel 层**：Marlin/Machete 的 `can_implement(config)` 用 `weight_type` 判断能不能处理这种类型；kernel 接收 ScalarType 的 `id` 来知道位宽和 bias。

第一讲的 `MPLinearLayerConfig` 有个 `weight_type` 字段（第二讲 `auto_awq.py:462`），就是 ScalarType。**整个 kernel 选择树（第二讲第五节）的分支条件，本质是在问"我能不能实现这个 ScalarType"**。

### 5.1 命名约定（`__str__`，scalar_type.py:218）

ScalarType 的字符串表示遵循 [ml_dtypes](https://github.com/jax-ml/ml_dtypes) 约定：
- 浮点：`float<size>_e<exp>m<man>[f|n]`（如 `float8_e4m3fn`：f=finite only, n=nan supported）
- 整数：`[u]int<size>[b<bias>]`（如 `uint4b8`：4bit 无符号，bias 8）

这个名字你会在日志、配置、文档里到处看到，认得它就能反推出四个字段。

---

## 六、把第四讲和前三讲连起来

| 讲次 | 视角 | 第四讲的位置 |
|------|------|-------------|
| 第1讲 | 机制骨架 | ScalarType 是 Config 的 `quant_type` 字段 |
| 第2讲 | AWQ 位序打包 | 第二讲硬编码 PACK_FACTOR/MASK，第四讲证明它们**可由 ScalarType 派生** |
| 第3讲 | 统一格式 | compressed-tensors 的 `num_bits/type/symmetric` 解析后映射到 ScalarType |
| **第4讲** | **类型系统** | **ScalarType 是贯穿三层的"类型语言"** |

**第四讲是量化主线的"底层语法"**：前三讲讲的是"怎么配置、怎么打包、怎么统一格式"，第四讲回答"这些配置里反复出现的数值类型，本质上是什么"。理解了 ScalarType，你回看前三讲会发现——所有关于 `uint4`/`fp8`/`bias` 的细节，都是这一个数据结构的不同实例。

---

## 七、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：四字段模型（基础）
1. 打开 `scalar_type.py:327`。`scalar_types.int4` 和 `scalar_types.uint4` 的四个字段分别是什么？它们的 `size_bits` 相同吗？`mantissa` 相同吗？为什么？
2. `scalar_types.uint4b8`（350行）的 `min()` 和 `max()` 返回什么？写出推导过程（提示：`min = _raw_min() - bias`）。
3. `scalar_types.float8_e4m3fn`（332行）的 `exponent` 和 `mantissa` 各是多少？它的 `size_bits` 是多少？`is_floating_point()` 为什么返回 True？

### 任务 B：pack/unpack（核心）
4. 读 `quant_utils.py:461 pack_quantized_values_into_int32`。如果传入 `wtype=scalar_types.int8`（8bit），`pack_factor` 和 `mask` 分别是多少？一个 int32 能装几个 int8？
5. 同一函数，循环 `for i in range(pack_factor)` 里 `(w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i` 这行在做什么？用 uint4 举例说明第 i 次迭代把哪些值放到 int32 的哪个 bit 段。
6. 对比第二讲你写的 `convert_awq_qweight_matrix`：vLLM 的 `pack_quantized_values_into_int32` 是**标准位序**（不像 AWQ 有 `[0,4,1,5,...]` 置换）。结合第三讲的 compressed-tensors（它用标准位序），解释为什么 compressed-tensors 格式的 W4A16 不需要 AWQ 那种位序转换。

### 任务 C：跨边界与 bias（机制）
7. 读 `scalar_type.py:136 id`。`uint4b8` 的 `id` 计算时，bias=8 占了 32 个 bit 中的低位某段。如果两个 ScalarType 的 `exponent/mantissa/signed` 都相同但 bias 不同，它们的 id 会不同吗？这对 C++ kernel 区分"uint4 vs uint4b8"有什么意义？
8. `quant_utils.py:699 SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]`。为什么 GPTQ 用带 bias 的类型（`b8`/`b128`）而不是普通的 `uint4`/`uint8`？结合第二节 2.2 的"GPTQ 存无符号、表带符号"解释。
9. 思考题：如果你要给 vLLM 加一个新的 `int2`（2bit 有符号）量化类型，按 ScalarType 的四字段模型，应该怎么构造？写出一行等价的 `ScalarType(...)` 调用。

---

## 八、干中学实践任务（核心！）

> 在 `practice_scalar_type.py` 里实现一个 mini ScalarType + 基于 ScalarType 的通用 pack/unpack。
> 依赖：仅标准库。不需要装 vllm/torch。
> 设计哲学：你不读 vLLM 的 ScalarType，而是**重建**一个最小可用版本。能用它驱动 pack/unpack 并复现 GPTQ bias，才算真懂。

### 实践 1：实现 mini ScalarType（热身）
实现 `ScalarType` 类（dataclass），包含：
- 四字段：`exponent, mantissa, signed, bias`
- 派生属性：`size_bits`、`is_floating_point()`
- 方法：`min()`、`max()`（考虑 bias）
- 工厂方法：`uint(size_bits, bias)`、`int_(size_bits, bias)`、`float_IEEE754(exp, man)`
- `__str__`：输出 `uint4`/`int8`/`uint4b8`/`float8_e4m3` 格式

验证：构造 `uint4`、`int4`、`uint4b8`、`float8_e4m3`，检查 size_bits/min/max/str 全对。

### 实践 2：基于 ScalarType 的通用 pack/unpack（核心）
实现 `pack_into_int32(values: list[int], wtype: ScalarType) -> int` 和 `unpack_from_int32(packed: int, wtype: ScalarType) -> list[int]`：
- **从 wtype 派生** pack_factor 和 mask（不许硬编码！）
- 这是 `quant_utils.py:461/483` 的标量版

验证：
```python
wtype = ScalarType.uint(4, None)          # uint4
v = [1, 2, 3, 4, 5, 6, 7, 8]              # 8 个值（因为 pack_factor=8）
assert unpack_from_int32(pack_into_int32(v, wtype), wtype) == v

wtype2 = ScalarType.int_(8, None)         # int8
v2 = list(range(-4, 4))                   # 4 个值（pack_factor=4）
# 注意 int8 存的是带符号值，pack 时要转无符号（补码）
```

### 实践 3：GPTQ bias 机制（精髓）
用你的 ScalarType 验证 GPTQ 的 `uint4b8`：
- 构造 `uint4b8 = ScalarType.uint(4, 8)`
- 假设 GPTQ 量化后的实际权重值是 `[-8, -1, 0, 3, 7]`（带符号）
- 用 bias 计算它们的**存储值**（应为 `[0, 7, 8, 11, 15]`，无符号）
- 把存储值 pack 进 int32，再 unpack，再减回 bias，应还原原始带符号值

验证：这个 round-trip 证明"GPTQ 用无符号存储 + bias 还原带符号"的机制，并且你的 ScalarType 正确编码了 bias。

> 💡 实践 2 的 int8 有个坑：带符号数要转成无符号补码才能 pack（`stored = actual & mask`）。这正是 bias 之外的另一种"符号处理"。vLLM 的 GPTQ 走 bias 路线（uint4b8），而有些量化走补码路线（int8），两条路都能解决"存储带符号"，ScalarType 都支持。

---

## 九、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_scalar_type.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_scalar_type.py 运行结果）

---

## 十、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现 ScalarType 后，你对"类型"这个概念的理解有什么变化？② bias 机制和补码机制两条路，你觉得哪个更优雅？为什么 GPTQ 选 bias？③ size_bits 从类型派生 vs 硬编码，工程上各有什么得失？

（完成实践后填写）

---

## 十一、个人复盘感悟（留给你写）

> 你是量化方向研究生，建议角度：① ScalarType 这种"用一个数据结构描述整个类型空间"的设计，你在量化研究里见过类似的抽象吗（比如你用 numpy/torch 时怎么描述自定义数值类型）？② bias 编码进类型本身 vs 散落在量化逻辑里，这种"把约定固化进类型"的思想对你设计量化方法/写库有什么启发？③ sub-byte 类型（int4/uint2）是量化的刚需，但 PyTorch 至今不支持原生，你怎么看框架对量化的支持滞后于算法发展的现象？

（在此写下你的感悟）


---
---

> ✅ **量化主线四讲全部完成！** 你现在拥有 vLLM 量化子系统的完整图景：
> - 第1讲 **机制骨架**（注册表/工厂/override/CPA）
> - 第2讲 **AWQ 算法实现**（位序打包 + Marlin 分发）
> - 第3讲 **统一格式**（compressed-tensors 元语言）
> - 第4讲 **类型系统**（ScalarType 子字节类型）
>
> 这四讲覆盖了架构层、算法层、格式层、类型层——面试时被问"vLLM 量化怎么实现的"，你能从这四个层次系统作答。
>
> 完成实践和感悟后告诉我，下一步建议进入 **vLLM 招牌特性 PagedAttention / Continuous Batching**（面试必问，和量化是两个不同维度的 Infra 知识），或你指定的其他特性。
