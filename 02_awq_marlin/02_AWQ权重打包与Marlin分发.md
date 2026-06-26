# 特性 #2：AWQ 权重打包格式 + Marlin Kernel 分发（INT4 量化在引擎里怎么"跑起来"）

> 学习阶段：AI Infra 基础储备 / 量化方向（你的主场）
> 对应源码：vllm-project/vllm @ main（2026-06）
> 本讲定位：你研究生的量化数学已经懂了（scale/zero-point/group），这一讲回答——**量化后的 INT4 权重，到底以什么"字节布局"存在显存里？引擎又怎么为它挑一个 GEMM kernel？**
> 干中学原则：本讲的实践任务要求你**亲手实现 INT4 pack/unpack 和 AWQ 位序转换**，并用 vLLM 源码里一模一样的算法验证。

---

## 一、这一讲为什么对你重要

你在实验室里做量化，通常这样描述一个 INT4 权重：

> "一个 `[K, N]` 的矩阵，每个元素是 4bit 无符号整数（0~15），按 group_size=128 分组，每组有一个 fp16 scale 和一个 int4 zero-point。"

这是**数学描述**。但落到推理引擎里，问题立刻变成：

1. **内存里这 4bit 怎么排？** GPU 内存最小寻址单位是 1 字节（8bit），一个 4bit 数没法单独存。必须把**8 个 4bit 数塞进一个 int32**（4 字节）里。
2. **塞进去的顺序是什么？** 这就是"打包顺序"（pack order）。AWQ 和 GPTQ 用了**不同的顺序**，这是两套生态互不兼容的根源之一。
3. **scale/zero-point 存哪一维？** AWQ 原始格式把 `qweight` 沿**输出维** pack，而 Marlin kernel 要求沿**输入维** pack。不转换就跑不了 Marlin。
4. **跑哪个 kernel？** 同样是 W4A16，A100 上跑 Machete/Marlin 最快，老显卡只能跑 Triton/Exllama。引擎怎么选？

这四个问题，本讲全部回答。**其中问题 2（pack order）和问题 3（维度转换）是你要亲手实现的部分**——这是真正的"干中学"。

---

## 二、前置知识：INT4 权重的三个张量

一个 AWQ INT4 量化的 Linear 层，磁盘上有三个张量（对应 vLLM `create_weights` 里注册的三个参数，`auto_awq.py:900`）：

| 张量 | 数学含义 | 形状（原始 AWQ 格式） | dtype |
|------|---------|----------------------|-------|
| `qweight` | 量化后的权重（4bit 值） | `[K, N//8]` | int32（每 int32 装 8 个 4bit） |
| `scales` | 每组缩放因子 | `[K//group_size, N]` | fp16 |
| `qzeros` | 每组零点（4bit 值，已减 1 存储） | `[K//group_size, N//8]` | int32 |

> 注意 **AWQ 原始格式的 qweight 是 `[K, N//8]`**——pack 在**输出维 N** 上。这点非常关键，后面要和 Marlin 要求的格式对比。

反量化公式（你熟悉的）：
```
W_real = (qweight - qzeros) * scales      # 逐 group
```
其中 `qweight`、`qzeros` 是 0~15 的 uint4，`scales` 是 fp16。

---

## 三、核心难点①：INT4 怎么塞进 int32？（pack order）

### 3.1 标准打包顺序

8 个 4bit 数 `[v0, v1, ..., v7]` 装进一个 int32，"标准顺序"是按位从小到大排：

```
int32 = v0 | (v1 << 4) | (v2 << 8) | (v3 << 12) | ... | (v7 << 28)
       位 [0:4]  [4:8]    [8:12]    [12:16]            [28:32]
       索引  0    1         2          3                  7
```

**unpack**（取出来）就是：
```python
v_i = (int32 >> (4 * i)) & 0xF     # i = 0..7
```

### 3.2 AWQ 的非标准打包顺序（重点！）

`vllm/model_executor/layers/quantization/auto_awq.py:72`

```python
# AWQ uses a non-standard packing order within int32 values.
# For 4-bit: standard order stores values at bit positions [0,4,8,12,16,20,24,28]
# for indices [0,1,2,3,4,5,6,7], while AWQ stores them for indices
# [0,4,1,5,2,6,3,7]. This permutation reverses that ordering.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

要精确理解这个置换的语义（这是你做实践任务时最容易踩的坑，我先帮你趟平）：

> **`_REVERSE_AWQ_PACK_ORDER` 是一张"槽位→标准序值"的查表。** 即 AWQ 格式的 int32 有 8 个 4bit 槽位（bit[0:4]、bit[4:8]、…、bit[28:32]，记作槽位 0~7）。AWQ 规定：
> - 槽位 0 装的是标准序的**第 0** 个值
> - 槽位 1 装的是标准序的**第 4** 个值
> - 槽位 2 装的是标准序的**第 1** 个值
> - ……即槽位 j 装标准序值 `reverse[j]`

由此推出**两个方向的操作**（你的实践任务要实现的）：
- **unpack AWQ**：取出 8 个槽位的值，按 `reverse` 重排，得标准序值。这正是 vLLM `_convert_awq_to_standard_format` 第 119 行 `unpacked[:, :, reverse_order]` 做的事。
- **pack 成 AWQ**（实践脚本构造输入要用）：标准序值 `v_i` 要放进满足 `reverse[j]==i` 的那个槽位 `j`，也就是放进 `j = reverse⁻¹[i]`。`reverse` 的逆置换是 `[0,2,4,6,1,3,5,7]`（实践脚本里叫 `_INV_AWQ_PACK_ORDER`）。

> ⚠️ 易错点：很多人（包括我第一遍）以为"pack 成 AWQ 直接用 `reverse` 索引"就行，结果 round-trip 对不上。**pack 用逆置换，unpack 用正置换**，两者不能混。这个坑我在给你准备实践脚本时亲自踩过、定位过——见实践脚本头部注释。

为什么 AWQ 这么做？——历史原因（早期 GPU kernel 为了对齐线程束 warp 的访问模式做的优化），但代价是**和 GPTQ/Marlin 生态不兼容**。

> 💡 这是量化工程里非常典型的一课：**一个看似随意的"字节排布"决定，会绑死整个软件栈**。你在做自己的量化方法时，pack order 的设计要三思——它影响所有下游 kernel。

**你接下来要在实践脚本里亲手实现**：给定一个 AWQ 格式的 int32，按这个非标准顺序 unpack 出 8 个值，再按标准顺序重新 pack。这就是 vLLM 的 `_convert_awq_to_standard_format` 的第一步。

---

## 四、核心难点②：从 AWQ 格式转到 Marlin 格式

看 vLLM 的转换函数 `_convert_awq_to_standard_format`（`auto_awq.py:92`）。它干两件事：

### 4.1 修 pack order（位序）

```python
# 第 110-120 行
shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)  # [0,4,8,...,28]

# Unpack int32 → 8 个值，按 AWQ 顺序取出
unpacked = (qw.unsqueeze(-1) >> shifts) & 0xF        # (K, N_packed, 8)
unpacked = unpacked[:, :, reverse_order]             # ← 修正 AWQ 位序
unpacked = unpacked.reshape(K, N)                     # (K, N) 标准顺序的 8 个值
```

### 4.2 换 pack 维度（输出维 → 输入维）

AWQ 原始 `qweight` 是 `[K, N//8]`（pack 在输出维 N 上）。但 Marlin/Machete kernel 要求 `qweight` 是 `[K//8, N]`（pack 在输入维 K 上）。所以：

```python
# 第 122-126 行：沿输入维重新 pack
unpacked = unpacked.reshape(K // pack_factor, pack_factor, N)
new_qw = (unpacked.to(torch.int32) << shifts[None, :, None]).sum(dim=1, dtype=torch.int32)
# 现在 new_qw 是 [K//8, N]，pack 在输入维
```

> 💡 为什么要 pack 在输入维？因为 W4A16 的 GEMM 是 `x[K] @ W[K,N]`，把 K 维 pack 起来，kernel 一次读一个 int32 就拿到 8 个连续的 K 维权重，配合 x 的 8 个连续激活做点积，访存效率最高。这是**kernel 友好的数据布局**的经典例子。

### 4.3 qzeros 也要转

qzeros 同样要从 AWQ 的 `[G, N//8]`（G=组数）转成 Marlin 要的 `[N//8, G]`，并修正位序。代码在 `auto_awq.py:141-167`，逻辑和 qweight 对称，只是转置了维度。

---

## 五、核心难点③：Kernel 怎么选？（Marlin 分发）

转换完格式后，`AutoAWQMarlinLinearMethod` 不自己写 GEMM，而是把活儿交给 `choose_mp_linear_kernel`（`auto_awq.py:469`，实现在 `kernels/linear/__init__.py:648`）。

### 5.1 候选 kernel 优先级表

`kernels/linear/__init__.py:359`

```python
_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[MPLinearKernel]]] = {
    PlatformEnum.CUDA: [
        CutlassW4A8LinearKernel,   # 优先级最高
        MacheteLinearKernel,       # Hopper 上最快
        AllSparkLinearKernel,
        MarlinLinearKernel,        # Ampere 及以上通用的主力
        HummingLinearKernel,
        ConchLinearKernel,
        ExllamaLinearKernel,       # 老牌 fallback
        TritonW4A16LinearKernel,   # 纯 Triton，兼容性兜底
    ],
    # ROCm / CPU / XPU 各有各的列表...
}
```

**这是一个按性能从高到低排好的责任链。** 列表前面的 kernel 更快但要求更高（要新硬件、特定 shape），后面的更慢但更兼容。

### 5.2 探测逻辑

`choose_mp_linear_kernel`（`__init__.py:690`）核心循环：

```python
for kernel in platform_kernels:
    # ① 被环境变量禁用了？跳过
    if kernel.__name__ in envs.VLLM_DISABLED_KERNELS: continue
    # ② 算力够吗？（Marlin 要 80+，Machete 要 90+）
    if kernel.get_min_capability() > compute_capability: continue
    # ③ 这个 kernel 支持当前的 weight_type/group_size/shape 吗？
    can_implement, reason = kernel.can_implement(config)
    if can_implement:
        return kernel          # ← 第一个能用的就用，返回
    else:
        failure_reasons.append(...)

raise ValueError(...)          # 全都不行才报错
```

**这就是问题④的答案**：按优先级挨个问"你能跑吗"，第一个说"能"的就接管。这叫 **capability probe + fallback chain**，是高性能库的通用设计。

### 5.3 三个旋钮

用户/系统可以通过三种方式影响选择：
1. **硬件算力**：`get_min_capability()` 天然过滤（A100=80 跑不了 Machete）。
2. **环境变量** `VLLM_DISABLED_KERNELS`：显式禁用某 kernel（调试用）。
3. **CLI** `--linear-backend marlin`：`_filter_kernels_by_backend` 强制只考虑某个集合（`__init__.py:272`）。

### 5.4 选完 kernel 之后

`AutoAWQMarlinLinearMethod.__init__`（`auto_awq.py:438`）拿到 kernel 类型后，在 `create_weights` 里实例化它，并把 `qweight/scales/qzeros` 的参数名告诉 kernel：

```python
self.kernel = kernel_type(
    mp_linear_kernel_config,
    w_q_param_name="qweight",
    w_s_param_name="scales",
    w_zp_param_name="qzeros",
)
```

之后 `process_weights_after_loading` 调 `self.kernel.process_weights_after_loading(layer)`（让 kernel 自己做最后的排布，比如 Marlin 的 tile 重排），`apply` 调 `self.kernel.apply_weights(layer, x, bias)`。**LinearMethod 彻底沦为"格式转换 + kernel 调度"的薄壳**，真正的计算全在 kernel 里。这是 vLLM 新一代 modular kernel 架构的精髓。

---

## 六、把第二讲和第一讲连起来

| 层次 | 第一讲（机制） | 第二讲（一个具体方法的工程实现） |
|------|--------------|------------------------------|
| 注册 | `get_quantization_config("awq")→AutoAWQConfig` | `AutoAWQConfig` 就是第一讲工厂里的一项 |
| override | 责任链确定用 `auto_awq` | `AutoAWQConfig.override_quantization_method` 认 `"awq"` checkpoint（`auto_awq.py:258`） |
| 分发 Method | `get_quant_method` 按 layer 类型 | `AutoAWQConfig.get_quant_method` 再按**硬件**分到 Marlin/Triton/XPU（`auto_awq.py:284`） |
| CPA | Create/Process/Apply 抽象 | `create_weights` 注册 qweight/qzeros/scales；`process_weights` 做格式转换；`apply` 调 kernel |

**第二讲是第一讲的"具体化"**：第一讲的骨架在 AWQ 这里填满了血肉。你会发现 `get_quant_method` 内部还有第二级分发（按硬件选 kernel），这是第一讲没展开的——**分层分发是 vLLM 一以贯之的设计哲学**。

---

## 七、AutoAWQConfig.get_quant_method 的完整决策树（必看）

`auto_awq.py:284` 的分发逻辑，画成树：

```
get_quant_method(layer, prefix)
├── is_layer_skipped(prefix)? ──yes──→ UnquantizedLinearMethod（不量化这层）
├── platform.is_xpu()? ──yes──→ AutoAWQXPULinearMethod
├── platform.is_cpu()? ──yes──→ AutoAWQMarlinLinearMethod（内部选 CPUWNA16）
├── platform.is_cuda() & check_marlin_supported()?
│   └── check_marlin_supports_layer(layer)?
│       ├── yes → AutoAWQMarlinLinearMethod（★ 主力路径）
│       └── no  → AutoAWQLinearMethod（fallback，走 awq_gemm/awq_dequantize）
└── else → AutoAWQLinearMethod（Triton 路径）
```

注意 fallback 的存在：**即使决定了用 AWQ，shape 不满足 Marlin 要求时还会退回 Triton kernel**。鲁棒性就是这样一层层兜底建起来的。

---

## 八、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码里完成，答案写进本文件末尾"任务答卷区"。

### 任务 A：打包格式（基础）
1. 打开 `auto_awq.py`。`AutoAWQConfig.__init__` 里 `self.pack_factor` 是怎么算的（`auto_awq.py:192`）？如果将来支持 `weight_bits=8`，`pack_factor` 会变成多少？
2. `auto_awq.py:72` 注释说 AWQ 的 4bit 打包索引是 `[0,4,1,5,2,6,3,7]`。请用纸笔推算：把标准顺序的 8 个值 `[a,b,c,d,e,f,g,h]` 按 AWQ 顺序 pack 进一个 int32，bit[0:4] 槽位放的是哪个字母？（提示：`_REVERSE_AWQ_PACK_ORDER` 是用来**反向**修正的）
3. `BaseAWQLinearMethod.create_weights`（`auto_awq.py:829`）里，`qweight` 的 shape 是 `(input_size_per_partition, output_size_per_partition // pack_factor)`。结合第二节的表格，解释为什么是 `// pack_factor` 而不是 `* pack_factor`。

### 任务 B：格式转换（核心）
4. 读 `_convert_awq_to_standard_format`（`auto_awq.py:92`）。第 118 行 `(qw.unsqueeze(-1) >> shifts) & mask` 的 `mask` 是多少？这行代码一次取出多少个 4bit 值？
5. 第 123-126 行把 unpacked 的 `(K, N)` 重组成 `(K//8, 8, N)` 再 pack。请解释：为什么重组的第二维恰好是 `pack_factor`（即 8）？这和"沿输入维 pack"有什么关系？
6. qzeros 的转换（第 145-157 行）比 qweight 多了一步 `.T`（转置）。结合第四节 4.3，说明 AWQ 的 qzeros 原始形状是什么、目标形状是什么，为什么需要转置。

### 任务 C：Kernel 分发（机制）
7. 打开 `kernels/linear/__init__.py`。`_POSSIBLE_KERNELS[PlatformEnum.CUDA]` 列表里，`MacheteLinearKernel` 排在 `MarlinLinearKernel` 前面。如果你在一个 A100（算力 80）上加载 AWQ 模型，会选中 Machete 吗？为什么？（提示：看 `choose_mp_linear_kernel` 第 696-705 行的算力检查）
8. `choose_mp_linear_kernel` 在所有 kernel 都不可用时 `raise ValueError`。去找一个 `MarlinLinearKernel` 的 `can_implement` 实现（在 `kernels/linear/mixed_precision/marlin.py`），列出它会因为哪些原因拒绝一个 layer（至少写 2 条）。
9. 思考题：`AutoAWQMarlinLinearMethod` 在 `create_weights` 里调 `choose_mp_linear_kernel`（`auto_awq.py:469`），而不是在 `__init__` 里。为什么选 kernel 的时机要放在 `create_weights`？（提示：这时候才知道 layer 的真实 shape）

---

## 九、干中学实践任务（必做，核心！）

> 本讲实践任务**全部要你亲手写代码**，并在 `practice_awq_pack.py` 里跑通验证。
> 设计哲学：你不是在"读" pack 算法，而是**重新发明**它，然后和 vLLM 源码对照——只有能复现，才算真懂。

### 实践 1：实现标准 INT4 pack/unpack（热身）
在 `practice_awq_pack.py` 里实现两个函数：
- `pack_standard(values: list[int]) -> int`：把 8 个 0~15 的值按**标准顺序** pack 成一个 int32。
- `unpack_standard(packed: int) -> list[int]`：逆操作。

验证：`unpack_standard(pack_standard([1,2,3,4,5,6,7,8])) == [1,2,3,4,5,6,7,8]`，并且每个值确实落在 0~15。

### 实践 2：实现 AWQ ↔ 标准 pack order 转换（核心）
这是第二讲第三节的精华。实现：
- `pack_awq_order(values: list[int]) -> int`：按 **AWQ 顺序 `[0,4,1,5,2,6,3,7]`** pack。
- `unpack_awq_order(packed: int) -> list[int]`：逆操作。

然后用 `_REVERSE_AWQ_PACK_ORDER = [0,4,1,5,2,6,3,7]` 实现：
- `awq_to_standard(packed: int) -> int`：把一个 AWQ 格式的 int32 转成标准格式。

验证（关键）：
```python
# 1. round-trip
assert unpack_awq_order(pack_awq_order(v)) == v
# 2. 和 vLLM 算法一致：标准值 v，先按 AWQ pack，再 awq_to_standard，应等于标准 pack
assert awq_to_standard(pack_awq_order(v)) == pack_standard(v)
```
第二条断言的意义：**证明你的转换和 vLLM 的 `_convert_awq_to_standard_format` 在位序修正上完全等价**。

### 实践 3：矩阵级转换（进阶，模拟 vLLM 的真实流程）
把实践 2 推广到矩阵：实现 `convert_awq_qweight_matrix(qw_awq: torch.Tensor) -> torch.Tensor`，输入 `[K, N//8]` 的 AWQ 格式 int32 矩阵，输出 `[K//8, N]` 的标准格式 int32 矩阵。**要求用向量化操作（不要 for 循环遍历每个 int32）**，这正是 vLLM 源码的做法（见 `auto_awq.py:118-126`）。

验证：随机生成一组 4bit 值，先排成 AWQ 格式矩阵，转成标准格式后 unpack，应能还原出原始 4bit 值。

> 💡 这些实践不需要 GPU、不需要装 vllm，只要 `torch`（CPU 版即可）和标准库。完成实践 1+2 就能拿到本讲 80% 的收获；实践 3 是给想深入的你。

---

## 十、任务答卷区

> 代码阅读任务 A/B/C 的答案写这里。实践任务的代码放进同目录 `practice_awq_pack.py`，跑通后把关键输出贴这里。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（在此贴 practice_awq_pack.py 的运行结果）

---

## 十一、学习过程与实践总结（这一节我们一起填）

> 完成实践后，在这里用 3-5 句话总结你**亲手实现**时踩的坑和顿悟的瞬间。示例方向：① 第一次看到 AWQ 位序时懵不懵？自己 pack 一遍后理解了吗？② 向量化 pack 和 for 循环 pack 的思路差异。③ 为什么"数据布局"这种看似底层的细节，其实是量化的灵魂。

（完成实践后填写）

---

## 十二、个人复盘感悟（留给你写）

> 你是量化方向的研究生，这里写你的深度思考。建议角度：① AWQ 的 pack order 设计从今天的视角看是"技术债"，你怎么看量化算法选择和工程生态的博弈？② vLLM 用一个统一的 `MPLinearKernel` 接口把 Marlin/Machete/Exllama/Triton 抽象掉，这种"格式转换 + kernel 调度"分层对你设计自己的量化推理后端有什么启发？③ 你做过的量化方法，如果接进 vLLM，pack order 该怎么设计才能尽量复用现有 kernel？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。完成后告诉我，下一讲建议进入 **特性 #3：compressed-tensors 统一量化格式 + 它如何"吞掉" AWQ/GPTQ/FP8（override 责任链的王者）**，继续你的量化主线；或者如果你想换领域，可以跳到 **PagedAttention / Continuous Batching** 等 vLLM 招牌特性。
