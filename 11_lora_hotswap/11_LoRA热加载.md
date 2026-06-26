# 特性 #11：LoRA 热加载 —— 一个模型怎么同时服务多个 adapter

> 学习阶段：AI Infra 基础储备 / 多 adapter 服务（新领域）
> 对应源码：`vllm/lora/layers/base_linear.py`（LoRA 权重叠加 + stacked 预分配）+ `vllm/lora/punica_wrapper`（多 adapter 混合 batch）+ `vllm/lora/request.py`（LoRARequest）
> 本讲定位：MoE 是"模型内部稀疏"，LoRA 是"一个 base 模型 + 多个轻量微调"。这一讲回答：**怎么在不复制整个模型的前提下，让一个推理服务同时服务几十个不同的 LoRA adapter，且同一个 batch 里的不同请求用不同 adapter？** 这是多租户/多任务 LLM 服务的核心特性。
> 干中学原则：本讲你要**亲手实现一个 mini LoRA**——base 权重 + LoRA(A,B) 低秩叠加 + 多 adapter 混合 batch（每个 token 按自己的 adapter index 取对应 A/B）。

---

## 一、为什么需要 LoRA 热加载？（背景）

### 1.1 微调的显存痛点

你有了一个 base 模型（比如 Llama-70B）。10 个不同任务要微调。朴素方案：每个任务全参数微调，存 10 份 70B 权重——显存爆炸，服务时还要在 GPU 间切换，慢。

### 1.2 LoRA：低秩适配器

LoRA（Low-Rank Adaptation）的洞察：微调的权重变化 ΔW 是**低秩**的，可以分解成 `ΔW = B @ A`，其中 A 是 `[r, in]`、B 是 `[out, r]`，r 是很小的秩（如 8、16）。

- base 权重 W 冻结不动（一份，所有 adapter 共享）
- 每个任务只存小矩阵 A、B（r×维度，比 full W 小几个数量级）
- 前向：`output = W @ x + scaling * B @ A @ x`

10 个任务：1 份 base W + 10 份小 (A,B)，显存几乎不增。**这就是 LoRA 的魅力——用低秩分解把微调成本降几个数量级**。

### 1.3 热加载：一个服务，多 adapter 混合

vLLM 的进阶：不只是"加载时选一个 adapter"，而是**运行时动态切换 + 同 batch 混合**：
- 请求 1 用 adapter A，请求 2 用 adapter B，请求 3 用 adapter C——**同一个 forward batch 里**
- adapter 可热加载/卸载（不停服务）
- base 模型只存一份，所有 adapter 共享

这是多租户 LLM 服务的刚需（不同租户/任务用不同 adapter，但共享 base 省显存）。

> 💡 面试一句话答：**LoRA 把微调权重变化分解为低秩 B@A，base 权重共享、每 adapter 只存小矩阵；vLLM 把所有 adapter 的 A/B 预先 stack 成大张量，用 punica wrapper 在一个 batch 里按 token 的 adapter index 分别应用对应 LoRA，实现单 base 模型 + 多 adapter 混合服务，adapter 可热加载。**

---

## 二、LoRA 的数学（你量化方向会很熟悉）

对一层线性层 `y = W @ x`（W 是 `[out, in]`）：

**全参数微调**：学一个完整的 `ΔW [out, in]`，参数量 `out × in`。

**LoRA**：把 ΔW 分解：
```
ΔW = B @ A
A: [r, in]    (down-projection，降维到 r)
B: [out, r]   (up-projection，升维回 out)
```
参数量从 `out × in` 降到 `r × (out + in)`。比如 out=in=4096，r=8：full 是 16M 参数，LoRA 只有 65K——**省 250 倍**。

前向：
```
y = W @ x + scaling * B @ (A @ x)
```
`scaling` 通常 = `alpha / r`（alpha 是超参），控制 LoRA 的影响强度。A 初始化随机，B 初始化为 0（训练开始时 ΔW=0，不破坏 base）。

> 💡 **和你量化方向的联系**：LoRA 的"低秩"和量化的"低比特"是两种"压缩信息"的方式。有人研究 **LoRA + 量化组合**（QLoRA：base 量化 + LoRA 微调），以及 **量化感知的 LoRA**（让 LoRA 适配量化 base）。你感悟区可以深挖这个交叉。

---

## 三、核心机制①：Stacked 预分配（base_linear.py:129）

vLLM 不为每个 adapter 单独存 A/B，而是**预分配一个容纳所有 adapter 的大张量**：

```python
# base_linear.py:129
self.lora_a_stacked = tuple(
    torch.zeros(num_adapters, 1, max_lora_rank, lora_a_out_size, ...)
    for s_index in range(self.n_slices)
)
# base_linear.py:140
self.lora_b_stacked = tuple(
    torch.zeros(num_adapters, 1, lora_b_out_size, max_lora_rank, ...)
    for s_index in range(self.n_slices)
)
```

`lora_a_stacked[index]` 就是第 index 个 adapter 的 A 矩阵。`set_lora`（158行）把指定 adapter 的 A/B 拷进对应 index 槽位：

```python
def set_lora(self, index, lora_a, lora_b, ...):
    self.reset_lora(index)   # 先清零该槽
    self.lora_a_stacked[0][index, 0, :r, :].copy_(lora_a)
    self.lora_b_stacked[0][index, 0, :r, :].copy_(lora_b)
```

**为什么 stack？** 因为 GPU 批量 GEMM 比逐 adapter 串行快得多。stack 成 `[num_adapters, ...]` 后，一次 batched GEMM 能同时算所有 adapter 的 `B @ A @ x`。

---

## 四、核心机制②：混合 batch（punica_wrapper）

这是 LoRA 热加载的魔法核心。一个 batch 里有 M 个 token，每个 token 可能用不同 adapter。怎么算？

`_apply_lora_to_output`（base_linear.py:215）调：
```python
self.punica_wrapper.add_lora_linear(
    output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
)
```

punica_wrapper 内部：每个 token i 有自己的 `adapter_index[i]`，它从 `lora_a_stacked[adapter_index[i]]` 取 A、`lora_b_stacked[adapter_index[i]]` 取 B，算 `B_i @ A_i @ x_i`，加到 `output[i]`。

**关键**：这是**分组 GEMM**（segmented/grouped matmul）——不同 token 组（按 adapter index 分）用不同权重。vLLM 的 punica_wrapper（源自 punica 项目）用 CUDA kernel 高效实现这个分组操作，避免逐 token 串行。

```
batch: [tok0(adapter0), tok1(adapter0), tok2(adapter1), tok3(adapter1)]
       ↓ 按 adapter index 分组
       组0(tok0,tok1) 用 lora_a_stacked[0], lora_b_stacked[0]
       组1(tok2,tok3) 用 lora_a_stacked[1], lora_b_stacked[1]
       各自算 B@A@x，结果拼回去
```

这和第10讲 MoE 的"token 按专家排序"是**同一个思想**——把稀疏的路由（MoE 选专家 / LoRA 选 adapter）转成分组批量计算。

---

## 五、核心机制③：LoRARequest 与热加载生命周期

`vllm/lora/request.py` 的 `LoRARequest`：每个请求带一个 `lora_int_id`（adapter 的整数 ID）和 `lora_local_path`。

生命周期：
1. **注册**：启动时或运行时，`load_lora_adapter(lora_request)` 把 adapter 的 A/B 加载进 `lora_a/b_stacked` 的某个 index
2. **请求**：用户请求带 `lora_int_id`，调度器知道这个请求用哪个 adapter
3. **混合**：forward 时，batch 里每个 token 的 `adapter_index` 指向 stacked 张量的槽位
4. **卸载**：`unload_lora_adapter` 释放槽位（ref_cnt 归零），供新 adapter 复用

这和第9讲 prefix cache 的 ref_cnt 共享、第6讲的请求调度都呼应——**LoRA adapter 也是一种"共享资源"，需要引用计数管理**。

---

## 六、LoRA 的工程挑战

1. **max_lora_rank**：所有 adapter 的 rank 必须 ≤ `max_lora_rank`（因为 stacked 张量按最大秩预分配）。rank 不同的 adapter 要 padding。
2. **max_loras**（同时加载的 adapter 数）：stacked 张量的第一维大小，决定显存。多了显存涨，少了频繁换入换出。
3. **LoRA + 量化**：base 是量化的（如 AWQ/FP8），LoRA 的 A/B 是 fp16。叠加时 `W_quant @ x + B @ A @ x` 要处理 dtype 不一致。vLLM 的量化层和 LoRA 层要协作（第1讲的 QuantizeMethodBase + 本讲的 BaseLinearLayerWithLoRA）。
4. **长尾 adapter**：如果 adapter 数远超 max_loras，会出现频繁换入换出（thrashing），类似 OS 的页面抖动。

---

## 七、把第十一讲和前十讲连起来

| 讲次 | 关系 |
|------|------|
| 第1~4讲（量化） | LoRA 的 A/B 可量化；QLoRA = 量化 base + LoRA；LoRA 叠加时要处理量化 base 的 dtype |
| 第6讲（Continuous Batching） | LoRA adapter index 是 batch 调度的一部分（每个请求带 lora_int_id） |
| 第9讲（Prefix Cache） | LoRA 的 extra_keys 影响 prefix cache hash（不同 adapter 不能共享 KV，第9讲 extra_keys） |
| 第10讲（MoE） | LoRA 和 MoE 的"分组计算"思想一致（adapter index 分组 vs expert 分组） |
| **第11讲（LoRA）** | **多 adapter 共享 base，混合 batch 服务** |

**LoRA 是"模型外部的多路复用"（多 adapter 共享 base），MoE 是"模型内部的多路复用"（多专家稀疏激活）**。两者都是用"共享 + 稀疏"省资源，但维度不同。面试被问"怎么高效服务多个微调模型"，答案就是 LoRA 热加载。

---

## 八、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：LoRA 数学（基础）
1. 一个 `in=4096, out=4096, r=8` 的 LoRA，相比 full 微调省多少参数？（算具体倍数）
2. 为什么 LoRA 的 B 初始化为 0、A 初始化随机？（提示：训练开始时 ΔW 应为多少，才不破坏 base）
3. `scaling = alpha / r`。alpha 越大，LoRA 影响越大还是越小？为什么除以 r？

### 任务 B：stacked 与混合（核心）
4. 读 `base_linear.py:129-151`。`lora_a_stacked` 的 shape 各维分别代表什么？为什么第一维是 `num_adapters` 而不是 1？
5. 读 `base_linear.py:215 _apply_lora_to_output`。`punica_wrapper.add_lora_linear` 做了什么？它怎么让一个 batch 里不同 token 用不同 adapter？
6. `set_lora`（158行）先 `reset_lora(index)`（153行清零）再 copy。为什么必须先清零？（提示：rank padding——新 adapter 的 r 可能小于槽位的 max_lora_rank，残留旧值）

### 任务 C：生命周期与交互（机制）
7. `LoRARequest` 的 `lora_int_id` 在整个流程里起什么作用？从用户请求到 forward 时取 adapter，这个 ID 怎么流转？
8. LoRA adapter 和 prefix cache（第9讲）的交互：两个请求用不同 adapter，即使 prompt 完全相同，它们的 KV cache 能共享吗？为什么？（提示：第9讲的 extra_keys 包含 lora_id）
9. 思考题：QLoRA（量化 base + LoRA），base 是 4bit 量化的，LoRA 是 fp16。前向 `W_4bit @ x + B_16 @ A_16 @ x` 的 dtype 不一致怎么处理？这种组合相比纯 fp16 base + LoRA，省什么、亏什么？

---

## 九、干中学实践任务（核心！）

> 在 `practice_lora.py` 里实现一个完整的 mini LoRA。
> 依赖：仅标准库（`random`）。不需要装 vllm/torch。
> 设计哲学：你不只实现单 LoRA 叠加，还要实现**多 adapter 混合 batch**——一个 batch 里不同 token 用不同 adapter，这是 vLLM LoRA 的灵魂。

### 实践 1：单 LoRA 叠加（热身）
实现：
- `lora_forward(x, W_base, A, B, scaling) -> y`：`y = W_base @ x + scaling * B @ A @ x`
- `matmul(a, b)`：辅助矩阵乘
- 构造小 W_base `[out,in]`、A `[r,in]`、B `[out,r]`，验证 `lora_forward` 与手算 `W@x + scaling*B@(A@x)` 一致

验证：scaling=0 时输出应 == base 输出（LoRA 不生效）；scaling>0 时输出 = base + 修正。

### 实践 2：多 adapter stacked + 混合 batch（核心）
实现 `lora_mixed_batch(tokens, W_base, adapters_A, adapters_B, adapter_indices, scaling)`：
- `adapters_A[i]`、`adapters_B[i]`：第 i 个 adapter 的 A/B（模拟 stacked 张量）
- `adapter_indices[t]`：token t 用哪个 adapter（index）
- 每个 token：`y[t] = W_base @ x[t] + scaling * adapters_B[idx[t]] @ adapters_A[idx[t]] @ x[t]`

验证：构造 2 个 adapter、4 个 token（前2用adapter0、后2用adapter1），检查每个 token 用了正确的 adapter，且与逐 token 独立计算一致。

### 实践 3：LoRA 热加载生命周期（进阶）
实现 `LoRAManager`：
- `load_adapter(adapter_id, A, B)`：分配一个空闲 index 槽，存入 A/B（模拟 set_lora）
- `unload_adapter(adapter_id)`：释放 index 槽（ref_cnt 归零，可复用）
- `serve(tokens, adapter_ids)`：根据每个 token 的 adapter_id 查 index，混合 batch 计算
- 维护 `adapter_id -> index` 映射 + 槽位空闲列表

验证：加载 2 个 adapter（index 0,1），服务一批混合请求，卸载 adapter0，加载新 adapter（应复用 index 0），检查生命周期正确。

> 💡 实践 2 是灵魂。要点：① base 权重共享，只有 LoRA 部分按 adapter 不同 ② 每个 token 查自己的 adapter_index 取对应 A/B ③ 数学上 `W@x + s*B_i@(A_i@x)`。这模拟了 punica 的分组计算——真实 vLLM 用 CUDA kernel 批量做，你用 Python 循环，但算法一致。

---

## 十、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_lora.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_lora.py 运行结果）

---

## 十一、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现 LoRA 后，你对"低秩分解省参数"的理解？scaling 的作用你体会了吗？② 混合 batch 里不同 token 用不同 adapter，这个"单 base 多 adapter"的设计对多租户服务的价值？③ 热加载生命周期（load/unload/复用 index），和 OS 的内存管理像吗？

（完成实践后填写）

---

## 十二、个人复盘感悟（留给你写）

> 你是量化方向研究生、AI Infra 求职者，建议角度：① LoRA 的"低秩"和量化的"低比特"都是信息压缩，你怎么看它们的数学联系（低秩 ≈ 压缩到小子空间，量化 ≈ 压缩到少 bit）？② QLoRA（量化 base + LoRA）是你研究方向的高价值交叉，base 量化误差会如何影响 LoRA 训练/推理？③ LoRA 的 stacked 预分配 + 分组计算，和 MoE 的"按专家排序批量"是同一思想，你怎么评估这种"稀疏路由→分组批量化"范式的普适性？④ 多 adapter 混合 batch 对多租户 LLM 服务（每个租户一个 adapter）的价值？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**进入多 adapter 服务新领域**。完成后告诉我下一步：
> - **① 结构化输出**（grammar constrained generation，另一个高频特性）
> - **② KV offload / 多模态**
> - **③ 回量化**：QLoRA 深水区（量化 base + LoRA 的交互）
> - **④ EP（Expert Parallelism）**：MoE 的多卡专家分布（第10讲延伸）
> - 或你指定的
