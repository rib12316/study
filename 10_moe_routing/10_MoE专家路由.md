# 特性 #10：MoE 专家路由 —— 稀疏激活怎么用 topk gating 选专家

> 学习阶段：AI Infra 基础储备 / 模型架构（新领域，从量化/调度切换）
> 对应源码：`vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`（gating）+ `vllm/model_executor/layers/fused_moe/fused_moe.py:1460`（token 按专家排序）+ `fused_moe/layer.py`（FusedMoE 层）
> 本讲定位：前 9 讲都是"怎么高效跑一个稠密模型"。但当前 LLM 主流（DeepSeek-V3、Llama-4、Qwen3-MoE）是 **MoE（Mixture of Experts）**——参数量大但每次只激活一小部分。这一讲回答：**MoE 怎么决定每个 token 用哪几个专家？又怎么高效地批量算？** 这是当前 LLM 架构的核心，面试高频。
> 干中学原则：本讲你要**亲手实现一个 mini MoE**——gate 路由（topk softmax）+ 稀疏专家分发 + 加权融合输出。这是 MoE 算法的亲手实现，理解它就懂了所有 MoE 变体。

---

## 一、为什么需要 MoE？（背景，面试必答）

### 1.1 稠密模型的算力瓶颈

传统 Transformer 每层是稠密 FFN：`hidden → 4×hidden → hidden`，每个 token 都过完整 FFN。想要更大模型 → 参数和算力都线性涨。100B 参数的稠密模型，每个 token 的 forward 要算 100B 参数——**又贵又慢**。

### 1.2 MoE：参数大但激活稀疏

MoE（混合专家）的思路：**把一个大 FFN 拆成 N 个小 FFN（专家），每个 token 只激活其中 top-k 个**。

举例 DeepSeek-V3：256 个专家，每个 token 只激活 8 个。总参数 671B，但每个 token 只算 ~37B 的激活参数。**参数容量大（知识多），单 token 算力小（快）**。这就是 MoE 的魅力——用稀疏激活打破"参数量=算力"的耦合。

### 1.3 核心问题：怎么选专家？

每个 token 怎么知道该用哪 8 个专家？这就是 **gating / routing**（门控/路由）的问题。答案：一个小的线性层（gate）给每个 token 对每个专家打分，选 top-k 个分高的。

> 💡 面试一句话答：**MoE 用一个 gate 线性层给每个 token 对所有专家打分，选 top-k 个最高分的专家（topk gating），只算这 k 个专家的 FFN 并按 gate 权重加权融合——实现"大参数量、小激活算力"。vLLM 用 fused MoE kernel 把"按专家排序 token + 批量 GEMM"融合，避免 k 次串行专家计算。**

---

## 二、核心算法：Top-k Gating（fused_topk_router.py:69）

### 2.1 Gate 打分

```python
# gate 是一个线性层：hidden_dim → num_experts
gating_output = gate(hidden_states)   # [num_tokens, num_experts]
```
每个 token 得到对每个专家的"亲和度分数"。

### 2.2 两种打分函数：softmax vs sigmoid

`fused_topk`（router/fused_topk_router.py:69）支持两种：

**Softmax routing**（传统，Mixtral/GPT-4 风格）：
```python
# 对 num_experts 维度 softmax，再选 top-k
probs = softmax(gating_output, dim=-1)   # 概率分布
topk_weights, topk_ids = topk(probs, k)  # 选 k 个最大
if renormalize:
    topk_weights = topk_weights / topk_weights.sum()   # 归一化（权重和=1）
```

**Sigmoid routing**（DeepSeek-V3 风格）：
```python
# 每个 expert 独立 sigmoid（不互斥），再选 top-k
scores = sigmoid(gating_output)          # 每个专家独立 0~1
topk_weights, topk_ids = topk(scores, k)
```

**区别**：softmax 是"专家互斥竞争"（一个专家高分会压低其他），sigmoid 是"专家独立评分"（多个专家可以同时高分）。DeepSeek-V3 用 sigmoid + renormalize。

### 2.3 renormalize 的作用

`renormalize=True`（默认）：top-k 权重归一化到和=1。这样加权融合时不会因为"漏掉其他专家"而输出幅度变小。`router/fused_topk_router.py` 的 `FusedTopKRouter.__init__`（123行）默认 `renormalize=True`。

---

## 三、专家计算：Fused MoE 的核心技巧

朴素实现：每个 token 选了 k 个专家 → 对每个专家，收集分配给它的 token → 逐个专家做 FFN。这是 **k 次（或 num_experts 次）串行小 GEMM**，效率极低（每次 GEMM 太小，GPU 吃不饱）。

### 3.1 Fused MoE：token 按专家排序 + 批量 GEMM

vLLM 的核心优化（`fused_moe.py:1460 _prepare_expert_assignment`）：

1. **收集所有 (token, expert) 对**：M 个 token，每个选 k 个专家 → M×k 个 (token, expert) 配对
2. **按专家排序**（`sorted_token_ids`）：把分配给专家 0 的 token 排一起，专家 1 的排一起……
3. **批量 GEMM**：一个 Triton kernel 扫描排序后的数组，遇到同一专家的 token 块就批量算 FFN（一次大 GEMM 而非多次小 GEMM）

```python
# fused_moe.py:161（Triton kernel 核心）
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
# 用 off_experts 索引正确的专家权重
a_ptrs = a_ptr + ...               # token 的输入
b_ptrs = b_ptr + off_experts * ... # 对应专家的权重
```

`expert_ids` 记录"这个 block 该用哪个专家的权重"。这样**一次 kernel 调用处理所有 token×所有被选专家**，GPU 充分饱和。

### 3.2 加权融合

每个 token 的 k 个专家输出，按 `topk_weights` 加权求和：
```python
output = sum(topk_weights[i] * expert_i_output for i in range(k))
```

这就是 MoE 的最终输出。

---

## 四、MoE 的工程挑战（面试深水区）

1. **负载均衡（load balancing）**：如果所有 token 都选同一个专家，那个专家成为瓶颈，其他专家闲置。训练时要加 aux loss 鼓励均匀分布；推理时 vLLM 靠 expert parallelism（EP）把不同专家分到不同 GPU。
2. **显存**：所有专家的权重都要加载（即使每次只用 k 个）。这是 MoE 的"参数大"代价。**量化在这里价值巨大**——把 256 个专家量化到 4bit，显存省 4 倍（这是你量化方向的直接应用）。
3. **EP（Expert Parallelism）**：把专家分布到多卡，每卡只持有部分专家。`expert_map`（fused_moe.py:1385）记录"全局专家 ID → 本卡专家 ID"，token 路由到别的卡的专家需要 all2all 通信。
4. **shared expert**（DeepSeek）：除了 routed experts（topk 选的），还有一个所有 token 都走的 shared expert，吸收公共知识，减少 routed expert 的负担。

---

## 五、把第十讲和前九讲连起来

| 讲次 | 关系 |
|------|------|
| 第1~4讲（量化） | MoE 的专家权重可量化（你量化方向的核心应用场景，vLLM 的 Fp8MoEMethod/AutoAWQMoEMethod） |
| 第6讲（Continuous Batching） | MoE 的 token 按专家排序，和 batch 调度深度耦合 |
| 第7讲（Chunked Prefill） | MoE 的稀疏性让 prefill 的算力需求不同于稠密 |
| **第10讲（MoE）** | **模型架构层的稀疏，和系统层的稀疏（量化）是两个维度** |

**MoE 是"架构稀疏"（每次少算专家），量化是"数值稀疏"（每个值少占位）**。两者正交且可叠加：MoE + 量化 = 又省算力又省显存，是当前大模型部署的主流。面试被问"MoE 怎么部署"，你的答案就该串联这两条线。

---

## 六、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：路由（基础）
1. 读 `router/fused_topk_router.py:69 fused_topk`。softmax 和 sigmoid 两种打分函数的区别是什么？为什么 DeepSeek-V3 用 sigmoid？
2. `renormalize=True`（123行默认）的作用？如果设为 False，输出幅度会有什么问题？
3. `gate`（layer.py 的参数）是什么形状的线性层？输入输出维度各是多少？

### 任务 B：fused 计算（核心）
4. 读 `fused_moe.py:1460 _prepare_expert_assignment`。`sorted_token_ids` 是怎么生成的？为什么要把 token 按专家排序？
5. `fused_moe.py:161`（Triton kernel）里 `off_experts = tl.load(expert_ids_ptr + pid_m)`。这个 `off_experts` 决定了什么？它是怎么让一个 kernel 处理多个专家的？
6. 思考题：朴素 MoE（逐专家串行 GEMM）vs fused MoE（排序+批量），为什么后者快？（提示：GEMM 的算术强度和 batch 大小）

### 任务 C：量化 + EP（机制）
7. vLLM 的 `Fp8MoEMethod`（第1讲引用）和 `AutoAWQMoEMethod`（第2讲引用）说明了什么？MoE 量化相比稠密量化，收益更大还是更小？为什么？
8. `expert_map`（fused_moe.py:1385）的作用？在 EP=4（4卡）下，一个路由到专家 5 的 token，如果专家 5 在卡 1，卡 0 怎么处理？
9. 思考题：DeepSeek-V3 的 shared expert（所有 token 都走）和 routed expert（topk 选）分离设计，对负载均衡有什么帮助？

---

## 七、干中学实践任务（核心！）

> 在 `practice_moe.py` 里实现一个完整的 mini MoE。
> 依赖：仅标准库（`math`, `random`）。不需要装 vllm/torch。
> 设计哲学：你不读 vLLM 的 fused kernel，而是用纯 Python 重建 MoE 的**算法逻辑**（gate→topk→专家分发→加权融合），并对比"朴素逐专家"vs"按专家排序批量"两种实现思路。

### 实践 1：Gate + Top-k 路由（热身）
实现：
- `gate(hidden: list[float], gate_weights: list[list[float]]) -> list[float]`：线性层打分，返回对每个专家的分数 `[num_experts]`
- `topk_softmax(scores: list[float], k: int, renormalize: bool) -> (list[float], list[int])`：softmax 后选 top-k，返回 (权重列表, 专家ID列表)

验证：构造 4 个专家的 gate，对一组 scores 检查 topk 选对、softmax 权重和=1（renormalize 时）。

### 实践 2：MoE 前向（核心）
实现 `moe_forward(tokens: list[list[float]], experts_ffn, gate_weights, k) -> list[list[float]]`：
- 每个 token：gate 打分 → topk 选 k 个专家 → 算这 k 个专家的 FFN → 加权融合
- `experts_ffn[e](x)`：模拟专家 e 的 FFN（简单函数即可，如线性变换）

验证：单个 token、k=2、4 专家，检查输出是 top-2 专家 FFN 的加权融合。

### 实践 3：朴素 vs 按专家排序（进阶）
实现两种 MoE 计算方式并对比：
- **朴素**：遍历每个专家，收集分配给它的 token，算 FFN（专家外循环）
- **按专家排序**：先把所有 (token, expert) 对按 expert 排序，再按专家分组批量算（模拟 fused MoE 的思路）

验证：两种方式输出完全相同（数学等价）；统计"专家外循环次数"，证明排序版更利于批量（实际场景下更快）。

> 💡 实践 2 是灵魂。要点：① gate 是普通线性层 ② topk softmax 要先 softmax 再选 topk（不是选 topk 再 softmax）③ 加权融合用 gate 的权重 ④ 只算被选的 k 个专家（稀疏）。实践 3 让你体会"为什么 fused MoE 快"——排序后同一专家的 token 聚集，能批量算大 GEMM。

---

## 八、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_moe.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_moe.py 运行结果）

---

## 九、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现 topk softmax 后，你对"先 softmax 再选"的理解？② 朴素 vs 排序两种实现，输出相同但效率不同，你对"算法等价但工程不等价"的体会？③ MoE 的稀疏性，结合你量化方向，你怎么看 MoE 量化的特殊价值？

（完成实践后填写）

---

## 十、个人复盘感悟（留给你写）

> 你是量化方向研究生、AI Infra 求职者，建议角度：① MoE 的"架构稀疏"和量化的"数值稀疏"是两个正交维度，组合使用的上限你怎么评估？② fused MoE 用"排序+批量"把 k 次小 GEMM 变 1 次大 GEMM，这种"用排序换批量"的思路你在别处见过吗（如分布式 shuffle）？③ MoE 量化时，gate 是否要量化（gate 影响路由，量化误差可能导致选错专家）？这是你研究方向的好问题。④ DeepSeek-V3 的 sigmoid routing vs Mixtral 的 softmax routing，对量化误差的鲁棒性谁更强？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**进入模型架构新领域**。完成后告诉我下一步：
> - **① LoRA 热加载**（另一新领域，多 adapter 服务）
> - **② 结构化输出**（grammar constrained generation）
> - **③ KV offload / 多模态**
> - **④ 回量化**：MoE 量化的特殊问题（gate 量化、专家负载均衡与量化交互）
> - 或你指定的
