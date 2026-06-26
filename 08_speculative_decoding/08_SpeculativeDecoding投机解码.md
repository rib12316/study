# 特性 #8：Speculative Decoding —— 用"猜"来加速解码，且永远不吃亏

> 学习阶段：AI Infra 基础储备 / 解码加速（v1 热点特性）
> 对应源码：`vllm/v1/spec_decode/`（EAGLE/Medusa/N-gram/draft model/suffix decoding）+ `vllm/v1/sample/sampler.py:358` + `vllm/v1/spec_decode/metrics.py`
> 本讲定位：前 7 讲都在"省资源/提利用率"，这一讲换到**绝对加速**——同一模型、同一硬件，怎么让生成更快？答案是用一个小模型/启发式"猜"接下来几个 token，大模型"一次性验证"。这是 2023-2024 最重要的 LLM 推理加速范式之一，v1 把它做进了核心调度。
> 干中学原则：本讲你要**亲手实现一个 draft-verify 循环**——draft 预测 K 个 token、target 并行验证、accept/reject、计算加速比。这是教科书算法的亲手实现，理解它你就懂了所有 spec decode 变体。

---

## 一、为什么需要投机解码？（背景，面试高频）

### 1.1 自回归解码的瓶颈

LLM 生成是自回归的：每生成 1 个 token，要做一次完整 forward（过所有层、读所有 KV cache），只产出 1 个 token。这是 **memory-bound**——GPU 大部分时间在等显存搬数据，算力闲置（arithmetic intensity 极低）。

后果：生成 100 个 token 要 100 次 forward，每次 forward 都要等内存。**算力严重浪费**。

### 1.2 核心洞察：可以"并行验证多个 token"

假设我们有一段已生成序列，**猜**接下来 K 个 token 是 `[t1, t2, ..., tK]`。怎么验证猜得对不对？

朴素：一个一个验证，每次 1 次 forward——没省。

**投机解码的魔法**：target 大模型可以**一次 forward 同时算 K+1 个位置的 logits**（把 `[当前最后一个真实token, t1, t2, ..., tK]` 作为一次输入）。因为 attention 的并行性，这 K+1 个位置的预测可以**一起算出来**。然后：
- 比较 target 在位置 i 的预测 vs draft 的 t_i
- 接受连续匹配的前缀，第一个不匹配处截断

**关键**：这次"算 K+1 个位置"的 forward，**耗时和算 1 个位置几乎一样**（因为 memory-bound，主要时间在读 KV，多算几个 query 的开销可忽略）。所以：
- 如果猜对了 K 个 → 一次 forward 产出 K+1 个 token（含 bonus）→ **快 K+1 倍**
- 如果全猜错 → 一次 forward 产出 1 个 token（bonus）→ **没亏**（和普通 decode 一样）

> 💡 面试一句话答：**投机解码用一个小模型/启发式 draft K 个 token，大模型一次 forward 并行验证（接受匹配前缀+1个bonus），利用 memory-bound 下"多算几个位置几乎不增加耗时"的特性，猜对就加速、猜错也不亏，期望加速比 = 1 + 接受长度。**

### 1.3 为什么"永远不吃亏"？

这是面试必答的精髓。看 `metrics.py:114`：
```python
mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)
```

即使 draft 一个都没猜对（`num_accepted=0`），`mean_acceptance_length = 1`——每次投机还是产出 1 个 token（bonus），和不开投机解码完全一样。**所以投机解码的下界就是"不开"**，它只会加速不会减速（忽略 draft 自身开销）。

---

## 二、Draft 从哪来？五种 Proposer

vLLM v1 支持多种 draft 来源（`vllm/v1/spec_decode/`），核心区别是"谁猜"：

| Proposer | 怎么猜 | 代价 | 适用 | 文件 |
|----------|--------|------|------|------|
| **Draft Model** | 另一个小 LLM | 要加载小模型，占显存 | 通用，质量高 | `draft_model.py` |
| **EAGLE** | 一个轻量 head（基于 target 的 hidden state） | 小，但要训练 | 当前主流，质量高 | `eagle.py` |
| **Medusa** | 多个并行 head | 小，要训练 | 早期方案 | `medusa.py` |
| **N-gram** | 在已生成序列里找重复 n-gram | **零成本**（无模型！） | 重复性强的文本（代码/文档） | `ngram_proposer.py` |
| **Suffix Decoding** | 后缀树缓存 | 零成本 | 高重复（RAG/agent） | `suffix_decoding.py` |

**N-gram 是最适合教学的**：它用 KMP 算法（LPS 数组）在已生成 token 里找最长匹配的 n-gram，预测接下来的 K 个 token。无模型、纯启发式，原理简单。EAGLE 是工业主流（用 target 的 hidden state 训练一个小 head），效果最好但要训练。

> 💡 你的量化背景在这里有联系：draft model 可以是 target model 的量化版（比如 target 是 fp16，draft 是 4bit）。这也是"量化+投机解码"的常见组合——用便宜的量化小模型 draft，贵的大模型 verify。

---

## 三、核心算法：Draft-Verify 循环

### 3.1 一次投机 step 的完整流程

```
1. Draft 阶段：proposer 预测 K 个 token [d1, d2, ..., dK]
   - N-gram: 在历史里找匹配，取后面 K 个
   - EAGLE: 小 head forward K 次（或并行）
   
2. Verify 阶段：target 一次 forward
   输入 = [last_real_token, d1, d2, ..., dK]   （K+1 个位置）
   输出 = K+1 个 logits → K+1 个 target 预测 [p0, p1, ..., pK]
   
3. Accept/Reject：从左到右比较
   - p0 vs d1: 相等则接受 d1，继续比 p1 vs d2...
   - 第一个不匹配处 j：接受 d1..d_{j-1}，丢弃 d_j..d_K
   - 接受 p_j 作为 bonus（target 的预测，肯定对）
   
4. 结果：产出 [d1, ..., d_{j-1}, p_j] 共 j+1 个 token（j=0 时只有 bonus）
```

### 3.2 为什么 target 能"一次算 K+1 个位置"？

这是工程核心。回忆第6/7讲：continuous batching 里，一个请求在一个 step 可以 query 多个 token（prefill chunk 就是多 token query）。spec decode 的 verify 就是**让一个 decode 请求在这个 step query K+1 个 token**（K 个 draft + 1 个真实位置），target 对这 K+1 个位置并行算 attention 和 logits。

vLLM 用 `num_lookahead_tokens`（scheduler.py 的 allocate_slots 参数）给 spec token 预留 KV cache 槽位。这和 chunked prefill 共享同一套"多 token query"基础设施——**spec decode 是 continuous batching 框架的又一特例**。

### 3.3 接受策略：exact match vs 概率拒绝

- **N-gram/Medusa**：exact match（target 的 argmax == draft token 才接受）。简单但严格。
- **Draft Model/EAGLE**：可用**概率拒绝采样**（rejection sampling）——即使 target 和 draft 的 token 不同，按概率也可能接受 draft 的 token。这保证**输出分布和纯 target 解码完全一致**（数学上等价）。vLLM 默认对小模型 draft 用概率拒绝，对 N-gram 用 exact match。

> 💡 概率拒绝采样是 spec decode 论文的精髓（Leviathan 2023）：`accept draft token x with prob = min(1, p_target(x)/p_draft(x))`。它让投机解码"不只是快，而且输出和原模型逐 token 同分布"。本讲实践用 exact match（简单），概率拒绝作为思考题。

---

## 四、加速比的数学

这是面试最爱问的"量化分析"。设：
- K = draft token 数（num_speculative_tokens）
- α = 接受概率（每个位置 draft 猜对的概率，假设独立）
- T = 一次 target forward 时间，t = 一次 draft 时间

**期望接受长度**（每次投机产出的 token 数）：
```
E[接受长度] = 1 + α + α² + ... + α^(K-1) + α^K
            = (1 - α^(K+1)) / (1 - α)        （等比数列，含 bonus）
```
- α→1（全猜对）：E = K+1（最大加速）
- α→0（全猜错）：E = 1（bonus，不亏）

**加速比**（近似，忽略 draft 开销）：
```
speedup ≈ E[接受长度] / (1 + t/T)
```
draft 越快（t/T 小）、接受率越高（α 大），加速比越大。这就是为什么要用"便宜"的 draft（N-gram 零成本，或量化小模型）。

`metrics.py:108` 的 `acceptance_rate = accepted/draft * 100` 和 `mean_acceptance_length = 1 + accepted/drafts` 就是这套数学的运行时度量。

---

## 五、Spec Decode 在 v1 调度里的位置

回忆第6讲的统一抽象："每个请求有 num_computed vs num_tokens_with_spec"。spec decode 的"spec"就是 `num_tokens_with_spec`！

```python
# scheduler.py 的注释（第6讲引用过）
# num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)
```

一个开了 spec decode 的请求，它的 `num_tokens_with_spec` 比 `num_tokens` 多出 K（spec token）。调度器给它的 budget 是 K+1（verify 一次算这么多），KV cache 也要预留 K 个槽（`num_lookahead_tokens`）。

**所以 spec decode 不是独立子系统，而是"让某些请求的 num_tokens_with_spec 多出 K 个 spec token"**。它复用了 continuous batching + chunked prefill 的全部基础设施。这是 v1 设计的统一性之美。

---

## 六、把第八讲和前七讲连起来

| 讲次 | 维度 | 和 spec decode 的关系 |
|------|------|---------------------|
| 第1~4讲（量化） | 省权重 | draft model 可用量化版（4bit 小模型 draft） |
| 第5讲（PagedAttention） | 省 KV | spec token 要占 KV 槽，靠 paged 分配 |
| 第6讲（Continuous Batching） | 统一调度 | spec decode 是 num_tokens_with_spec 的特例 |
| 第7讲（Chunked Prefill） | 长 prompt | verify 的"多 token query"和 chunked 共享 varlen 机制 |
| **第8讲（Spec Decoding）** | **解码加速** | **用 draft+verify 把 memory-bound 变成算力优势** |

**前7讲是"省/用资源"，第8讲是"绝对加速"**。合起来：省资源（量化+paged）→ 高效用资源（continuous batching+chunked）→ 加速（spec decode）。这是 vLLM 性能的完整栈。

---

## 七、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：架构与 proposer（基础）
1. `dir vllm/v1/spec_decode`。v1 支持哪几种 proposer？其中哪些**不需要加载额外模型**（零成本 draft）？
2. 读 `ngram_proposer.py:207 _find_longest_matched_ngram_and_propose_tokens`。它用什么算法找最长匹配？（提示：LPS 数组，KMP 的核心）为什么用这个而不是暴力匹配？
3. EAGLE（`eagle.py`）和 N-gram 的本质区别是什么？为什么 EAGLE 接受率通常更高？（提示：EAGLE 用 target 的 hidden state，信息更丰富）

### 任务 B：验证与接受（核心）
4. 读 `sampler.py:358 _combine_outputs_with_spec_tokens`。它把 output 和 spec token 怎么组合？为什么需要这个组合（提示：sampling 时的 output_token_ids 要包含 draft 才能算 penalty）？
5. 投机解码的 verify 阶段，target 一次 forward 算 K+1 个位置。这 K+1 个位置的 logits 是怎么和 K 个 draft token 比较的？（结合讲义 3.1，描述 accept/reject 的顺序）
6. `metrics.py:114`：`mean_acceptance_length = 1 + num_accepted_tokens/num_drafts`。为什么加 1？这个 1 代表什么？（提示：bonus token）

### 任务 C：调度集成（机制）
7. spec decode 的"spec token"在调度器里对应哪个字段？（提示：第6讲引用的 `num_tokens_with_spec`）这个字段比 `num_tokens` 多出了什么？
8. 思考题：verify 阶段算 K+1 个位置，但只接受 j 个（j≤K）。那剩下 K+1-j 个位置的 KV cache 计算是不是"浪费"了？vLLM 怎么处理这些被拒绝 token 的 KV 槽？（提示：回忆 PagedAttention 的 block 释放，scheduler.py 里 `num_lookahead_tokens` 和被拒 token 的 slot 回收）
9. 思考题：draft model 用 4bit 量化小模型，target 用 fp16 大模型。这种"量化+spec decode"组合，加速比和纯 fp16 draft 比会更好还是更差？为什么？（考虑 t/T 比值）

---

## 八、干中学实践任务（核心！）

> 在 `practice_speculative_decoding.py` 里实现一个完整的 draft-verify 循环。
> 依赖：仅标准库（`random`）。不需要装 vllm/torch。
> 设计哲学：你不读 vLLM 的 spec decode，而是**重建** draft-verify-accept 算法。用模拟的 target/draft（控制接受率），亲手算出加速比，验证"永不吃亏"。

### 实践 1：模拟 Target/Draft + Accept/Reject（热身）
实现：
- `mock_target_predict(context) -> token`：模拟大模型预测下一个 token（用一个确定函数，如基于 context 的 hash）
- `mock_draft_predict(context, k) -> [token]*k`：模拟 draft，**以概率 p 猜对**（猜对时返回和 target 一样的 token，猜错时返回随机错 token）
- `verify(real_token, draft_tokens, p_accept) -> (accepted_count, bonus_token)`：实现 accept/reject——从左到右比较，连续匹配则接受，第一个不匹配截断，返回接受数和 bonus

验证：构造 p_accept=1.0（全对）应接受全部 K 个 + bonus；p_accept=0.0（全错）应接受 0 + bonus。

### 实践 2：完整 Draft-Verify 循环 + 加速比（核心）
实现 `speculative_decode(target_fn, draft_fn, prompt, num_tokens, K, p_accept) -> (output, stats)`：
- 循环生成 num_tokens 个 token
- 每次：draft K 个 → verify → 接受 j 个 + 1 bonus → 产出 j+1 个
- 记录：总 target forward 次数、总 draft 次数、总接受 token 数
- 算 **实际加速比** = num_tokens / target_forward 次数（对比普通 decode 的 num_tokens 次 forward）
- 算 **理论加速比** = (1 - α^(K+1))/(1-α)（α=p_accept）

验证：p_accept=0.5、K=4，跑 1000 token，对比实际 vs 理论加速比（应接近）。

### 实践 3：N-gram Draft（进阶）
实现一个真实的 N-gram proposer（不靠概率，靠模式匹配）：
- 维护一个"已见 n-gram → 下一个 token"的表
- draft 时：取最后 N 个 token，查表预测接下来的 K 个
- 用一段重复性强的文本（如代码 `def foo(): return foo()`）测试，观察 N-gram 的接受率

验证：重复模式文本的接受率应明显高于随机文本（这是 N-gram spec decode 对代码/文档有效的根本原因）。

> 💡 实践 2 是灵魂。要点：① verify 时一次 forward 算 K+1 个位置（模拟成 1 次 target_fn 调用）② accept 是"连续匹配前缀"，不是独立判断每个 ③ bonus 保证至少产出 1 ④ 加速比 = 产出 token / forward 次数。理论公式 (1-α^(K+1))/(1-α) 要手推，理解等比数列求和。

---

## 九、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_speculative_decoding.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_speculative_decoding.py 运行结果，重点贴加速比对比表）

---

## 十、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实践 2 的实际加速比和理论值差多少？为什么有差距（随机性？）② p_accept=0 时加速比=1（不亏），你亲手验证了吗？这个"下界保证"给你的感觉？③ N-gram draft 在重复文本上接受率多高？这解释了为什么 spec decode 对代码补全特别有效？

（完成实践后填写）

---

## 十一、个人复盘感悟（留给你写）

> 你是量化方向研究生、AI Infra 求职者，建议角度：① "用便宜的 draft 模型猜、贵的大模型验"——这种"不对称计算"的思路你在量化/蒸馏里见过吗（比如用小模型蒸馏大模型）？② 投机解码"永不吃亏"的下界保证，数学上来自 bonus token，这种"带保险的优化"你在别的系统见过吗？③ 概率拒绝采样保证输出分布不变——spec decode 不只是"快"，还"等价"，这种"加速不牺牲质量"的特性对线上服务意味着什么？④ 你做量化的，draft model 用量化版能进一步降 t/T，你怎么评估这个组合的上限？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。完成后告诉我下一步：
> - **① Prefix Caching 深入**（hash 链/eviction/multimodal，第5讲点到）
> - **② 换领域**：MoE 专家路由 / LoRA 热加载 / 结构化输出 / KV offload / 多模态
> - **③ 回量化**：DeepGEMM/Machete 等 Hopper/Blackwell 新 kernel
> - 或你指定的

---

## 附录：学习问答记录（Q&A 疑难澄清）

> 以下是在学习本讲过程中产生的关键提问与详细解答，集中在 spec decode 最容易卡住的三个点：① 验证的并行本质 ② 为何不能让大模型自己并行出 K+1 个 ③ 草稿模型到底快在哪。这三个问题其实同根同源——都来自"**生成 vs 验证的不对称性**"。先把这个根点透，三个疑问一起解开。

### 根：生成与验证的根本不对称

|  | 输入是否已知？ | 能否并行？ |
|---|---|---|
| **生成（autoregressive）** | ❌ 第 i 步的输入 = 第 i-1 步的输出 | ❌ **必须串行**（死锁） |
| **验证（verify）** | ✅ draft 已经把 K+1 个输入全猜出来了 | ✅ **一次并行** |

投机解码的全部魔法，都建立在这个不对称上。

---

### Q1：为什么 target 能"一次算 K+1 个位置"？

**关键认知：验证那一次 forward，本质上就是一次 prefill。**

draft 已经把 K 个 token 猜出来了 `[d1, d2, ..., dK]`，现在把它们当作一个**已知 prompt** 喂给 target，前面拼上真实的 `last_real_token`：

```
输入: [last_real_token, d1, d2, ..., dK]   ← 共 K+1 个，全部已知
```

target 对这条序列做一次普通的 prefill。回忆第 7 讲——**prefill 就是 attention 并行算所有位置，每个位置 i 只看前 i 个（causal）**。于是这一次 forward 同时吐出 K+1 个 logits：

```
位置 0 (last_real_token) 的 logit → 预测"真实 token 之后该是啥" → p0   ← 用来验 d1
位置 1 (d1)               的 logit → 预测"d1 之后该是啥"        → p1   ← 用来验 d2
位置 2 (d2)               的 logit → 预测"d2 之后该是啥"        → p2   ← 用来验 d3
...
位置 K (dK)               的 logit → 预测"dK 之后该是啥"        → pK   ← bonus
```

每个位置的语义，正好就是验证时需要问的问题：
- 验 d1 对不对 → 看 p0（在"真实 token 之后"的预测）和 d1 比
- 验 d2 对不对 → 看 p1（在"真实 token, d1 之后"的预测）和 d2 比
- 验 d3 对不对 → 看 p2（在"真实 token, d1, d2 之后"的预测）和 d3 比

这些问句嵌套在一起——"给定 [last_real]"、"给定 [last_real, d1]"、"给定 [last_real, d1, d2]"……恰好就是 causal attention 一次 forward 算出来的东西。**位置 i 的 logit = "给定前 i 个输入，下一个是啥"的答案**。

所以"一次算 K+1 个位置"不是新魔法，它就是**一次普通 prefill**。target 根本不知道、也不在乎 `d1..dK` 是真实历史还是投机猜测——它只是按 causal 规则算。

**为什么这次 forward 几乎不比算 1 个位置慢？** 因为 decode 是 **memory-bound**：

```
普通 decode（1 个 query）：
  读 KV cache ←── 大头，占总时间 ~90%
  算 1 个 query 的 attention ←── 毛毛雨，算力闲置

投机 verify（K+1 个 query）：
  读 KV cache ←── 还是读一次！KV 内容没变，被 K+1 个 query 复用
  算 K+1 个 query 的 attention ←── 多了点活，但算力本来就闲着
```

**KV 是读一次、被 K+1 个 query 共用的**。多出来的成本只是 K 个额外 query 的算——而算力恰是 decode 时被浪费的那部分。总时间几乎不变，白赚 K 次验证机会。这就是"memory-bound 下多算几个位置几乎不增加耗时"的物理含义。

---

### Q2：验证还不还是得一个一个验？

**不。** 要区分两个动作：① 算出 target 的预测 ② 把 target 预测和 draft 比对。你混淆的"一个一个"，在贵的那步（①）里根本不存在。

**第①步（算预测）——一次并行，不是 K 次。**

潜在误区："要验 d1，得让大模型先**生成**出它的下一个 token，再和 d1 比；要验 d2，又得让大模型生成下一个……所以是 K 次生成。"

错。验证**根本不生成**，验证是**算条件预测**。算"给定 [last_real]，下一个是啥"、"给定 [last_real, d1]，下一个是啥"……这些**输入全部已知**（draft 提供），所以它们能像 prefill 一样**一次性并行算完**（见 Q1）。1 次 forward 出 K+1 个 logits，不是 K 次生成。

为什么生成做不到？因为**生成的输入是未知的**——算 next2 的输入是 next1，而 next1 要等上一次 forward 算完才知道。输入依赖输出，死锁，只能串行。验证没有这个死锁：draft 已经把所有输入摊在桌上。

**第②步（比对）——确实是逐个比，但这步几乎免费。**

```python
for i in range(K):
    if p[i] == d[i+1]:    接受 d[i+1]
    else:                  在这里截断，接受 p[i] 作为 bonus，停
```

这是在 logits 已经算出来之后的**简单比较**，几个 tensor 相等判断，纳秒级，和"再跑一次 forward"完全不是一个量级。而且是**前缀匹配**——第一个不匹配就停，后面不再比。

所以整个验证的"贵"部分（算 K+1 个 logits）是 1 次 forward；"便宜"部分（逐个比对）是轻量操作。**没有任何地方需要 K 次大模型 forward。**

---

### Q3：为什么不直接用大模型并行得到 K+1 个输出，省掉验证？

**因为大模型做不到。** 这是"生成 vs 验证"不对称的最直接后果。

要让大模型一次并行产出 K+1 个 token，需要把 K+1 个**输入**同时喂进去。但生成的输入就是它自己的输出：

```
要产出 next1，输入 = last_real         ← 已知 ✓
要产出 next2，输入 = next1             ← 未知！要等 next1 算出来
要产出 next3，输入 = next2             ← 未知！要等 next2 算出来
...
```

**next2 的输入是 next1，而 next1 此刻还不存在。** 没法把一个还没算出来的 token 当输入塞进去并行算。所以大模型生成 K+1 个 token，**物理上只能是 K+1 次串行 forward**，无并行余地。

对比验证：验证的输入是 draft 猜的 `d1..dK`，**draft 已经先把它们算出来了**（用便宜的小模型串行算的），所以验证时 K+1 个输入全部已知，可以并行。

**代价账（核心）：**

| 方案 | 产出 K+1 token 的代价 |
|---|---|
| 纯大模型生成 | **K+1 次大模型 forward**（串行，每次都贵） |
| 投机解码 | **K 次小模型 forward（便宜）+ 1 次大模型 verify（贵）** |

投机解码的全部收益，就是用"K 次便宜 + 1 次贵"替换了"K+1 次贵"。当 draft 远比 target 便宜（尤其 N-gram 零成本），这一换就赚。

**所以验证这步不是"多余的累赘"，而是把大模型从"被迫串行生成"中解救出来的关键：**
- 没有 draft 提供输入 → 大模型只能串行生成 K+1 次
- 有 draft 提供输入 → 大模型能并行验证 1 次

draft 干的是"给大模型喂数据，把它从生成模式转成 prefill 模式"这件事。**prefill 模式能用满算力，生成模式用不满**——投机解码就是把昂贵的串行生成，变成廉价的串行 draft + 昂贵但并行的 verify。

---

### Q4：那为什么草稿模型能一下得到 K+1 个输出？

**关键澄清（藏着一个常见误解）：草稿模型并不能"一下"得到 K+1 个输出。它跟大模型一样，是老老实实串行自回归生成 K 个 token——每次 1 个，共 K 次 forward。草稿模型的"快"不是来自并行，而是来自便宜（每次 forward 代价低）。**

三种方式的本质对比：

|  | 输入是否已知？ | 怎么得到 K 个 token？ | 并行性 |
|---|---|---|---|
| **大模型生成** | ❌ 输入=自己的输出 | K 次大模型 forward，串行 | 无（死锁） |
| **草稿模型生成** | ❌ 输入=自己的输出 | K 次小模型 forward，串行 | 无（同样死锁） |
| **大模型验证** | ✅ draft 已提供输入 | 1 次大模型 forward（prefill） | 并行 |

**草稿模型和大模型生成，在"串行"这件事上一模一样**——都自回归，都受死锁约束。区别只在**单次 forward 的代价**：草稿小（参数少、可能量化成 4bit）→ 单次又快又省；大模型大 → 单次又慢又贵。

**那串行 K 次小模型，凭啥比 1 次大模型 verify 还便宜？**（命门）

```
投机解码一步 = K × (小模型单次) + 1 × (大模型单次)
普通解码一步 = 1 × (大模型单次)        ← 只产 1 token
```

要让投机解码划算，`K × 小模型单次` 必须足够小，不能把大模型省下来的时间吃回去。能做到的原因：
1. **小模型访存量小**：decode 是 memory-bound，主要时间在读权重。草稿参数是 target 的 1/10~1/50（如 0.5B draft 验 70B target），读的权重少几十倍 → 单次快几十倍。
2. **可以量化**：草稿常压成 4bit/int8（本讲第二节专门提了"draft model 可用 target 的量化版"），访存再砍一半以上。
3. **N-gram / Suffix 完全不跑模型**：直接查表/查树，`K 次草稿 = K 次内存查表`，纳秒级，几乎为零。

举例 K=4、草稿是大模型的 1/20：

```
草稿开销 ≈ 4 × (1/20) = 0.2 倍大模型单次
总开销 ≈ 0.2 + 1 = 1.2 倍大模型单次   ← 一步花销
这一步接受 3 个 + bonus，产出 4 token
普通解码产 4 token 要 4 倍大模型单次

加速比 ≈ 4 / 1.2 ≈ 3.3 倍
```

**赚的就是"草稿便宜"这个差价。草稿不是靠并行赢，是靠"单价低"赢。** 这也是本讲反复强调"用便宜的 draft"（N-gram 零成本、量化小模型）的原因——`t/T`（草稿单次 / 大模型单次）越小，加速比越高。

**有没有"真并行"出 K 个的草稿？有，但机制不同：**

- **Medusa**：训练 K 个额外 head，每个 head 直接预测"往后数第 i 个 token"。一次 forward，K 个 head 并行各吐一个预测。但 head_2 预测 t+2 时**用的是 t 时刻的 h，不知道 t+1 是什么**——是"跳着猜"，靠训练学到的联合分布，不是因果推导。所以 Medusa 接受率比 draft model 低（猜得没 draft model 准），但胜在一次出 K 个、零串行开销。这是用"准确率"换"并行度"。
- **EAGLE**：草稿是一个吃 target hidden state 的小 head，通常也是**串行 K 次**（每次吃上一步输出特征推下一步）。部分实现做"树形推测"（一次 forward 算一棵候选树，大模型用 tree attention 一次性验证整棵树），那才有并行成分——但那是**验证端的并行**，不是"草稿一次出 K 个"。

**所以"草稿一次并行出 K 个"严格成立的，主要就是 Medusa 这类多 head 方案，且代价是准确率下降。主流 draft model 和 EAGLE，草稿阶段仍是串行 K 次。**

---

### 一句话总收口

> 大模型生成 K+1 个 token 必须串行（下一个的输入是上一个的输出，死锁）；但只要有人把 K+1 个输入提前算出来（便宜的小模型 draft），大模型就能像 prefill 一样**一次并行**算出这 K+1 个位置的预测来验证 draft 猜得对不对。验证不是"再生成 K 次"，而是"1 次 prefill + 一串廉价比对"。代价从「K+1 次贵 forward」变成「K 次便宜 draft + 1 次贵 verify」。**草稿模型不并行，它只是"便宜地串行" K 次把候选摆上桌；并行的红利全在大模型验证那一步——因为 draft 喂齐了输入，才把大模型从"被迫串行生成"切换成了"一次 prefill 并行验证"。**
