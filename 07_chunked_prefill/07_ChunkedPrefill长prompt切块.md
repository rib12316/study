# 特性 #7：Chunked Prefill —— 长 prompt 怎么切块，又怎么和 decode 混在一起算

> 学习阶段：AI Infra 基础储备 / 推理调度（vLLM 招牌特性第三弹）
> 对应源码：`vllm/v1/core/sched/scheduler.py:786-810`（分块决策）+ `vllm/config/scheduler.py:84`（默认开启）+ `vllm/v1/attention/backends/flash_attn.py:248`（变长 attention）
> 本讲定位：第六讲 Continuous Batching 多次点到 chunked prefill 但没展开。这一讲专门回答：**长 prompt（比如 32K token）一次塞不进 token_budget，怎么办？切多少块、按什么策略切、切完怎么和 decode 混在同一个 attention 里算？**
> 干中学原则：本讲你要**亲手实现一个支持 chunking 策略的 mini 调度器**，对比"切 vs 不切"和"不同 chunk size"对吞吐/延迟的影响。

---

## 一、为什么需要 Chunked Prefill？（背景，面试高频追问）

### 1.1 长 prompt 的两难

第六讲的 Continuous Batching 用 `token_budget` 统一调度。但现在来一个 32K token 的长 prompt 请求。两难：

- **不切**：这个请求要独占 32K token_budget，意味着这一 step **几乎不能服务别的请求**（budget 被吃光）。正在 decode 的请求要"等"这个长 prefill 算完——**decode 延迟飙升**。更糟：32K 一次性 prefill 单 step 计算量巨大，GPU 跑很久，违反了"单 step 要快"的承诺。
- **排队**：等当前所有 decode 请求都结束再 prefill 这个长请求？那长请求的 TTFT（Time To First Token，首 token 延迟）爆炸，用户体验差。

### 1.2 Chunked Prefill 的解法

把长 prompt **切成多个 chunk（块）**，每 step 只算一个 chunk，跨越多个 step 把整个 prompt 算完。比如 32K prompt、chunk=2048：
```
step1: 算 prompt[0:2048]      + 别的请求的 decode
step2: 算 prompt[2048:4096]   + 别的请求的 decode
...
step16: 算 prompt[30720:32768] + 别的请求的 decode
```

**每个 step 都混了"一小段长 prefill + 多个 decode"**，GPU 始终吃饱，decode 请求延迟不被长 prompt 拖累，长 prompt 也在稳步推进。这就是 chunked prefill。

> 💡 面试一句话答：**Chunked Prefill 把长 prompt 切成固定大小的 chunk，每 step 只 prefill 一个 chunk 并和别的请求的 decode 混合调度，避免了长 prompt 独占 step 导致的 decode 延迟飙升和 TTFT 过长，同时保持 GPU 利用率。**

### 1.3 和 Continuous Batching 的关系

这是关键认知：**Chunked Prefill 不是独立特性，而是 Continuous Batching 的一个特例**。回到第六讲的统一抽象——"每个请求只有 num_computed vs num_tokens 的差"。一个被 chunked 的请求，它在多个 step 里都是"还有 token 没算"，每 step 算一部分。调度器根本不区分"这是 chunked prefill"还是"这是普通 prefill"，它只管"给这个请求分多少 token"。

第六讲的实践 2 其实已经支持 chunking（budget 不够就放回 waiting 头部继续）。第七讲专门深挖：**切的策略、切和 decode 的混合比例、为什么 v1 默认开启**。

---

## 二、核心：两种触发 chunking 的机制（scheduler.py:790-810）

看源码核心：

```python
# 795行：剩余待算 token 数
num_new_tokens = request.num_tokens - num_computed_tokens
# 796-798行：long_prefill_token_threshold —— 单次 prefill 上限
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold          # 主动切：即使 budget 够，也只算 threshold 个

# 802-808行：chunked_prefill 开关
if (not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > token_budget):
    break                                # 禁用 chunking → 一次算不完就不算，等下次
# 810行：被动切 —— 受 budget 约束
num_new_tokens = min(num_new_tokens, token_budget)
```

**两种切法的本质区别**：

| 触发 | 条件 | 目的 |
|------|------|------|
| **被动切（budget 约束）** | `num_new_tokens > token_budget` | 防止一个请求吃光整个 step 的预算 |
| **主动切（threshold）** | `num_new_tokens > long_prefill_token_threshold` | 防止单 step 计算量过大，控制 decode 延迟 |

`long_prefill_token_threshold` 默认 `max_model_len * 0.04`（config/scheduler.py:259）。比如 max_model_len=32768，threshold≈1310。意思是：**即使 budget 够，单个请求一次最多 prefill 1310 token**——超过就切，留给别的请求和 decode。

> 💡 这个 threshold 是"主动让出"的优雅度。没有它，一个富 request 可能一口气吃满 budget；有了它，长 request 被强制"细水长流"，保证短请求和 decode 不被饿死。

---

## 三、关键反直觉：prefill 和 decode 为什么能混在同一个 attention 里？

这是面试最容易被追问的难点，也是 chunked prefill 的工程核心。

### 3.1 朴素想法的陷阱

你可能想：prefill 是 `[1, prompt_len, hidden]`，decode 是 `[1, 1, hidden]`。它们形状不同，怎么 batch？朴素做法是 padding 到一样长——但 prefill 几千 token、decode 1 token，padding 浪费巨大。

### 3.2 真相：FlashAttention 的 varlen 接口

vLLM 用 FlashAttention 的 **varlen（variable-length）接口**。核心数据结构（`flash_attn.py:248`）：

```python
query_start_loc: torch.Tensor   # 每个请求 query 的【累积】起始位置，如 [0, 2048, 2049, 2050]
seq_lens: torch.Tensor          # 每个请求的 KV 总长度，如 [2048, 100, 50]
```

举例一个混合 step：
- 请求 A 在 prefill chunk，query = 2048 token（prompt 的第 1 段）
- 请求 B 在 decode，query = 1 token，但它有 100 个历史 KV
- 请求 C 在 decode，query = 1 token，50 个历史 KV

`query_start_loc = [0, 2048, 2049, 2050]`（累积），`seq_lens = [2048, 100, 50]`。FlashAttention 用 `cu_seqlens`（累积序列长度）格式，**在一个 batch 里精确地为每个请求切出它自己的 query-KV 对，零 padding**。

```
batch query:  [A的2048个 | B的1个 | C的1个]   ← 拼成一条，不 padding
cu_seqlens_q: [0, 2048, 2049, 2050]
cu_seqlens_k: [0, 2048, 2148, 2198]   ← KV 按 seq_lens 拼接
```

每个请求在自己的区间内做 causal attention，请求之间互不干扰。**这就是 chunked prefill 能和 decode 混合的底层秘密**——FlashAttention 的 varlen 天然支持变长混合。

> 💡 旧版 vLLM（v0）曾用"prefill 和 decode 分开两个 kernel 调用"的方案（padding 浪费），v1 统一成一次 varlen 调用，效率更高。这是 v1 架构的重要改进之一。

### 3.3 causal mask 的处理

prefill 的 query 内部要 causal mask（第 i 个 query 只看前 i 个 key），decode 的 query 看所有历史 key。varlen 接口里，每个请求区间的 causal 关系由 `cu_seqlens` 隐式定义，FlashAttention 内部自动处理。你不用手动构造 mask——这是它高效的原因。

---

## 四、chunk 大小的权衡（面试调参题）

`long_prefill_token_threshold`（chunk 上限）是个关键调参旋钮：

| chunk 大小 | 优点 | 缺点 |
|-----------|------|------|
| **大**（如 8192） | 长 prompt 很快算完，TTFT 低 | 单 step 久，decode 延迟高，短请求被挤压 |
| **小**（如 512） | decode 延迟稳定，公平 | 长 prompt 要切很多次，跨 step 开销大，总吞吐略降 |
| **默认**（≈4% max_len） | 平衡点 | 经验值，不同负载要调 |

实际中要根据**负载特征**调：如果用户请求多是短 prompt + 长生成，chunk 调大（prefill 快进 decode）；如果是长 prompt（RAG/文档问答）多，chunk 调小（防 TTFT 饿死别人）。

---

## 五、把第七讲和前六讲连起来

| 讲次 | 这讲的位置 |
|------|-----------|
| 第5讲 PagedAttention | chunked prefill 跨 step 时，每 chunk 的 KV 写入不同 block（PagedAttention 让跨 chunk 的 KV 连续存放于 block_table） |
| 第6讲 Continuous Batching | chunked prefill 是 continuous batching 的特例——被 chunk 的请求就是"多 step 才算完 prefill"的请求 |
| **第7讲 Chunked Prefill** | **专门讲切的策略 + varlen 混合 attention** |

**三讲构成完整的 vLLM 调度栈**：PagedAttention（显存）→ Continuous Batching（统一调度）→ Chunked Prefill（长 prompt 处理）。面试被问"vLLM 怎么处理长 prompt"，你的答案就是这三讲的串联。

---

## 六、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：分块决策（基础）
1. 读 `scheduler.py:795-810`。`num_new_tokens = request.num_tokens - num_computed_tokens` 算的是什么？如果 `num_computed_tokens=1000`、`num_tokens=5000`，`num_new_tokens` 是多少？
2. `scheduler.py:796-798` 的 `long_prefill_token_threshold`。它在什么条件下把 `num_new_tokens` 截断？这是"主动切"还是"被动切"？目的是什么？
3. `scheduler.py:802-808`：如果 `enable_chunked_prefill=False` 且 `num_new_tokens > token_budget`，执行 `break`。这个 break 意味着什么？和 `enable_chunked_prefill=True` 时的行为有何不同？

### 任务 B：状态与混合（核心）
4. 读 `scheduler.py:1146`。`is_prefill_chunk` 是怎么判断的？一个 prompt_len=5000、已算 2000 的请求，`is_prefill_chunk` 是 True 还是 False？
5. 读 `flash_attn.py:248-250`。`query_start_loc` 和 `seq_lens` 分别记录什么？为什么 prefill（query=2048）和 decode（query=1）能放在同一个 batch？它们靠什么"分隔"？
6. 思考题：朴素 batching 要把 prefill 和 decode padding 到一样长才能 batch。FlashAttention varlen 是怎么避免 padding 浪费的？（提示：`cu_seqlens`）

### 任务 C：默认配置与权衡（机制）
7. 读 `config/scheduler.py:84`。`enable_chunked_prefill` 默认是 True 还是 False？为什么 v1 选择默认开启？（提示：考虑长 prompt 在线上服务的频率）
8. `config/scheduler.py:259`：`long_prefill_token_threshold = int(max_model_len * 0.04)`。如果 max_model_len=32768，threshold 是多少？这个 4% 是怎么来的——为什么不是 1% 或 10%？
9. 思考题：如果你的服务 90% 是短 prompt（<512）+ 长生成，你会把 `long_prefill_token_threshold` 调大还是调小？为什么？反过来（长 prompt 多）呢？

---

## 七、干中学实践任务（核心！）

> 在 `practice_chunked_prefill.py` 里实现一个支持 chunking 策略的 mini 调度器，并对比不同策略。
> 依赖：仅标准库。
> 设计哲学：你不只实现 chunking，还要**量化对比**不同 chunk 策略对延迟/吞吐的影响——这是 Infra 调参的核心技能。

### 实践 1：支持 threshold 的 chunking 调度（热身）
基于第六讲的 Scheduler，扩展：
- 增加 `long_prefill_token_threshold` 参数
- `step()` 里对 prefill 请求，`num_new_tokens = min(剩余, threshold, budget)` —— 主动 + 被动都切
- 记录每个请求的 prefill 跨了多少 step（`num_prefill_steps`）

验证：一个 prompt_len=5000 的请求，threshold=1000、budget=2000，应该跨 5 个 step prefill（每 step 1000），不是 3 个（每 step 2000）。

### 实践 2：对比 chunking 策略（核心）
实现一个**模拟器**，跑同一个负载（几个长 prompt + 几个短 decode 请求），分别用：
- 策略A：`enable_chunked_prefill=False`（不切，长 prompt 独占）
- 策略B：`long_prefill_token_threshold=很大`（只被动切，budget 不够才切）
- 策略C：`long_prefill_token_threshold=适中`（主动切）

记录并对比三个指标：
- **decode 请求的平均 step 延迟**（被长 prefill 拖累的程度）
- **长 prompt 的 TTFT**（首 token 时间 = prefill 完的 step 数）
- **总 step 数**（吞吐代理）

验证：策略C 应该在 decode 延迟和 TTFT 之间取得最佳平衡。

> ⚠️ **一个诚实的实验观察**（我在准备参考实现时发现的）：在这个**简化模型**里，策略 B（只被动切）和 C（主动切）的指标可能**非常接近**，甚至相同。原因是我们把每个 step 都当成"单位时间"，没有建模"step 的实际计算时间与该 step 的 token 数成正比"。真实场景下，C 更优——因为主动切让每个 step 的 prefill token 更少 → step 更快 → decode 等待更短。**策略 A（不切）则会明显劣于 B/C**（长 prompt 独占 budget，短 decode 请求被饿死，甚至死循环），这一点简化模型就能清楚体现，是 chunked prefill 必要性的活教材。做这个实践时，重点观察 A 与 B/C 的巨大鸿沟；B/C 的细微差异作为思考题——想想怎么改 simulate 让差异显现（提示：让 step 耗时正比于 token 数）。

### 实践 3：varlen 模拟（进阶）
实现一个简化版 `query_start_loc` 构造：
- 给定一个 step 里各请求的 query 长度列表（如 `[2048, 1, 1]`），算出 `query_start_loc`（`[0, 2048, 2049, 2050]`）
- 给定各请求的 KV 长度（如 `[2048, 100, 50]`），算出 `cu_seqlens_k`（`[0, 2048, 2148, 2198]`）
- 验证：用这俩能正确切分一个拼接的 batch

> 💡 实践 2 是灵魂。要点：① 模拟器要能记录每 step 各请求分到多少 token ② decode 延迟用"被 prefill 挤占的程度"近似（一个 step 里 prefill token 越多，decode 相对越慢）③ TTFT 用 prefill 完成的 step 序号。如果实践 2 太难，确保实践 1 完美，实践 2 可简化为只对比两个策略。

---

## 八、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_chunked_prefill.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_chunked_prefill.py 运行结果，重点贴策略对比表）

---

## 九、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实践 2 的策略对比，哪个策略在 decode 延迟和 TTFT 上最优？你的直觉和实验结果一致吗？② 实现 varlen 的 cu_seqlens 后，你对"为什么 padding 浪费"的理解？③ 4% 这个默认 threshold，你服务的话会怎么调？

（完成实践后填写）

---

## 十、个人复盘感悟（留给你写）

> 你是 AI Infra 方向求职者，建议角度：① Chunked Prefill 本质是"时间换空间/公平"——把一个大任务切片让出资源，这种"细水长流"的调度哲学你在别的系统见过吗？② FlashAttention varlen 让 prefill/decode 零 padding 混合，这种"用数据布局（cu_seqlens）消除 padding"的思路，对你优化别的 kernel 有启发吗？③ 长 prompt 服务（RAG/agent）越来越常见，chunked prefill 对这些场景的价值你怎么评估？④ v1 默认开启 chunked prefill，而 v0 要手动开——这种"特性默认化"反映的产品取舍？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**vLLM 调度三件套（PagedAttention + Continuous Batching + Chunked Prefill）全部讲完**。完成后告诉我下一步：
> - **① Prefix Caching 深入**（hash 链/eviction/multimodal，第5讲点到）
> - **② Speculative Decoding**（投机解码，v1 热点，vllm/v1/spec_decode）
> - **③ 换领域**：MoE 专家路由 / LoRA / 结构化输出 / KV offload
> - 或你指定的
