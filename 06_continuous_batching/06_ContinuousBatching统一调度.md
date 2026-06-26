# 特性 #6：Continuous Batching —— 没有 prefill/decode 阶段之分的统一调度

> 学习阶段：AI Infra 基础储备 / 推理调度（vLLM 招牌特性第二弹）
> 对应源码：`vllm/v1/core/sched/scheduler.py`（核心 `schedule()`，388行）
> 本讲定位：上一讲 PagedAttention 解决了"显存利用"，这一讲解决"算力利用"。**两者是 vLLM 调度全貌的两条腿**。Continuous Batching 是面试必问的吞吐优化第二名（仅次于 PagedAttention）。
> 干中学原则：本讲你要**亲手实现一个 mini continuous batching 调度器**——waiting/running 两队列、token_budget 统一分配、每个 step 混合调度 decode+prefill、显存不足时抢占。这是 v1 `Scheduler.schedule()` 的精简复刻。

---

## 一、为什么需要 Continuous Batching？（背景，面试必答）

### 1.1 朴素 batching 的两个致命问题

最朴素的 LLM 服务用**静态 batching**：攒一批请求 → 整批 prefill → 整批 decode → 等所有请求都生成完才结束整批 → 接下一批。问题：

1. **队头阻塞（head-of-line blocking）**：batch 里 8 个请求，7 个生成了 10 token 就结束（短回答），1 个要生成 2000 token（长回答）。那 7 个结束后**干等**剩下 1 个跑完，GPU 大部分时间只服务 1 个请求，吞吐暴跌。
2. **prefill 和 decode 的算力鸿沟**：prefill（处理整个 prompt）是 compute-bound，decode（每次 1 token）是 memory-bound。如果把它们强行分开成"prefill 阶段"和"decode 阶段"，GPU 在 decode 阶段严重吃不饱（因为 decode 算术强度低）。

### 1.2 Continuous Batching 的解法

也叫 **iteration-level scheduling / in-flight batching**（NVIDIA 的叫法）。核心思想两条：

1. **请求粒度的进出**：每个请求**独立**生成完就退出 batch，不用等别的。腾出的位置立即让 waiting 里的新请求进来。
2. **消除阶段概念**：**每个 step（一次 GPU forward）同时混合调度 prefill 和 decode**——既有在生成的请求的 decode（各 1 token），又有新请求的 prefill（一批 prompt token）。靠 token_budget 这个总量来约束。

> 💡 面试一句话答：**Continuous Batching 在每个 GPU forward step 用一个 token_budget 同时调度运行中请求的 decode 和等待请求的 prefill，请求完成即退出、有空位即接纳新请求，消除了静态 batching 的队头阻塞和 prefill/decode 阶段切换的算力浪费。**

### 1.3 和 PagedAttention 的关系

这是关键认知，面试常被追问：

- **PagedAttention 是 Continuous Batching 的使能者**。为什么？因为 continuous batching 要求"请求随时进出 batch"，而每次进出都要重新分配 KV cache。如果用连续分配（朴素），每次进出都要搬移/重排显存，开销巨大且碎片化。**只有分页（PagedAttention）让 KV cache 可以零散分配/释放，continuous batching 才能"请求进出几乎零成本"**。
- 反过来，continuous batching 让分页的好处真正发挥——动态调度让 block 的分配/释放/共享频繁发生，分页的高利用率才有用武之地。

**一句话：PagedAttention 给了"灵活的显存"，Continuous Batching 用它做了"灵活的调度"。两者缺一不可。**

---

## 二、核心抽象：没有阶段，只有 token_budget

这是 vLLM v1 调度器最优雅的设计。看源码注释（`scheduler.py:390`）：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec (= len(prompt) + len(output) + len(spec)).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

**翻译成人话**：调度器眼里没有"prefill 请求"和"decode 请求"之分，**每个请求只有一个状态：还有多少 token 没算**（`num_tokens_with_spec - num_computed_tokens`）。调度器每 step 从总量 `token_budget` 里分 token 给各请求，让它们的 computed 赶上目标。

- 一个新请求（刚到），有 100 个 prompt token 没算 → 分它一批（比如 100，或 chunked 的话分 64）→ 这就是"prefill"
- 一个正在 decode 的请求，还有 1 个 token 没算（下一个输出）→ 分它 1 → 这就是"decode"
- **两者在同一个 step 里、同一个 token_budget 下，被一视同仁地调度**

这种统一抽象威力巨大：chunked prefill（把长 prompt 分多次算）、speculative decoding（一次算多个 draft token）、prefix caching（部分 token 已命中不用算）……全是这个"让 computed 赶上目标"框架的自然特例。

---

## 三、三个队列与主调度循环

`Scheduler`（scheduler.py:68）维护三个核心队列：

```python
self.waiting: RequestQueue    # 181行：新请求 + 被抢占的请求
self.running: list[Request]   # 184行：正在被服务的请求
self._inflight_prefills: set  # 328行：还没 prefill 完的请求（chunked）
```

### 3.1 主循环 `schedule()`（388行）的结构

```python
def schedule(self, throttle_prefills=False) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens   # 408行：本 step 总 token 预算
    
    # === 第一阶段：先服务 RUNNING 请求 ===
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:   # 432行
        request = self.running[req_index]
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        # ... 分配 token 给这个请求（decode 是 1，inflight prefill 是 chunk）
        # ... 显存不足时抢占（见下）
        token_budget -= num_scheduled
        scheduled_running_reqs.append(request)
    
    # === 第二阶段：从 WAITING 接纳新请求（如果还有 budget 和显存）===
    if not preempted_reqs and ...:   # 626行：没发生抢占才接纳
        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs: break  # 630行：达上限
            request = self._select_waiting_queue_for_scheduling().peek_request()
            # ... prefill（可能 chunked）
            self.running.append(request)   # 939行：进入 running
            token_budget -= num_new_tokens
    
    return SchedulerOutput(scheduled_new_reqs, scheduled_running_reqs, ...)
```

### 3.2 为什么"先 running 后 waiting"？

注意顺序：**每个 step 优先保证 running 的 decode**（它们各只要 1 token，便宜），剩余 budget 再接纳 waiting 的 prefill（贵）。这保证了已经在生成的请求**延迟最低**（每 step 必被服务），新请求用"边角"算力 prefill。这就是 continuous batching 高吞吐低延迟的根源。

---

## 四、抢占：显存不足时的优雅降级（scheduler.py:538）

当 running 队列里某个请求需要的 KV cache 超过空闲块时，调度器不会 OOM，而是**抢占**：

```python
# 538行：从 running 里挑一个"最该被抢占"的（通常是最大/最不紧急的）
preempted_req = max(self.running, key=lambda r: <priority>)
self.running.remove(preempted_req)
self._preempt_request(preempted_req, scheduled_timestamp)  # 1106行
preempted_reqs.append(preempted_req)
```

`_preempt_request`（1106）做的事：
```python
def _preempt_request(self, request, timestamp):
    assert request in self.running, "Only running requests can be preempted"
    request.num_preemptions += 1           # 记录被抢占次数
    self.kv_cache_manager.free(request)     # 释放它的 KV block（回 PagedAttention 池！）
    self.waiting.prepend_request(request)   # 放回 waiting 队列【头部】
```

**关键细节**：被抢占的请求放回 waiting **头部**（`prepend`，不是 `append`），这样下一步它优先重新 prefill。它的 KV cache 被释放（还给 PagedAttention 的 block pool，可能被别的请求复用），重新 prefill 时再重新分配——这就是 PagedAttention 的灵活性在调度层面的体现。

> 💡 这和上一讲实践 3 的 `preempt` 完全呼应！上一讲你在 KVCacheManager 层实现了抢占（释放 block_table），这一讲在调度器层看到它被调用。**第五讲和第六讲在这里闭环了**。

---

## 五、token_budget 的两个约束

调度受两个独立 budget 约束（scheduler.py:408 + max_num_running_reqs）：

| Budget | 含义 | 作用 |
|--------|------|------|
| `max_num_scheduled_tokens` | 每 step 最多算多少 token | 控制单 step 计算量（防止单 step 太久，影响延迟） |
| `max_num_seqs`（→`max_num_running_reqs`） | running 队列最多多少请求 | 控制 batch size（防止 attention 的 O(N²) 爆炸） |

这两个旋钮是调 vLLM 性能的核心：
- `max_num_scheduled_tokens` 太小 → 每 step 干的活少，吞吐低；太大 → 单 step 久，P99 延迟高。
- `max_num_seqs` 太小 → 并发低；太大 → 显存吃紧，抢占频繁。

实际调参就是在吞吐和延迟之间找平衡。

---

## 六、把第六讲和前五讲连起来

| 讲次 | 解决的问题 | 这一讲的位置 |
|------|-----------|-------------|
| 第1~4讲（量化） | 权重显存 | 调度器不关心，量化是"模型加载时"的事 |
| 第5讲（PagedAttention） | KV cache 显存碎片 | 调度器通过 `kv_cache_manager` 调用它，抢占时 free block |
| **第6讲（Continuous Batching）** | **算力利用** | **统一调度 prefill+decode，用 PagedAttention 提供的灵活显存** |

**三件套合起来才是完整 vLLM**：量化省权重 → PagedAttention 省 KV → Continuous Batching 用好省下来的资源。面试被问"vLLM 为什么快"，你能从这三个维度系统作答，是 Infra 候选人的完整答案。

---

## 七、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：统一抽象（基础）
1. 读 `scheduler.py:390-399` 的注释。为什么调度器"没有 prefill/decode 阶段"？它用什么统一概念替代了"阶段"？
2. 一个正在 decode 的请求和一个刚到的请求，在调度器眼里唯一的区别是什么？（提示：`num_tokens_with_spec - num_computed_tokens`）
3. `max_num_scheduled_tokens` 和 `max_num_running_reqs` 分别约束什么？如果只调大后者不调前者，会发生什么？

### 任务 B：主循环（核心）
4. 读 `scheduler.py:432` 的主循环。为什么先服务 running 再服务 waiting（626行）？如果反过来（先 prefill 新请求再 decode 旧请求），会对延迟产生什么影响？
5. `scheduler.py:630` 的 `if len(self.running) == self.max_num_running_reqs: break`。这条 break 意味着什么？它保护了什么？
6. 一个请求从 waiting 进入 running 是在哪一行（提示：`self.running.append(request)`）？此时它一定 prefill 完了吗？（结合 chunked prefill 思考）

### 任务 C：抢占（机制）
7. 读 `scheduler.py:538-570` 的抢占逻辑。被抢占的请求用 `max(self.running, key=...)` 选择——这个选择策略的目标是什么？为什么不随机选？
8. `_preempt_request`（1106）把请求放回 `self.waiting.prepend_request`（头部）而不是 `append`（尾部）。为什么？这对被抢占请求的恢复延迟有什么影响？
9. 思考题：抢占会释放请求的 KV cache（调 `kv_cache_manager.free`，回到第五讲的 BlockPool）。这意味着重新 prefill 时要重算。如果用 PagedAttention 的 prefix caching，重新 prefill 的开销能减少吗？怎么减少？

---

## 八、干中学实践任务（核心！）

> 在 `practice_continuous_batching.py` 里实现一个 mini continuous batching 调度器。
> 依赖：仅标准库。不需要装 vllm/torch。
> 设计哲学：你不读 vLLM 的 Scheduler，而是**重建**它。能正确混合调度、抢占、让请求进出，才算真懂 continuous batching。

### 实践 1：请求模型 + 两队列（热身）
实现：
- `Request`：有 `req_id`、`prompt_len`（要 prefill 的 token 数）、`output_tokens`（已生成）、`max_output`（最多生成多少）、`num_computed`（已算到第几个 token）
- `Scheduler`：维护 `waiting: list[Request]` 和 `running: list[Request]`，`max_num_seqs`（running 上限）、`token_budget`（每 step 总预算）
- `add_request(req)`：加入 waiting
- `num_uncomputed(req)`：返回 `req` 还有多少 token 没算（prompt 部分 + 还要生成的部分 - num_computed）

### 实践 2：统一调度循环（核心）
实现 `Scheduler.step()`：模拟一次 GPU forward，返回本 step 调度了哪些请求、各分到多少 token。算法：
```
budget = token_budget
scheduled = {}
# 阶段1：先服务 running（每个 decode 请求分 1 token）
for req in running:
    if budget <= 0: break
    # decode: 算 1 个新 token
    req.num_computed += 1; req.output_tokens += 1
    budget -= 1
    scheduled[req.req_id] = 1
# 阶段2：用剩余 budget 接纳 waiting（prefill）
while waiting and budget > 0 and len(running) < max_num_seqs:
    req = waiting.pop(0)
    need = req.prompt_len - req.num_computed   # prefill 需要的 token 数
    alloc = min(need, budget)                   # 可能 chunked
    req.num_computed += alloc
    budget -= alloc
    scheduled[req.req_id] = alloc
    if req.num_computed < req.prompt_len:
        # chunked prefill 没完，放回 waiting 头部（下一 step 继续）
        waiting.insert(0, req)
    else:
        # prefill 完，进 running 开始 decode
        running.append(req)
# 移除生成完的请求
running = [r for r in running if r.output_tokens < r.max_output]
return scheduled
```

验证：构造 3 个请求（不同 prompt_len 和 max_output），跑若干 step，检查：
- 每个 step 的 scheduled 反映了"running decode 1 token + waiting prefill"
- 请求生成完后从 running 消失
- 短请求先结束（不等长请求）

### 实践 3：抢占 + KV 约束（进阶）
扩展 `Scheduler`，加入 `max_blocks`（总 KV 块数）和每请求占用块数（用第五讲的 cdiv 算）：
- 每 step 检查 running+新 prefill 的总块需求，超过 max_blocks 时抢占 running 里"输出最多"的请求（模拟 vLLM 的优先级）
- 被抢占的请求：num_computed 归零（KV 被释放，要重新 prefill）、放回 waiting **头部**

验证：构造一个会触发抢占的场景（max_blocks 很小），确认：
- 抢占发生时被抢占请求回到 waiting 头部、num_computed 归零
- 下一步它优先重新 prefill

> 💡 实践 2 是灵魂。要点：① 先 running 后 waiting ② decode 只分 1 token ③ prefill 可能被 chunked（budget 不够时）④ 生成完即移除。实践 3 把第五讲和第六讲串起来——抢占的本质是"KV 不够时牺牲某个请求"。如果实践 3 太难，先确保实践 2 完美，实践 3 可选。

---

## 九、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_continuous_batching.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_continuous_batching.py 运行结果）

---

## 十、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现"没有阶段、只有 token_budget"的统一调度后，你对"prefill 和 decode 其实是同一件事"的理解？② 先 running 后 waiting 的顺序，对延迟的意义你体会到了吗？③ 抢占时 num_computed 归零（重算）是不是很"浪费"？vLLM 为什么愿意付出这个代价？

（完成实践后填写）

---

## 十一、个人复盘感悟（留给你写）

> 你是 AI Infra 方向求职者，建议角度：① Continuous Batching 借鉴了 OS 的进程调度（时间片、抢占），这种跨领域迁移你怎么看？② "消除阶段概念、用统一抽象（token_budget）"这种设计哲学，你在别的系统里见过吗（比如数据库的统一查询计划）？③ vLLM 愿意为不 OOM 而付出"抢占重算"的代价，这种"宁可慢不要崩"的取舍，对线上 LLM 服务意味着什么？④ 吞吐 vs 延迟的 trade-off（max_num_scheduled_tokens / max_num_seqs），你服务过的话会怎么调？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**PagedAttention + Continuous Batching 这对组合讲完了，vLLM 调度全貌已成**。完成后告诉我下一步：
> - **① Chunked Prefill**（长 prompt 切块的细节，上一讲已点到，可单独深挖它和 continuous batching 的配合）
> - **② Prefix Caching 深入**（hash 链/eviction/multimodal，第五讲点到）
> - **③ Speculative Decoding**（投机解码，vllm/v1/spec_decode，v1 的热点特性）
> - **④ 换回量化或其他领域**
> - 或你指定的
