# 特性 #9：Prefix Caching 深入 —— 链式 Hash + LRU 驱逐 + ref_cnt 共享

> 学习阶段：AI Infra 基础储备 / 推理调度（PagedAttention 的"超能力"展开）
> 对应源码：`vllm/v1/core/kv_cache_utils.py:577`（链式 hash）+ `vllm/v1/core/block_pool.py:199/226/574/597`（查询/缓存/驱逐/touch）
> 本讲定位：第5讲 PagedAttention 点到了 prefix caching（"block 可被多请求共享"），但没展开它是**怎么识别可共享的 block、怎么保护它不被误驱逐、怎么在显存满时优雅淘汰**的。这一讲完整回答这三个问题。理解它，你就理解了 RAG/agent/多租户 LLM 服务为什么能"第二个相同 prompt 秒回"。
> 干中学原则：本讲你要**亲手实现一个 mini prefix cache**——链式 hash 识别、LRU 驱逐队列、ref_cnt 引用计数共享。这是 vLLM BlockPool 的完整精简复刻，也是第5讲的深化（第5讲只有空闲队列，没有 hash 缓存）。

---

## 一、为什么 Prefix Caching 是 PagedAttention 的"超能力"？（背景）

### 1.1 RAG/Agent 的重复 prefix 痛点

现代 LLM 服务里，请求之间有大量**相同前缀**：
- **RAG**：每个请求都带同一份 system prompt + 检索到的文档前缀
- **Agent**：多轮对话里前 N 轮完全相同，只追加最新一轮
- **Few-shot**：所有请求共用同一组示例
- **多租户**：不同用户的 prompt 复用平台默认 prefix

朴素（连续 KV 分配）下，即使两个请求 prefix 完全相同，**它们的 KV cache 也各存一份**——因为两个请求的连续空间无法重叠。这是巨大的浪费。

### 1.2 分页让共享成为可能

PagedAttention 把 KV cache 切成固定 block。**如果两个请求的某个 block 内容相同（token 序列相同），它们就可以共享同一个物理 block**——只需让各自的 block_table 指向同一个物理 block ID。

但问题来了：
1. **怎么知道两个 block 内容相同？** → 链式 hash
2. **共享的 block 怎么不被一个请求结束就释放掉？** → ref_cnt 引用计数
3. **显存满了，共享 block 谁先被淘汰？** → LRU 驱逐

这三个机制就是本讲的全部。它们让 prefix caching 在 vLLM 里几乎是"免费"的——命中率取决于负载，但机制本身开销极小。

> 💡 面试一句话答：**Prefix caching 用链式 hash（每个 block 的 hash = f(父block hash, 本block tokens)）识别相同前缀，命中时多个请求通过 ref_cnt 共享同一物理 block（ref_cnt>0 时绝不驱逐），显存不足时按 LRU 淘汰 ref_cnt=0 的 block——让 RAG/agent 等"重复前缀"场景的 TTFT 大幅下降。**

---

## 二、核心机制①：链式 Hash（kv_cache_utils.py:577）

这是整个 prefix caching 的识别基础。

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**关键设计：block_hash = hash(parent_block_hash, curr_tokens)**。每个 block 的 hash 依赖**前驱 block 的 hash**，形成链式结构（像区块链）。

为什么这么设计？考虑两个请求 A=`[t1,t2,t3,t4,t5,t6]` 和 B=`[t1,t2,t3,t4,x,y]`（前 4 个 token 相同）。block_size=2：
```
A: block0=[t1,t2] block1=[t3,t4] block2=[t5,t6]
B: block0=[t1,t2] block1=[t3,t4] block2=[x,y]
```
- A.block0.hash = hash(None, [t1,t2])
- A.block1.hash = hash(A.block0.hash, [t3,t4])
- B.block0.hash = hash(None, [t1,t2]) = **A.block0.hash**（相同！）
- B.block1.hash = hash(B.block0.hash, [t3,t4]) = hash(A.block0.hash, [t3,t4]) = **A.block1.hash**（相同！）
- B.block2.hash = hash(B.block1.hash, [x,y]) ≠ A.block2.hash（不同）

**链式 hash 的精妙**：因为 B.block1.hash 依赖 B.block0.hash（=A.block0.hash），所以"前 N 个 block hash 相同"自动等价于"前 N×block_size 个 token 完全相同"。**只需逐 block 比较一个 hash 值，就能找到最长公共前缀**，无需逐 token 比较。

> 💡 `extra_keys` 参数（kv_cache_utils.py:581）是为多模态/LoRA 设计的：同一 token 序列，如果附带的图片特征不同或 LoRA adapter 不同，hash 也要不同。这让 prefix cache 在多模态/多 LoRA 下正确区分。

---

## 三、核心机制②：查询与缓存（block_pool.py:199/226）

### 3.1 查询命中：get_cached_block（199行）

```python
def get_cached_block(self, block_hash, kv_cache_group_ids):
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)
        block = self.cached_block_hash_to_block.get_one_block(block_hash_with_group_id)
        if not block:
            return None              # 任一 group miss → 整体 miss
        cached_blocks.append(block)
    return cached_blocks
```

BlockPool 维护一个 `cached_block_hash_to_block: dict[hash, KVCacheBlock]`。新请求来了，把 prompt 切成 block，逐块算 hash，调 `get_cached_block` 查表。命中就复用，miss 就新算。

**逐块查询 + 截断**：从 block0 开始逐块查，第一个 miss 的 block 处停止——前面的命中 block 就是"可复用的公共前缀"。这就是 `KVCacheManager.get_computed_blocks`（第5讲引用）做的事。

### 3.2 缓存写入：cache_full_blocks（226行）

当一个 block 被完整填充（prefill 到了 block_size），就把它的 hash 写入缓存表（block_pool.py:302）：
```python
self._insert_block_hash(block_hash_with_group_id, blk, num_tokens=num_hash_tokens)
```

**注意时机**：只有 block **填满**才缓存。半满的 block（partial）不进缓存表——因为它的内容还不完整，hash 不稳定。这是 `cache_partial_block`（358行）单独处理的原因。

---

## 四、核心机制③：ref_cnt 共享与 LRU 驱逐（block_pool.py:597/574）

这是 prefix caching 最精巧的部分，也是和第5讲（纯空闲队列）的关键区别。

### 4.1 touch：共享时保护 block（597行）

```python
def touch(self, blocks):
    for block in blocks:
        # ref_cnt=0 意味着 block 在 free list（驱逐候选），先移除
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

当一个请求命中了一个已缓存的 block：
- 如果该 block 的 `ref_cnt==0`（当前空闲，在 free list 里，是驱逐候选）→ **立刻从 free list 移除**（保护它不被驱逐）
- `ref_cnt += 1`

多个请求共享同一 prefix block 时，ref_cnt 累加（2、3、4...）。**只要 ref_cnt > 0，这个 block 就不在 free list，绝不会被驱逐**。这就是"共享保护"。

### 4.2 free：引用归零才回到驱逐候选（第5讲的 free_blocks）

请求结束时，释放它的 block。每个 block `ref_cnt -= 1`。当 `ref_cnt` 降到 0，block 才被放回 free list 尾部（成为 LRU 驱逐候选）。如果还有别的请求在用（ref_cnt > 0），block 保留。

### 4.3 _maybe_evict_cached_block：显存满时淘汰（574行）

当需要分配新 block 但 free list 不够时，从 free list 头部取（LRU 最久未用）。取出的 block 如果有 hash（曾在缓存表里），调 `_maybe_evict_cached_block`：
```python
def _maybe_evict_cached_block(self, block):
    evicted_hashes = self._remove_cached_block_hashes(block)  # 从缓存表移除
    if not evicted_hashes:
        return False
    return True
```

驱逐 = 从 `cached_block_hash_to_block` 表里删除该 block 的 hash 记录 + 物理块重用。**注意只能驱逐 ref_cnt=0 的 block**（ref_cnt>0 的不在 free list，根本不会被取到）。

### 4.4 LRU 顺序的秘密

回忆第5讲：`FreeKVCacheBlockQueue` 的 `free_blocks` 把归还的 block 加到**尾部**，`alloc` 从**头部**取。所以：
- 刚释放的 block 在尾部（最近用过，LRU 最后淘汰）
- 最久没用的 block 在头部（最先淘汰）

`touch` 会把命中 block 从 free list 移除，等于"插队保护"。被驱逐时，最久未访问（free list 头部）的先走。这就是完整的 LRU 语义。

---

## 五、完整数据流：一个请求怎么吃到 prefix cache

```
1. 新请求到达，prompt = [t1, ..., t100]
2. 切成 block：block0=[t1..t16], block1=[t17..t32], ..., block5=[t81..t96], block6=[t97..t100](未满)
3. 算每个 block 的链式 hash（hash_block_tokens）
4. get_computed_blocks: 从 block0 开始逐块查 get_cached_block
   - block0 命中！block1 命中！block2 命中！block3 miss
   - 返回 [block0, block1, block2]，num_computed_tokens = 3*16 = 48
5. touch([block0, block1, block2]): ref_cnt 各 +1，从 free list 移除（如果之前 idle）
6. 从 block3 开始 allocate 新 block（block3,4,5,6）
7. prefill 只需算 t49..t100（52 token），而非全部 100 token → TTFT 大降
8. block3,4,5 填满后 cache_full_blocks 写入缓存表（以后别人能复用）
```

**这就是"第二个相同 prompt 秒回"的全部秘密**：前 48 个 token 的 KV 直接复用（连 prefill 都不用算），只需算后面的新 token。

---

## 六、Prefix Caching 的代价与边界

天下没有免费午餐：
1. **hash 计算开销**：每个 block 要算 hash（但比 forward 便宜几个数量级，且用 LRU cache 缓存 hash 本身）
2. **驱逐重算**：被驱逐的 cache 后续若再命中，要重新 prefill。但这是"软失败"（miss 退化成正常 prefill），不是错误
3. **block_size 影响**：block 太大→命中粒度粗（16 token 差一点就整块 miss）；太小→hash 表项多，开销大。默认 16 是平衡点
4. **多模态/LoRA 的 extra_keys**：要正确区分，否则会错误共享（第3讲 compressed-tensors 的 lora_id、本讲的 extra_keys）

---

## 七、把第九讲和前八讲连起来

| 讲次 | 关系 |
|------|------|
| 第5讲 PagedAttention | 第9讲是它的"缓存层"——第5讲实现 block 分配/释放，第9讲实现 block 的**内容识别与共享** |
| 第6讲 Continuous Batching | prefix cache 命中让 prefill token 减少，直接影响 token_budget 分配 |
| 第8讲 Spec Decoding | spec 的 draft token 也可吃 prefix cache（相同 draft 模式） |
| **第9讲 Prefix Caching** | **让 PagedAttention 的"灵活显存"变成"可复用显存"** |

**第5讲和第9讲合起来才是完整的 vLLM KV cache 管理**：第5讲（物理层：分页分配）+ 第9讲（逻辑层：内容识别与共享）。面试被问"vLLM 怎么处理重复 prompt"，你的答案就是这两讲的串联。

---

## 八、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：链式 hash（基础）
1. 读 `kv_cache_utils.py:577 hash_block_tokens`。block_hash 依赖哪两个输入？为什么 parent_block_hash 是 None 时用 NONE_HASH？
2. 用讲义第二节的例子：A=`[t1..t6]`、B=`[t1..t4,x,y]`、block_size=2。写出 A.block1.hash 和 B.block1.hash 的计算式，证明它们相等。
3. `extra_keys` 参数（581行）的作用是什么？为什么多模态/LoRA 场景需要它？（提示：相同 token 但不同图片特征，KV 不同）

### 任务 B：查询与缓存（核心）
4. 读 `block_pool.py:199 get_cached_block`。为什么"任一 group miss 就整体返回 None"？（提示：多 group 一致性——要么全命中要么全 miss）
5. 读 `block_pool.py:226 cache_full_blocks`。第 260 行 `if num_cached_blocks >= num_full_blocks: return`——什么情况下会触发？为什么已缓存的不用再缓存？
6. 第 287 行 `num_hash_tokens = (num_cached_blocks + i + 1) * block_size`。这个 num_hash_tokens 记录的是什么？为什么 block 要记录"它覆盖了多少 token"？（提示：partial→full 升级时判断）

### 任务 C：共享与驱逐（机制）
7. 读 `block_pool.py:597 touch`。为什么 `ref_cnt==0` 时要先 `free_block_queue.remove(block)`？如果不移除，会发生什么错误？
8. 读 `block_pool.py:574 _maybe_evict_cached_block`。一个 block 被"驱逐"具体做了什么？为什么 ref_cnt>0 的 block 永远不会被驱逐（它根本不在 free list）？
9. 思考题：请求 A 和 B 共享 block0（ref_cnt=2）。A 结束释放，block0 的 ref_cnt 变成 1。此时 block0 会被驱逐吗？为什么？如果 B 也结束呢？

---

## 九、干中学实践任务（核心！）

> 在 `practice_prefix_caching.py` 里实现一个完整的 mini prefix cache。
> 依赖：仅标准库（`hashlib`）。不需要装 vllm/torch。
> 设计哲学：你不只实现 block 分配（第5讲做过），还要实现**内容识别（hash）+ 共享（ref_cnt）+ 驱逐（LRU）**。这是 BlockPool 的完整复刻。

### 实践 1：链式 hash + 缓存查询（热身）
实现：
- `compute_block_hash(parent_hash, tokens) -> hash`：链式 hash，`hash = hashlib(parent_hash + tokens)`
- `PrefixCache` 类：维护 `hash_to_block: dict[bytes, int]`（hash → 物理 block ID）
  - `query(block_hash) -> int | None`：查命中，返回 block ID 或 None
  - `insert(block_hash, block_id)`：写入缓存表
  - `evict(block_hash)`：从缓存表移除

验证：相同 (parent_hash, tokens) 产生相同 hash；不同 tokens 产生不同 hash。

### 实践 2：ref_cnt 共享 + LRU 驱逐（核心）
扩展 `PrefixCache`，给每个 block 加 `ref_cnt` 和 `last_used`（LRU 时间戳）：
- `acquire(block_hash) -> int | None`：查命中；命中则 `ref_cnt += 1`，更新 last_used，返回 block ID；miss 返回 None
- `release(block_id)`：`ref_cnt -= 1`；若 ref_cnt 归 0，标记为"可驱逐"（加入 LRU 候选）
- `evict_one() -> int | None`：从 ref_cnt=0 的 block 里按 LRU 选一个驱逐（移除 hash 记录），返回释放的 block ID
- `num_cached()`：返回当前缓存 block 数

验证场景：
```python
cache = PrefixCache(num_blocks=3)
# 请求 A 和 B 共享 prefix [t1,t2,t3,t4]
cache.insert(h0, 0); cache.insert(h1, 1)   # block0,1 缓存
bid = cache.acquire(h0)   # A 用 block0 → ref_cnt[0]=1
bid = cache.acquire(h0)   # B 也用 block0 → ref_cnt[0]=2（共享！）
cache.release(0)          # A 结束 → ref_cnt[0]=1，不驱逐（B 还在用）
# block0 不在驱逐候选，evict_one 不会选它
```

### 实践 3：端到端——两请求共享 prefix（整合）
实现 `serve_requests(requests: list[list[int]], block_size, num_blocks) -> dict`：
- 把每个请求的 prompt 切 block、算链式 hash
- 逐块查 cache：命中就 acquire（ref_cnt+1），miss 就分配新 block + insert
- 显存不够时 evict_one
- 统计：每个请求的 cache 命中率、prefill 节省的 token 数

验证：两个相同 prefix 的请求，第二个命中率应高（复用第一个的 block）；不同 prefix 的请求命中率低。

> 💡 实践 2 是灵魂。要点：① ref_cnt>0 的 block 绝不在驱逐候选集 ② release 只在 ref_cnt 降到 0 时才加入驱逐候选 ③ LRU 按 last_used 选最旧的 ④ 命中时更新 last_used（"最近用过"）。这套机制让"正在用的 cache 不会被误删"是 prefix caching 可靠性的根基。

---

## 十、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_prefix_caching.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_prefix_caching.py 运行结果，重点贴共享场景的 ref_cnt 变化和命中率）

---

## 十一、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现链式 hash 后，你对"区块链式累积"识别前缀的理解？② ref_cnt 共享保护——实践 2 里 A 释放后 B 还能用，这个"引用计数保护"你在哪见过（GC？）？③ 端到端测试里第二个相同 prefix 的命中率多高？这解释了 RAG 为什么快？

（完成实践后填写）

---

## 十二、个人复盘感悟（留给你写）

> 你是 AI Infra 求职者，建议角度：① 链式 hash 让"前 N block 相同 = 前 N×bs token 相同"自动成立，这种"用数据结构把 O(N) 比较降成 O(1) 查表"的设计哲学你怎么看？② ref_cnt 共享 + LRU 驱逐——这套"引用计数防误删 + LRU 淘汰冷数据"的组合，在 OS（页缓存）、数据库（buffer pool）里都有，你怎么评估它的普适性？③ prefix caching 对 RAG/agent 的价值，结合你的研究方向（量化），量化模型 + prefix cache 的组合上限在哪？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**PagedAttention（第5讲）+ Prefix Caching（第9讲）= 完整的 vLLM KV cache 管理**。完成后告诉我下一步：
> - **① 换领域**：MoE 专家路由 / LoRA 热加载 / 结构化输出（grammar）/ KV offload
> - **② 多模态**：prefix cache 的 extra_keys 在多模态的展开
> - **③ 回量化**：DeepGEMM/Machete 等 Hopper/Blackwell kernel
> - 或你指定的

---

## 附录：学习问答记录（Q&A 疑难澄清）

> 以下是在学习本讲过程中产生的关键提问与详细解答，集中在 prefix caching 最容易卡住的四个点：① block0 为何和 None 做 hash ② 为什么只认最长公共前缀 ③ 开头分叉/中间分叉的反例怎么办 ④ agent 多轮对话场景下"最长公共前缀"的合理性。这四个问题同根同源——**prefix caching 是为"前缀重复"而生的，不是为"任意片段重复"而生的**，这是它的设计边界，由 KV 的因果性物理决定。

---

### Q1：示例中为什么 block0 要和 None 做 hash？

这是链式 hash 的**起点问题**。

每个 block 的 hash 公式是：
```
block_hash = hash(parent_block_hash, curr_block_tokens)
```
这个公式对 block1、block2……都成立——它们都有"父 block"。但 **block0 没有父 block**，它是序列的第一个，前面什么都没有。公式要怎么套？

两个选择：
- **选择 A**：让 block0 特殊处理，用一套不同的规则（如 `hash(curr_tokens)` 不带 parent）
- **选择 B**：统一公式，给 block0 塞一个"占位的父 hash"

vLLM 选了 B，这个占位值就是 `NONE_HASH`（见源码 `kv_cache_utils.py:577`）：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    ...
    return BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

**为什么选 B 而不是 A？为了代码统一。** 如果 block0 用特殊规则，那整个系统里每个地方算 hash 都要分情况："这是不是第一个 block？"——查缓存、写缓存、链式比较全都要加 `if`。塞个 `NONE_HASH` 占位，所有 block 走**同一条公式、同一段代码**，没有特例。这就像链表里的**哨兵节点（dummy head）**：给链表加个不存数据的头节点，让"插入第一个元素"和"插入中间元素"用同一套指针操作。`NONE_HASH` 就是 block 链的 dummy head。

**一个关键的语义点：`NONE_HASH` 是个固定常量。** 它对所有请求、所有 block0 **都是同一个值**。这意味着：
- A 的 block0：`hash(NONE_HASH, [t1,t2])`
- B 的 block0：`hash(NONE_HASH, [t1,t2])`

两个请求只要第一个 block 的 token 一样，它们的 block0 hash 就**必然相等**（因为 parent 都是同一个 `NONE_HASH`）。这是链式 hash 能识别"公共前缀"的**地基**——所有前缀比较都从 `NONE_HASH` 这个共同起点出发。如果 `NONE_HASH` 每次随机变，那两个相同 token 的 block0 hash 就不同了，前缀识别从第一步就崩了。所以它**必须是固定的全局常量**。

> 一句话：**`NONE_HASH` 是链式 hash 公式的"起点哨兵"，让没有父 block 的 block0 也能套用统一公式，且作为所有请求的共同起点，保证相同 token 序列产生相同 hash。**

---

### Q2：为什么只考虑最长公共前缀？不前后比、不跳着比？

这是 prefix caching **最核心的设计哲学**，也是它和"任意子串匹配"的根本区别。答案分三层。

#### 第一层：为什么"前缀"？（不是任意片段）

KV cache 的本质决定了这点。回忆 attention 的因果性：**第 i 个 token 的 KV，依赖它前面所有 token**（因为每层 attention 都要 attend 到前面）。

```
token t1 的 KV  ← 只依赖 t1 之前的（空）
token t2 的 KV  ← 依赖 t1
token t3 的 KV  ← 依赖 t1, t2
...
token t100 的 KV ← 依赖 t1..t99
```

**KV 是"上下文相关"的，不是"token 本身相关"的。** 同一个 token `t5`，在序列 `[t1,t2,t3,t4,t5]` 里算出来的 KV，和在序列 `[t8,t2,t3,t4,t5]` 里算出来的 KV，**是完全不同的两份数据**——因为它们的前文不同，attention 算出来的值就不一样。

这意味着：**只有"前文完全相同"的那些 token，它们的 KV 才能复用。** 前文一旦不同，后面的 token 即使内容一样，KV 也得重算。所以能复用的，只能是"从序列开头连续相同的那一段"——也就是**前缀**。这是物理决定的，不是设计偷懒。

#### 第二层：为什么用"链式 hash"找最长公共前缀？

既然只能复用前缀，那就要找"两个请求从开头开始，连续相同到第几个 token"。链式 hash 是干这件事的**最优结构**：

```
block0.hash = hash(NONE_HASH, [t1,t2])
block1.hash = hash(block0.hash, [t3,t4])     ← 依赖 block0
block2.hash = hash(block1.hash, [t5,t6])     ← 依赖 block1
```

因为每个 block 的 hash 把"从开头到自己的整条历史"都压缩进去了，所以：
- **block0 hash 相同 ⟺ token[0:bs] 相同**
- **block1 hash 相同 ⟺ block0 相同 且 token[bs:2bs] 相同 ⟺ token[0:2bs] 相同**
- **block N hash 相同 ⟺ token[0:(N+1)bs] 全部相同**

于是"找最长公共前缀"退化成：从 block0 开始**逐块查表**，第一个 miss 的地方停。每个 block 查一次 O(1)，总共 O(N/bs)。**这就是第二节说的"逐 block 比较一个 hash 值，无需逐 token 比较"。**

#### 第三层：为什么"只考虑最长"，不考虑中间也重复？

**问题 ①：中间片段的 KV 不可复用（回到第一层）。** 就算两个请求中间都有 `[t3,t4,t5,t6]`，如果它们的前文不同，这 `[t3,t4,t5,t6]` 的 KV **就是不同的**，根本没法复用。所以"中间重复"在 KV cache 场景下**复用不了**，找了也白找。

**问题 ②：找"任意位置重复"的代价极高。** 找前缀是 O(N/bs)（从前往后扫）。找"任意子串匹配"是经典的**最长公共子串/后缀数组**问题，复杂度 O(N²) 甚至更高，而且要维护复杂的索引（后缀树/后缀数组）。对一个**每秒处理上百请求的推理引擎**来说，这个开销不可接受。prefix caching 用"只看前缀"换来了"几乎免费的命中率检测"。

> 一句话：**KV 的因果性决定了只有前缀可复用；链式 hash 让"找最长前缀"变成 O(N/bs) 的逐块查表；放弃"任意片段匹配"是因为它在 KV 场景复用不了且代价太高。prefix caching 是为 RAG/agent 这种"开头相同"的负载精确设计的。**

---

### Q3：开头分叉 / 中间分叉的反例怎么办？

这两个例子**正是 prefix caching 的盲区**，它们都不在"前缀重复"的范畴。逐个分析 vLLM 的真实行为，以及为什么这是"可接受的设计边界"。

#### 反例 (1)：A=`[t1,t2,t3,t4,t5]`，B=`[t8,t2,t3,t4,t5]`（block_size=2）

**直觉**：后面 `[t2,t3,t4,t5]` 四个完全一样，就因为第一个不同，要重算整个 prefill，太亏了。

**vLLM 实际行为**：确实**几乎全部重算**，但这是**正确的、不得不做的事**。看链式 hash：

```
A: block0=hash(NONE, [t1,t2])   block1=hash(A.b0, [t3,t4])   block2=hash(A.b1, [t5])
B: block0=hash(NONE, [t8,t2])   block1=hash(B.b0, [t3,t4])   block2=hash(B.b1, [t5])
        ↑ 不同!                      ↑ 因为 parent 不同,所以不同      ↑ 同理不同
```

B.block0 和 A.block0 的 hash **从第一个就不同**（因为 `[t8,t2] ≠ [t1,t2]`）。链式传播下去，B 的 block1、block2 全部 miss。**B 一点 cache 都吃不到，要 prefill 全部 5 个 token。**

**但这是对的，不是 bug。** 回到因果性：`[t8,t2,t3,t4]` 这个上下文下 t5 的 KV，和 `[t1,t2,t3,t4]` 上下文下 t5 的 KV，**是两份完全不同的数据**。B 就算 token 序列长得像 A，KV 也必须重新算。强行复用会导致 attention 算错——t5 会去"看"一个根本不属于它上下文的 KV。

**这种场景怎么办？不在 prefix caching 层解决，在上游规避。** 实际工程里：
- **RAG**：确保所有请求用**同一个 system prompt 开头**（system prompt 设计成固定前缀），差异部分放后面。这样前缀对齐，cache 命中。
- **Agent 多轮**：每轮的对话历史是**严格追加**的（前 N 轮一字不改，只加第 N+1 轮），天然是前缀结构。
- **Few-shot**：示例放在最前面固定不变，用户 query 放后面。

也就是说，**prefix caching 假设你的负载天然是"前缀重复"的**。它的设计哲学是"我不解决所有重复，我极其高效地解决最常见的那种重复"。如果你的请求是 A=`[t1,...]`、B=`[t8,...]` 这种开头就分叉的，prefix cache 帮不了你——但 RAG/agent 的真实负载**极少**长这样，因为 system prompt 是固定的。

#### 反例 (2)：A=`[t1,t2,t8,t4,t5]`，B=`[t1,t2,t3,t4,t5]`（中间不同）

**block_size=2 的划分**：
```
A: block0=[t1,t2]   block1=[t8,t4]   block2=[t5]
B: block0=[t1,t2]   block1=[t3,t4]   block2=[t5]
```

**vLLM 实际行为**：
- block0：`[t1,t2]` 相同 → **命中！复用**（省 2 个 token 的 prefill）
- block1：`[t8,t4] ≠ [t3,t4]` → miss
- block2：parent(B.block1) ≠ parent(A.block1)，链式传播 → miss

所以 B **只能复用 block0（2 个 token）**，后面的 `[t3,t4,t5]` 要重算。

**这里藏着一个更细的点：block_size 的影响。** 例子 block_size=2，粒度很细。如果 block_size=16（默认），那么 `[t1,t2]` 在同一个 block 里，但这个 block 还没填满（只有 2/16 个 token）。真实场景下，**只有填满的 block 才进缓存表**（见第三节 `cache_full_blocks`）。所以这种"开头 2 个相同"的情况，**在 block_size=16 下可能连 block0 都不命中**（因为 block0 是 partial 的）。这暴露了 prefix caching 的另一个边界：**命中粒度受 block_size 限制**。第六节专门列了这条："block 太大→命中粒度粗；太小→hash 表项多、开销大。默认 16 是平衡点。"

#### 总结：这两个反例说明了 prefix caching 的明确边界

| 场景 | 能否命中？ | 为什么 |
|------|----------|--------|
| 开头相同（前缀） | ✅ 命中 | KV 因果性允许复用，链式 hash 高效识别 |
| 开头不同、后面相同 | ❌ 不命中 | 后面的 KV 因前文不同而不同，**不能复用**（强行复用会算错） |
| 中间分叉 | ❌ 仅命中分叉前的部分 | 分叉后 KV 不同，不可复用 |

**核心认知**：prefix caching 不是"找两段文本哪里像"，而是"找两段 **KV** 哪里像"。而 KV 的相似性，**因果性地等价于前缀的相似性**。这是 attention 机制决定的物理事实，不是 vLLM 的偷懒。

---

### Q4：实际场景如 agent 多轮对话，"最长公共前缀"的合理性？举例说明

#### 先建立认知：agent 多轮对话在引擎眼里是什么

用户看到的是"对话"，但 LLM 推理引擎看到的是**每轮重新构造的一个完整 prompt**。每来一轮，引擎都要把"系统提示 + 全部历史轮次 + 最新用户输入"拼成一条新序列，做一次 prefill。

关键点：**每一轮的 prompt，都是上一轮 prompt 的"前缀 + 追加"。** 这不是巧合，而是多轮对话的固有结构。

设 system prompt 固定 S = `[s1, s2, ..., s100]`（100 token），block_size=16。每轮 user+assistant 约 16 token（1 个 block）。用 block 单位画图：

```
第1轮 prompt:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ s    │ s    │ s    │ s    │ s    │ s    │ r1   │  r1 = round1(user1+assistant1)
│ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ blk5 │ blk6 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘

第2轮 prompt:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ s    │ s    │ s    │ s    │ s    │ s    │ r1   │ r2   │  r2 = round2
│ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ blk5 │ blk6 │ blk7 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
└─────────────── 完全相同 ───────────────┘   ← 命中!复用
                                               只算这个新的

第3轮 prompt:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ s    │ s    │ s    │ s    │ s    │ s    │ r1   │ r2   │ r3   │
│ blk0 │ blk1 │ blk2 │ blk3 │ blk4 │ blk5 │ blk6 │ blk7 │ blk8 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
└─────────────────────── 完全相同 ───────────────────────┘  ← 命中!
                                                            只算这个
```

**看到关键规律了吗？多轮对话的每一轮 prompt，都恰好是上一轮 prompt 的严格前缀 + 尾部追加。** 这种结构下：
- **最长公共前缀 = 上一轮的全部内容**（去掉最后一轮的生成部分）
- 命中的 block 数量 = **上一轮 prompt 的 block 总数**
- **每一轮只需要 prefill 新增的那一两个 block**，而不是重算整个历史

#### 算一笔账：省了多少？

假设每轮 user+assistant 合计 50 token，system prompt 100 token。block_size=16。

| 轮次 | prompt 总长 | 不用 cache 要 prefill | 用 prefix cache 实际 prefill | 节省 |
|------|------------|---------------------|---------------------------|------|
| 1 | 150 | 150 | 150 | 0% |
| 2 | 200 | 200 | 50 | 75% |
| 3 | 250 | 256 | 50 | 80% |
| 4 | 300 | 300 | 50 | 83% |
| 5 | 350 | 350 | 50 | 86% |
| 10 | 600 | 600 | 50 | **92%** |

**到第 10 轮，只 prefill 50 token（这一轮新增的），其余 550 个 token 的 KV 全部命中复用。** 如果不用 prefix caching，每轮都要 prefill 整个 prompt，第 10 轨要算 600 token——**cache 让算力开销和"对话长度"解耦**，对话再长，每轮的实际 prefill 都只是"这一轮新增"的部分。

#### 为什么"最长公共前缀"在这里是"唯一合理"的？

**答案：在多轮对话里，"中间相同"和"后面相同"的片段，要么 KV 根本不可复用，要么自动就被前缀覆盖了。** 分两种情况看。

**情况 A：历史轮次里的内容，后续轮次不会改。** 这是多轮对话的**铁律**：你不会去修改第 1 轮的用户发言或助手回复。历史是**不可变**的、**只追加**的。所以历史部分（`[s]+[r1]+[r2]`）在三轮里**一模一样、顺序相同、位置相同**。它不是"中间有段相同"，而是**从开头到 round2 结尾，整条连续相同**——这就是最长公共前缀的精确定义。**如果只复用中间片段（比如只复用 r1 不复用 system prompt），反而更差**——因为 system prompt 那部分 KV 也要重算。最长公共前缀 = **能复用的最大化**，没有任何浪费。

**情况 B：那如果历史被"编辑"了呢？（中间分叉）** 比如第 3 轮时，用户回头改了第 1 轮的发言（有些 agent 框架允许"编辑历史"）。这时从 `user1` 开始，**后面所有的 KV 都失效了**——因为前面上下文变了，后面每个 token 的 KV 都要重算。这时最长公共前缀只剩 `[s]`（system prompt）那一小段。

**这恰恰证明了"只认前缀"的正确性**：
- 如果历史没改 → 前缀覆盖全部历史，最大化复用
- 如果历史改了 → 从改的地方往后 KV 全失效，前缀自动收缩到改动点之前，**不会错误复用已失效的 KV**

**最长公共前缀不是"妥协"，它是"在 KV 因果性约束下，能安全复用的最大范围的精确刻画"。** 它既能最大化复用（历史没改时），又能保证正确性（历史改了时自动收缩）。任何"超出前缀的复用"都会违反因果性导致 attention 算错。

#### 一个容易忽略的点：engine-side 的"自动对齐"

用户调用 API 时给的是 messages（结构化），引擎会套**对话模板（chat template）** 渲染成一条 token 序列：
```
<|system|>你是一个助手...<|user|>你好<|assistant|>你好!有什么可以帮你?<|user|>1+1等于几
```
模板里的特殊 token 和 system prompt 的渲染顺序是**确定的、可复现的**。所以只要 messages 的前缀部分（到某个 turn 为止）相同，渲染出来的 token 序列**前缀就完全相同** → block 对齐 → cache 命中。**这保证了 prefix caching 在 API 层面"开箱即用"**——用户不需要做任何特殊操作，只要按正常多轮对话方式调用，engine 渲染时天然产生前缀结构，cache 自动生效。这也是为什么 vLLM 文档说 prefix caching 对 agent 场景"几乎免费"。

#### 合理性的三个层次

| 层次 | 为什么"最长公共前缀"合理 |
|------|------------------------|
| **物理层** | KV 的因果性：token 的 KV 依赖所有前文。只有前文完全相同的 token，KV 才能复用。前缀 = "前文完全相同"的连续区段。 |
| **负载结构层** | agent/RAG 的 prompt 天然是"固定前缀 + 逐轮追加"，历史不可变。最长公共前缀正好覆盖整个历史，复用最大化。 |
| **工程效率层** | 链式 hash 把"找最长前缀"变成 O(N/bs) 逐块查表，几乎零开销。同时正确收缩（历史改了就只认改动点之前），保证不会错误复用。 |

---

### 一句话总收口

> block0 和 `NONE_HASH` 做 hash，是因为它没有父 block，塞个固定常量当哨兵，让所有 block 走统一公式，且作为所有请求的共同起点保证相同 token → 相同 hash。prefix caching 只认前缀，是因为 **KV 的因果性**决定了只有前文完全相同时 token 的 KV 才能复用，中间/后面相同但前文不同的片段**物理上不可复用**（强行复用会算错 attention）。链式 hash 把"找最长公共前缀"变成 O(N/bs) 的逐块查表，是这套机制高效的核心。反例（开头分叉、中间分叉）正是 prefix cache 的盲区，但真实 RAG/agent 负载天然是前缀结构（system prompt 固定 + 历史不可变逐轮追加），所以这个盲区在实践中很少踩到——**prefix caching 是为"前缀重复"这种最常见负载精确设计的，不是为"任意重复"设计的**。在 agent 多轮对话里，"最长公共前缀"恰好刻画了"能安全复用的 KV 范围"，让每轮 prefill 只算新增部分，TTFT 几乎不随对话长度增长。
