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
