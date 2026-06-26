# 特性 #5：PagedAttention —— 分页 KV Cache 是怎么消灭显存碎片的

> 学习阶段：AI Infra 基础储备 / 跨入推理调度领域
> 对应源码：`vllm/v1/core/block_pool.py` + `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/kv_cache_utils.py` + `vllm/v1/core/single_type_kv_cache_manager.py`
> 本讲定位：**这是 vLLM 一战成名的根本创新**，也是面试高频题第一名。前面四讲都是量化（"省权重显存"），这一讲换到调度（"省 KV cache 显存"）。理解 PagedAttention，你才理解为什么 vLLM 比 HuggingFace `generate()` 吞吐高 10 倍以上。
> 干中学原则：本讲你要**亲手实现一个 mini 分页 KV 管理器**——固定 block_size、物理块池、block_table（逻辑→物理映射）、LRU 空闲队列。这是 vLLM `BlockPool` + `KVCacheManager` 的精简复刻。

---

## 一、为什么需要 PagedAttention？（背景，面试必答）

### 1.1 朴素 KV cache 的灾难：显存碎片

LLM 推理时，每个 token 的 attention 需要 K、V 两个张量（从前面所有 token 来）。这些 KV 缓存起来避免重算。朴素做法（HF `generate`）是：**给每个请求预先分配一整块连续显存**，大小 = `max_seq_len × num_layers × 2 × hidden × dtype_size`。

举例：Llama-7B，max_seq_len=2048，batch=8。每个请求 KV cache 可能要 1~2GB。问题在于：

1. **内部碎片（internal fragmentation）**：你声明 max=2048，但实际请求可能只生成到 200 token。预分配的 2048 槽位里 90% 浪费。
2. **外部碎片（external fragmentation）**：batch 里各请求实际长度不同，连续分配后显存被切成大小不一的空洞，新请求找不到合适整块而失败，即使总剩余显存够。

> 💡 实测：朴素连续分配下，KV cache 显存利用率往往只有 **20%~40%**。这意味着同样一张 A100，能并发服务的请求数被严重限制——这正是吞吐的瓶颈。

### 1.2 OS 启发：分页（paging）

vLLM 的灵感来自**操作系统的虚拟内存分页**：

| OS 虚拟内存 | vLLM PagedAttention |
|------------|---------------------|
| 进程的虚拟地址空间 | 一个请求的逻辑 token 序列 |
| 物理内存页（4KB） | 物理 KV cache 块（block_size=16 token） |
| 页表（虚拟页→物理页） | block_table（逻辑块→物理块 ID） |
| 进程不需要连续物理内存 | 请求的 KV cache 不需要连续物理块 |
| 按需调页 | 按需分配 block |

**核心思想**：KV cache 不再是"一个请求一大块连续显存"，而是**被切成固定大小的 block（页），散落在物理显存各处，靠 block_table 记录映射**。这样：
- 没有内部碎片（用到几个 token 分配几个 block，按 block_size 向上取整，浪费 ≤15 token）
- 没有外部碎片（所有 block 等大，任意空闲 block 都能用）
- **显存利用率从 20% 飙升到 ~96%+**

> 💡 面试一句话答：**PagedAttention 把 KV cache 按固定大小分块（block/paging），用 block_table 把逻辑序列映射到非连续的物理块，消除了传统连续分配的内外碎片，显存利用率从 20% 提升到 96%，从而大幅提高并发请求数和吞吐。**

---

## 二、核心数据结构（v1 实现）

### 2.1 KVCacheBlock：物理块（kv_cache_utils.py:118）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                          # 物理块编号
    ref_cnt: int = 0                       # 引用计数（prefix cache 共享用）
    # 双向链表指针，用于空闲队列
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    # hash 相关（prefix cache 识别用）
    block_hash: BlockHash | None = None
```

每个物理块就这几个字段。`ref_cnt` 让多个请求共享同一块（比如大家都用了同样的 system prompt，那 prompt 的 KV cache 块共享，ref_cnt>1）。这是 prefix caching 的基础。

### 2.2 FreeKVCacheBlockQueue：空闲队列（kv_cache_utils.py:179）

这是最值得学的设计。vLLM 没有用 Python 的 `list` 或 `collections.deque` 当空闲队列，而是**手写了一个双向链表**：

```python
class FreeKVCacheBlockQueue:
    """Free blocks organized as a doubly linked list.
    
    The queue is ordered by block ID initially. When a block is allocated
    and then freed, it's appended back with eviction order:
    1. All blocks are freed in the same eviction batch...
    2. If two blocks have the same last accessed time...
    Note we maintain this order by *reversing* the block order when free.
    """
    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 哨兵头
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)  # 哨兵尾
        # 把 blocks 串成链表，head<->b0<->b1<->...<->tail
        ...
    def popleft(self) -> KVCacheBlock: ...      # O(1) 取头部
    def append(self, block): ...                # O(1) 加尾部（LRU 顺序）
```

**为什么不用 deque？** 注释（184行）说得很清楚：
> *"this class does not allocate any Python objects when..."*

意思是：vLLM 的目标是未来用 C++ 重写 worker，deque 是 Python 对象，跨 FFI 边界麻烦；而把链表指针直接存在 `KVCacheBlock` 的字段里（`prev_free_block`/`next_free_block`），C++ 侧只需一个裸指针数组就能实现，零额外分配。**这是"为性能和可移植性而设计"的工程思维**——你做 Infra 会反复遇到这种"看起来过度设计，实则有远见"的取舍。

> 💡 另一个细节：链表用**两个哨兵节点**（`fake_free_list_head/tail`，block_id=-1）简化边界判断，这是经典链表技巧。`popleft` 和 `append` 都不需要判空。

### 2.3 BlockPool：物理块池（block_pool.py:144）

```python
class BlockPool:
    def __init__(self, num_blocks, ...):
        self.free_block_queue = FreeKVCacheBlockQueue(...)   # 空闲队列
        self.block_hash_to_block: BlockHashToBlockMap = ...  # hash→block（prefix cache）
    
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        # 从空闲队列 popleft n 个
    def free_blocks(self, ordered_blocks):
        # 归还到空闲队列尾部（LRU）
    def get_cached_block(self, ...):
        # prefix cache 命中查询
    def _maybe_evict_cached_block(self, block):
        # LRU 驱逐：ref_cnt 归零且需要腾位时，从 hash 表移除
```

### 2.4 block_table：逻辑→物理映射（核心！）

这是 PagedAttention 的"页表"。每个请求有一个 `block_table`，形如 `[5, 2, 8, 1]`，意思是：
- 逻辑块 0（token 0~15）→ 物理块 5
- 逻辑块 1（token 16~31）→ 物理块 2
- 逻辑块 2（token 32~47）→ 物理块 8
- 逻辑块 3（token 48~63）→ 物理块 1

**物理块完全不需要连续**。attention kernel（如 FlashAttention）跑的时候，根据 block_table 去散落的物理块取 K/V。

KVCacheManager 通过 `get_block_ids(request_id)`（kv_cache_manager.py:584）返回这个 block_table。你的实践任务要实现的就是这个映射。

---

## 三、分配流程：allocate_slots（kv_cache_manager.py:244）

当一个请求要追加 token 时，`allocate_slots` 决定分配多少块：

```
布局：| 已计算(comp) | 新命中prefix(new_comp) | 外部计算(ext_comp) | 新token(new) | lookahead |
```

核心计算（简化）：
```python
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_required_blocks = cdiv(num_tokens_main_model, block_size)   # cdiv = ceil除法
# 已有的块数
num_existing_blocks = len(existing_block_table)
# 需要新分配的块数
num_blocks_to_alloc = num_required_blocks - num_existing_blocks
# 从 BlockPool 取
new_blocks = block_pool.get_new_blocks(num_blocks_to_alloc)
# 追加到 block_table
block_table.extend([b.block_id for b in new_blocks])
```

`cdiv`（ceil division）是 PagedAttention 的元运算：`cdiv(n, b) = (n + b - 1) // b`。比如 `cdiv(50, 16) = 4`（50 token 需要 4 个 16-token 的块）。`single_type_kv_cache_manager.py:296` 用它算需求。

### 3.1 分配失败的优雅降级

如果 `num_blocks_to_alloc > block_pool.get_num_free_blocks()`，`allocate_slots` 返回 `None`（kv_cache_manager.py:386-387）。调度器收到 None 就**不调度这个请求**（或抢占别的请求）。这就是 vLLM 的"按 KV cache 容量调度"——绝不会因为显存不足而 OOM，而是优雅地少服务几个请求。这是和朴素 `generate` 最大的运行时差异。

---

## 四、Prefix Caching（PagedAttention 的"超能力"）

分页带来的副产品是 **prefix caching 天然可行**。原理：

1. 每个 block 计算一个 hash（基于它的 token 内容 + 前驱 block 的 hash，像区块链）。`kv_cache_utils.py:577 hash_block_tokens`。
2. BlockPool 维护 `block_hash_to_block` 映射（block_pool.py:34）。
3. 新请求来了，把它的 prompt 切成 block，逐块算 hash，查 `get_cached_block`（block_pool.py:199）。命中就**复用已有物理块**（`ref_cnt += 1`），不重新计算。

因为 block 是固定大小、可共享的（ref_cnt），所以"两个请求共享同一个 system prompt 的 KV cache"这件事，在分页架构下是 O(1) 的指针共享，而连续分配架构下根本做不到（因为两个请求的连续空间无法重叠）。

> 💡 block hash 链式计算（`hash = f(tokens_i, prev_hash)`）是关键：它让"前 N 个 block 的 hash 命中"等价于"前 N×block_size token 的 KV cache 命中"。所以 prefix cache 命中粒度是 block_size（默认 16 token），不是单 token。

---

## 五、分页的代价：kernel 间接寻址

天下没有免费午餐。分页让 KV cache 散落，attention kernel 不能再 `memcpy` 连续内存，而要**按 block_table 间接寻址**。这带来两个工程挑战：

1. **kernel 要改写**：FlashAttention/xFormers 原本假设连续 KV，vLLM 要么用 paged 版 kernel（老 v0），要么在 v1 里靠 block_table gather。这是 PagedAttention 论文的核心技术贡献——把 paging 做进了 attention kernel。
2. **block_size 的权衡**：block 太大→碎片回潮；block 太小→block_table 变长，间接寻址开销大，kernel 效率低。vLLM 默认 `block_size=16`，是经验最优点。

> 💡 你的实践任务会亲手感受 block_size 的影响：用 block_size=16 vs 4 vs 64，看 block_table 长度和碎片率的变化。

---

## 六、把第五讲和量化主线连起来

| 维度 | 量化主线（第1~4讲） | PagedAttention（第5讲） |
|------|-------------------|----------------------|
| 优化的资源 | 权重显存（4bit/8bit） | KV cache 显存（分页） |
| 核心思想 | 减少权重精度 | 消除 KV 碎片 |
| 数据结构 | ScalarType（类型） | block_table（映射） |
| 对吞吐的影响 | 降显存→塞更大模型 | 提利用率→塞更多并发请求 |
| 面试定位 | 偏算法/系统优化 | 偏系统/调度，vLLM 招牌 |

两者正交：一个模型可以同时是 4bit 量化 + PagedAttention，双重省显存。理解这两个维度，你就理解了"为什么 vLLM 能在单卡跑大模型高并发"的核心。

---

## 七、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：分页基础（基础）
1. 读 `kv_cache_utils.py:179 FreeKVCacheBlockQueue`。为什么用双向链表而不是 `collections.deque`？注释（184行）怎么解释的？
2. `FreeKVCacheBlockQueue` 用了两个 `fake_free_list_head/tail` 哨兵节点（block_id=-1）。`popleft`（232行）时，如果队列里只有一个真 block，哨兵节点是怎么被正确更新的？画图说明指针变化。
3. 读 `single_type_kv_cache_manager.py:278 allocate_new_blocks`。`num_required_blocks = cdiv(num_tokens, self.block_size)` 里 `cdiv` 是什么？请求有 50 个 token、block_size=16，需要几个块？

### 任务 B：block_table 与分配（核心）
4. `kv_cache_manager.py:584 get_block_ids` 返回的是 block_table。它的元素是什么——逻辑块序号还是物理块 ID？为什么 attention kernel 需要的是后者？
5. 读 `kv_cache_manager.py:244 allocate_slots` 的布局图（290-311行）。`<comp>`、`<new_comp>`、`<new>` 三段分别代表什么？为什么 `<comp>` 和 `<new_comp>` 都算"已计算"但处理方式不同？
6. `allocate_slots` 第 386-387 行：当 `required_blocks > get_num_free_blocks()` 时返回 None。调度器收到 None 会怎么做？这种设计避免了什么灾难？

### 任务 C：prefix cache 与驱逐（机制）
7. 读 `block_pool.py:199 get_cached_block`。它用什么 key 查 hash 表？block 的 hash 是怎么算的（提示：`kv_cache_utils.py:577 hash_block_tokens`，注意它依赖前驱 block 的 hash）？
8. `block_pool.py:574 _maybe_evict_cached_block` 什么时候会驱逐一个 cached block？为什么用 LRU 而不是 FIFO？
9. 思考题：两个请求共享同一个 system prompt，它们的 block_table 会指向同样的物理块吗？此时 `ref_cnt` 是多少？如果其中一个请求结束（free），共享块会被释放吗？为什么？

---

## 八、干中学实践任务（核心！）

> 在 `practice_paged_attention.py` 里实现一个 mini 分页 KV 管理器。
> 依赖：仅标准库。不需要装 vllm/torch。
> 设计哲学：你不读 vLLM 的 BlockPool，而是**重建**它。能正确分配/释放/驱逐/复用，才算真懂 PagedAttention。

### 实践 1：BlockPool + 空闲队列（热身）
实现 `BlockPool`：
- 初始化 N 个物理块（id 0~N-1），全空闲
- `alloc(n)`：从空闲队列取 n 个块，返回它们的 block_id 列表（取头部，O(1)）
- `free(block_ids)`：归还块到空闲队列尾部（LRU 顺序：最后释放的排最后，最久没用的排前面）
- `num_free()`：返回空闲块数

要求用**双向链表**（带哨兵头尾）实现空闲队列，模拟 vLLM 的设计。不要直接用 deque（那是后面思考题）。

### 实践 2：block_table 分配（核心）
实现 `KVCacheManager`：
- 每个请求维护一个 `block_table: list[int]`（逻辑→物理块 ID）
- `allocate_slots(req_id, num_tokens, block_size)`：用 `cdiv` 算需要的总块数，减去已有块数 = 新分配数，调 `BlockPool.alloc`，追加到 block_table
- `get_block_table(req_id)`：返回该请求的 block_table

验证：
```python
mgr = KVCacheManager(block_pool=BlockPool(num_blocks=10), block_size=16)
mgr.allocate_slots("req0", 50)   # 50 token → cdiv(50,16)=4 块
assert len(mgr.get_block_table("req0")) == 4
assert mgr.block_pool.num_free() == 6
```

### 实践 3：LRU 驱逐 + 容量约束（进阶）
扩展 `KVCacheManager`：
- `allocate_slots` 在空闲块不足时返回 `None`（不 OOM，优雅降级）
- 增加抢占：`preempt(req_id)` 强制释放某请求所有块（把它的 block_table 清空，块归还 BlockPool）

验证场景：
```python
# 10 块，block_size=16。req0 占 8 块，req1 占 5 块 → req1 alloc 失败返回 None
# preempt req0 → 释放 8 块 → req1 现在 alloc 成功
```
这个场景模拟 vLLM 调度器"显存不足时抢占低优先级请求"的真实行为。

> 💡 实践 1 的双向链表是最大难点。要点：① 用哨兵节点避免判空 ② `popleft` 要同时更新被取节点的 prev/next 为 None ③ `free` 加到 tail 前。vLLM 源码（kv_cache_utils.py:232-265）就是这么做，可对照。我会留一个简化选项：如果你觉得链表太繁，先用 deque 实现一个能跑的版本，再挑战链表版——但要在感悟里写下两者的差异。

---

## 九、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_paged_attention.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_paged_attention.py 运行结果）

---

## 十、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现双向链表空闲队列时，哨兵节点帮你省了什么麻烦？② cdiv 这个简单的运算，为什么是整个分页系统的元运算？③ 实践 3 的抢占场景，你对"调度器为了不 OOM 而主动释放正在跑的请求"这种设计有什么感觉？

（完成实践后填写）

---

## 十一、个人复盘感悟（留给你写）

> 你是 AI Infra 方向求职者，建议角度：① PagedAttention 借鉴 OS 分页，这种"跨领域迁移"的系统设计，你还见过哪些例子？② vLLM 为了 C++ 可移植手写双向链表而不用 deque，这种"为未来重构铺路"的工程取舍你怎么看？③ prefix caching 能让多请求共享 KV block，这对你设计 RAG / 多租户 LLM 服务有什么启发？④ block_size 这个参数的 trade-off，你怎么权衡？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。完成后告诉我下一步方向：
> - **① Continuous Batching**（和 PagedAttention 是一对，前者解决"算力利用"，后者解决"显存利用"，合起来才是完整 vLLM 调度）
> - **② Prefix Caching 深入**（本讲只点到，可单独深挖 hash 链、multi-modal hash、eviction 策略）
> - **③ Chunked Prefill**（把长 prompt 切块和 decode 混合调度，v1 默认开启）
> - 或其他你想了解的
