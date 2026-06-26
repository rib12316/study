"""
特性 #5 干中学实践：mini PagedAttention 分页 KV 管理器

目标：亲手重建 vLLM 的分页 KV cache 管理核心——
  ① BlockPool + 双向链表空闲队列（哨兵节点）
  ② KVCacheManager 用 block_table 做逻辑→物理映射，cdiv 算块数
  ③ LRU 驱逐 + 抢占（显存不足优雅降级）

参考 vLLM 源码：
  - kv_cache_utils.py:118  KVCacheBlock
  - kv_cache_utils.py:179  FreeKVCacheBlockQueue（双向链表 + 哨兵）
  - block_pool.py:144      BlockPool
  - kv_cache_manager.py:244 allocate_slots
  - single_type_kv_cache_manager.py:278 allocate_new_blocks (cdiv)

依赖：仅标准库。不需要装 vllm/torch。
运行：python practice_paged_attention.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations


def cdiv(n: int, b: int) -> int:
    """ceil division：PagedAttention 的元运算。cdiv(50,16)=4。"""
    return (n + b - 1) // b


# ============================================================================
# 实践 1：KVCacheBlock + 双向链表空闲队列
# ============================================================================
class KVCacheBlock:
    """物理块。参考 kv_cache_utils.py:118。
    简化：只保留 block_id 和链表指针（ref_cnt/hash 留到实践 4 可选）。
    """
    __slots__ = ("block_id", "prev_free", "next_free")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.prev_free: "KVCacheBlock | None" = None
        self.next_free: "KVCacheBlock | None" = None

    def __repr__(self):
        return f"Block({self.block_id})"


class FreeBlockQueue:
    """双向链表空闲队列，带哨兵头尾节点。参考 kv_cache_utils.py:179。

    设计要点（vLLM 这样做的原因）：
    - 哨兵 head/tail（block_id=-1）省去判空
    - popleft 取头部（最久未用的块）= LRU
    - append/free 加到尾部（最近释放的排最后）
    - 指针存在 block 字段里，便于未来 C++ 移植（不依赖 Python 对象容器）
    """
    def __init__(self, blocks: list[KVCacheBlock]):
        self.num_free_blocks = len(blocks)
        # TODO(你): 创建哨兵 head/tail，把 blocks 串成 head<->b0<->b1<->...<->tail
        pass

    def popleft(self) -> KVCacheBlock:
        """O(1) 取头部第一个真块，更新指针。参考 kv_cache_utils.py:232。
        若队列空（head.next 是 tail），raise ValueError。
        """
        # TODO(你): 实现
        pass

    def append(self, block: KVCacheBlock) -> None:
        """O(1) 把块加到 tail 前。参考 kv_cache_utils.py:322。"""
        # TODO(你): 实现
        pass

    def __len__(self):
        return self.num_free_blocks


# ============================================================================
# 实践 1（续）：BlockPool
# ============================================================================
class BlockPool:
    """物理块池。参考 block_pool.py:144。简化版：无 prefix cache hash。"""
    def __init__(self, num_blocks: int):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)

    def alloc(self, n: int) -> list[int]:
        """从空闲队列取 n 个块，返回 block_id 列表。不够则 raise。"""
        # TODO(你): 实现（调 free_queue.popleft n 次）
        pass

    def free(self, block_ids: list[int]) -> None:
        """归还块到空闲队列尾部（按给定顺序，先释放的排更前）。"""
        # TODO(你): 实现
        pass

    def num_free(self) -> int:
        return len(self.free_queue)


# ============================================================================
# 实践 2：KVCacheManager + block_table
# ============================================================================
class KVCacheManager:
    """请求级 KV 管理器。参考 kv_cache_manager.py:110。
    每请求维护 block_table: list[int]（逻辑块序号 i → 物理块 ID）。
    """
    def __init__(self, block_pool: BlockPool, block_size: int):
        self.block_pool = block_pool
        self.block_size = block_size
        self.block_tables: dict[str, list[int]] = {}   # req_id -> block_table

    def allocate_slots(self, req_id: str, num_tokens: int) -> list[int] | None:
        """给请求追加 token 对应的块。参考 kv_cache_manager.py:244。
        算法：
          num_required = cdiv(num_tokens, block_size)
          num_existing = len(block_table[req_id]) (不存在则0)
          need = num_required - num_existing
          若 need > num_free(): 返回 None（优雅降级，不 OOM）
          否则 alloc(need)，追加到 block_table，返回新分配的 block_id 列表
        """
        # TODO(你): 实现
        pass

    def get_block_table(self, req_id: str) -> list[int]:
        """返回请求的 block_table（逻辑→物理映射）。"""
        return self.block_tables.get(req_id, [])

    # 实践 3：抢占
    def preempt(self, req_id: str) -> None:
        """强制释放某请求所有块（清空它的 block_table，块归还 BlockPool）。
        模拟 vLLM 调度器显存不足时的抢占行为。"""
        # TODO(你): 实现
        pass

    def free(self, req_id: str) -> None:
        """请求正常结束，释放其块。"""
        self.preempt(req_id)


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：BlockPool + 双向链表 ===")
    pool = BlockPool(num_blocks=5)
    assert pool.num_free() == 5

    # alloc 3 个 → 应得 [0,1,2]（头部 LRU 顺序）
    ids = pool.alloc(3)
    assert ids == [0, 1, 2], f"alloc 顺序错: {ids}"
    assert pool.num_free() == 2

    # free [0,1] → 归还到尾部，现在空闲 = [3,4,1,0]（2 在用？不，2 也被free）
    pool.free([0, 1])
    assert pool.num_free() == 4

    # 再 alloc 2 → 取头部，应是 [3,4]（因为 1,0 在尾部，是最近释放的）
    ids2 = pool.alloc(2)
    assert ids2 == [3, 4], f"LRU 顺序错: {ids2}（应取最久未用的 3,4）"
    assert pool.num_free() == 2

    # alloc 超额应报错
    try:
        pool.alloc(5)
        print("  错误：超额 alloc 没报错")
        return
    except ValueError:
        pass

    print(f"  alloc/free/LRU 顺序正确 ✓")
    print(f"  双向链表 + 哨兵节点工作正常 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：block_table 分配 ===")
    pool = BlockPool(num_blocks=10)
    mgr = KVCacheManager(pool, block_size=16)

    # req0: 50 token → cdiv(50,16)=4 块
    new = mgr.allocate_slots("req0", 50)
    assert new is not None and len(new) == 4, f"50token 应分配 4 块: {new}"
    bt = mgr.get_block_table("req0")
    assert len(bt) == 4, f"block_table 长度: {len(bt)}"
    assert pool.num_free() == 6, f"剩余: {pool.num_free()}"

    # req0 追加到 100 token → cdiv(100,16)=7 块，已有4，再分3
    new = mgr.allocate_slots("req0", 100)
    assert new is not None and len(new) == 3, f"追加应分 3 块: {new}"
    assert len(mgr.get_block_table("req0")) == 7
    assert pool.num_free() == 3

    print(f"  req0 50token -> block_table {mgr.get_block_table('req0')[:4]}... (4块)")
    print(f"  req0 追加到100token -> +3块, 共7块")
    print(f"  cdiv 正确驱动分配 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：容量约束 + 抢占 ===")
    pool = BlockPool(num_blocks=10)
    mgr = KVCacheManager(pool, block_size=16)

    # req0 占 8 块 (128 token)
    mgr.allocate_slots("req0", 128)
    assert pool.num_free() == 2

    # req1 要 5 块 (80 token) → 空闲只有 2，应返回 None（不 OOM）
    result = mgr.allocate_slots("req1", 80)
    assert result is None, f"显存不足应返回 None，实际 {result}"
    assert len(mgr.get_block_table("req1")) == 0, "req1 不应被部分分配"

    # 抢占 req0 → 释放 8 块
    mgr.preempt("req0")
    assert pool.num_free() == 10, f"抢占后应全空闲: {pool.num_free()}"
    assert mgr.get_block_table("req0") == [], "req0 的 block_table 应清空"

    # 现在 req1 能分配了
    result = mgr.allocate_slots("req1", 80)
    assert result is not None and len(result) == 5, f"抢占后 req1 应成功: {result}"

    print(f"  req0 占8块, req1 申请5块 -> None (优雅降级, 无OOM) ✓")
    print(f"  preempt(req0) -> 释放8块 ✓")
    print(f"  req1 重新申请 -> 成功分5块 ✓")
    print(f"  → 你复现了 vLLM 调度器的抢占机制 ✓\n")


def main():
    print("mini PagedAttention 分页 KV 管理器实践\n" + "=" * 50 + "\n")
    try:
        verify_practice1()
    except Exception as e:
        print(f"  实践 1 未通过（先完成 TODO）: {e}\n")
        return
    try:
        verify_practice2()
    except Exception as e:
        print(f"  实践 2 未通过: {e}\n")
        return
    try:
        verify_practice3()
    except Exception as e:
        print(f"  实践 3 未通过: {e}\n")
        return
    print("=" * 50)
    print("🎉 全部通过！你已亲手实现 PagedAttention 的核心内存管理。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
class FreeBlockQueue:
    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        self.head = KVCacheBlock(-1)   # 哨兵
        self.tail = KVCacheBlock(-1)   # 哨兵
        if blocks:
            self.head.next_free = blocks[0]
            blocks[0].prev_free = self.head
            self.tail.prev_free = blocks[-1]
            blocks[-1].next_free = self.tail
            for i in range(len(blocks) - 1):
                blocks[i].next_free = blocks[i+1]
                blocks[i+1].prev_free = blocks[i]
        else:
            self.head.next_free = self.tail
            self.tail.prev_free = self.head

    def popleft(self):
        if self.head.next_free is self.tail:
            raise ValueError("No free blocks")
        first = self.head.next_free
        self.head.next_free = first.next_free
        first.next_free.prev_free = self.head
        first.prev_free = first.next_free = None
        self.num_free_blocks -= 1
        return first

    def append(self, block):
        last = self.tail.prev_free
        last.next_free = block
        block.prev_free = last
        block.next_free = self.tail
        self.tail.prev_free = block
        self.num_free_blocks += 1

    def __len__(self):
        return self.num_free_blocks


class BlockPool:
    def __init__(self, num_blocks):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)

    def alloc(self, n):
        if n > len(self.free_queue):
            raise ValueError(f"Not enough free blocks: need {n}, have {len(self.free_queue)}")
        return [self.free_queue.popleft().block_id for _ in range(n)]

    def free(self, block_ids):
        for bid in block_ids:
            self.free_queue.append(self.blocks[bid])

    def num_free(self):
        return len(self.free_queue)


class KVCacheManager:
    def __init__(self, block_pool, block_size):
        self.block_pool = block_pool
        self.block_size = block_size
        self.block_tables = {}

    def allocate_slots(self, req_id, num_tokens):
        num_required = cdiv(num_tokens, self.block_size)
        bt = self.block_tables.get(req_id, [])
        need = num_required - len(bt)
        if need > self.block_pool.num_free():
            return None
        if need > 0:
            new_ids = self.block_pool.alloc(need)
            bt = bt + new_ids
            self.block_tables[req_id] = bt
            return new_ids
        return []

    def get_block_table(self, req_id):
        return self.block_tables.get(req_id, [])

    def preempt(self, req_id):
        bt = self.block_tables.pop(req_id, [])
        if bt:
            self.block_pool.free(bt)
"""
