"""
特性 #9 干中学实践：mini Prefix Cache（链式 hash + ref_cnt 共享 + LRU 驱逐）

目标：亲手重建 vLLM BlockPool 的 prefix caching 核心——
  ① 链式 hash：block_hash = hash(parent_hash, tokens)，识别相同前缀
  ② ref_cnt 共享：多请求共享 block，ref_cnt>0 绝不驱逐
  ③ LRU 驱逐：ref_cnt=0 的 block 按 last_used 淘汰
  ④ 端到端：两请求共享 prefix，第二个吃 cache

参考 vLLM 源码：
  - kv_cache_utils.py:577  hash_block_tokens（链式 hash）
  - block_pool.py:199      get_cached_block（查询命中）
  - block_pool.py:226      cache_full_blocks（写入缓存）
  - block_pool.py:574      _maybe_evict_cached_block（驱逐）
  - block_pool.py:597      touch（ref_cnt 共享保护）

依赖：仅标准库（hashlib）。不需要装 vllm/torch。
运行：python practice_prefix_caching.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import hashlib


# ============================================================================
# 实践 1：链式 hash + 缓存查询
# ============================================================================
def compute_block_hash(parent_hash: bytes | None, tokens: tuple[int, ...]) -> bytes:
    """链式 hash。参考 kv_cache_utils.py:577。
    hash = sha256(parent_hash + tokens)
    parent_hash 为 None 时用 b"NONE"（对应 vLLM 的 NONE_HASH）。
    """
    # TODO(你): 实现
    pass


class PrefixCache:
    """mini prefix cache。维护 hash → block_id 映射。"""
    def __init__(self):
        # hash -> block_id
        self.hash_to_block: dict[bytes, int] = {}

    def query(self, block_hash: bytes) -> int | None:
        """查命中。返回 block_id 或 None。"""
        # TODO(你): 实现
        pass

    def insert(self, block_hash: bytes, block_id: int) -> None:
        """写入缓存表。"""
        # TODO(你): 实现
        pass

    def evict_hash(self, block_hash: bytes) -> int | None:
        """从缓存表移除一个 hash，返回被移除的 block_id。"""
        # TODO(你): 实现
        pass


# ============================================================================
# 实践 2：ref_cnt 共享 + LRU 驱逐
# ============================================================================
class CachedBlock:
    """一个被缓存的 block，带 ref_cnt 和 last_used。"""
    def __init__(self, block_id: int, block_hash: bytes):
        self.block_id = block_id
        self.block_hash = block_hash
        self.ref_cnt = 0
        self.last_used = 0     # 全局时钟，越大越近用


class PrefixCacheWithEviction:
    """支持 ref_cnt 共享和 LRU 驱逐的 prefix cache。
    参考 block_pool.py 的 touch/evict 机制。
    """
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.hash_to_block: dict[bytes, CachedBlock] = {}
        self.clock = 0    # 全局时钟，每次 touch/release 推进

    def _tick(self):
        self.clock += 1

    def acquire(self, block_hash: bytes) -> int | None:
        """查命中。命中则 ref_cnt+=1、更新 last_used，返回 block_id。miss 返回 None。
        参考 block_pool.py:597 touch。
        """
        # TODO(你): 实现
        pass

    def insert_new(self, block_hash: bytes, block_id: int) -> int | None:
        """分配新 block 并缓存。
        如果物理块用完（hash_to_block 满了），先 evict_one 腾位。
        返回 block_id；若无法腾位返回 None。
        """
        # TODO(你): 实现
        pass

    def release(self, block_id: int) -> None:
        """ref_cnt-=1。归零则成为驱逐候选（留在表里但 ref_cnt=0）。
        参考 block_pool.py free_blocks。
        """
        # TODO(你): 实现
        pass

    def evict_one(self) -> int | None:
        """从 ref_cnt=0 的 block 里选 last_used 最小的驱逐。
        参考 block_pool.py:574 _maybe_evict_cached_block。
        返回释放的 block_id；无候选返回 None。
        """
        # TODO(你): 实现
        pass

    def num_cached(self) -> int:
        return len(self.hash_to_block)

    def query(self, block_hash: bytes) -> int | None:
        """只读查询（不增 ref_cnt）。返回 block_id 或 None。"""
        blk = self.hash_to_block.get(block_hash)
        return blk.block_id if blk is not None else None

    def find_block_by_id(self, block_id: int) -> CachedBlock | None:
        for b in self.hash_to_block.values():
            if b.block_id == block_id:
                return b
        return None


# ============================================================================
# 实践 3：端到端——多请求共享 prefix
# ============================================================================
def serve_requests(requests: list[list[int]], block_size: int,
                   num_blocks: int) -> dict:
    """服务多个请求，统计 prefix cache 效果。
    每个请求：切 block、算链式 hash、逐块 acquire（命中复用）/insert_new（miss 新建）。
    所有请求结束后 release。
    返回：每请求命中率、总命中 block 数、总节省 token 数。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：链式 hash ===")
    # 相同输入相同 hash
    h0_a = compute_block_hash(None, (1, 2, 3))
    h0_b = compute_block_hash(None, (1, 2, 3))
    assert h0_a == h0_b, "相同输入应相同 hash"
    # 不同 token 不同 hash
    h1 = compute_block_hash(None, (1, 2, 4))
    assert h1 != h0_a, "不同 token 应不同 hash"
    # 链式：依赖 parent
    h_child_a = compute_block_hash(h0_a, (4, 5))
    h_child_b = compute_block_hash(h0_b, (4, 5))
    assert h_child_a == h_child_b, "相同 parent 相同 token 应相同"
    h_child_c = compute_block_hash(h1, (4, 5))   # 不同 parent
    assert h_child_c != h_child_a, "不同 parent 应不同 hash（即使 token 同）"

    cache = PrefixCache()
    cache.insert(h0_a, 0)
    assert cache.query(h0_a) == 0
    assert cache.query(h1) is None
    assert cache.evict_hash(h0_a) == 0
    assert cache.query(h0_a) is None
    print(f"  链式 hash 确定性 ✓，不同 parent/token 区分 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：ref_cnt 共享 + LRU 驱逐 ===")
    cache = PrefixCacheWithEviction(num_blocks=3)
    h0 = compute_block_hash(None, (1, 2, 3))

    # A 和 B 共享 block0
    cache.insert_new(h0, 0)
    assert cache.acquire(h0) == 0       # A 用 → ref_cnt=1
    assert cache.acquire(h0) == 0       # B 也用 → ref_cnt=2（共享！）
    blk = cache.find_block_by_id(0)
    assert blk.ref_cnt == 2, f"ref_cnt 应=2: {blk.ref_cnt}"

    # A 释放 → ref_cnt=1，不驱逐（B 还在用）
    cache.release(0)
    assert cache.find_block_by_id(0).ref_cnt == 1
    # evict_one 不应驱逐 block0（ref_cnt=1>0）
    evicted = cache.evict_one()
    assert evicted is None, f"ref_cnt>0 不应被驱逐: {evicted}"

    # B 也释放 → ref_cnt=0，成为驱逐候选
    cache.release(0)
    assert cache.find_block_by_id(0).ref_cnt == 0
    # 现在能驱逐了
    evicted = cache.evict_one()
    assert evicted == 0, f"ref_cnt=0 应可驱逐: {evicted}"
    assert cache.query(h0) is None

    # LRU 顺序测试
    cache2 = PrefixCacheWithEviction(num_blocks=3)
    ha, hb, hc = (compute_block_hash(None, (i,)) for i in range(3))
    cache2.insert_new(ha, 0); cache2.insert_new(hb, 1); cache2.insert_new(hc, 2)
    cache2.acquire(ha)   # 用 a，last_used 更新
    cache2.release(0)
    cache2.acquire(hb); cache2.release(1)
    cache2.acquire(hc); cache2.release(2)
    # 都释放了，按 last_used：a 最先 used（最旧），应先驱逐
    evicted = cache2.evict_one()
    assert evicted == 0, f"LRU 应驱逐最旧的 a: {evicted}"
    print(f"  ref_cnt 共享保护 ✓，LRU 驱逐顺序 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：多请求共享 prefix ===")
    # 两个请求，前缀相同 [1..32]，后缀不同
    block_size = 16
    req_a = list(range(32)) + [100, 101]   # [0..31] + [100,101]
    req_b = list(range(32)) + [200, 201]   # [0..31] + [200,201]  前32相同
    req_c = list(range(50, 70))            # 完全不同

    result = serve_requests([req_a, req_b, req_c], block_size, num_blocks=10)
    # req_b 应复用 req_a 的前 32 token（2 个 block），命中率 > req_c
    assert result["hit_rates"][1] > result["hit_rates"][2], \
        f"req_b(同前缀)命中率应 > req_c(不同): {result['hit_rates']}"
    assert result["hit_rates"][1] >= 0.5, f"req_b 命中率应高: {result['hit_rates'][1]}"
    print(f"  命中率: {[f'{r:.0%}' for r in result['hit_rates']]}")
    print(f"  总命中 block: {result['total_hits']}, 节省 token: {result['saved_tokens']}")
    print(f"  → 第二个相同 prefix 请求大量复用 ✓\n")


def main():
    print("mini Prefix Cache 实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 Prefix Cache 的 hash/ref_cnt/LRU 核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def compute_block_hash(parent_hash, tokens):
    p = parent_hash if parent_hash is not None else b"NONE"
    return hashlib.sha256(p + bytes(tokens)).digest()


class PrefixCache:
    def __init__(self):
        self.hash_to_block = {}
    def query(self, block_hash):
        return self.hash_to_block.get(block_hash)
    def insert(self, block_hash, block_id):
        self.hash_to_block[block_hash] = block_id
    def evict_hash(self, block_hash):
        return self.hash_to_block.pop(block_hash, None)


class PrefixCacheWithEviction:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.hash_to_block = {}
        self.clock = 0

    def _tick(self):
        self.clock += 1

    def acquire(self, block_hash):
        blk = self.hash_to_block.get(block_hash)
        if blk is None:
            return None
        blk.ref_cnt += 1
        self._tick()
        blk.last_used = self.clock
        return blk.block_id

    def insert_new(self, block_hash, block_id):
        if len(self.hash_to_block) >= self.num_blocks:
            evicted = self.evict_one()
            if evicted is None:
                return None
        blk = CachedBlock(block_id, block_hash)
        self.hash_to_block[block_hash] = blk
        return block_id

    def release(self, block_id):
        for b in self.hash_to_block.values():
            if b.block_id == block_id:
                b.ref_cnt = max(0, b.ref_cnt - 1)
                return

    def evict_one(self):
        # 从 ref_cnt=0 的里选 last_used 最小
        candidates = [b for b in self.hash_to_block.values() if b.ref_cnt == 0]
        if not candidates:
            return None
        victim = min(candidates, key=lambda b: b.last_used)
        del self.hash_to_block[victim.block_hash]
        return victim.block_id

    def num_cached(self):
        return len(self.hash_to_block)

    def query(self, block_hash):
        blk = self.hash_to_block.get(block_hash)
        return blk.block_id if blk is not None else None

    def find_block_by_id(self, block_id):
        for b in self.hash_to_block.values():
            if b.block_id == block_id:
                return b
        return None


def serve_requests(requests, block_size, num_blocks):
    cache = PrefixCacheWithEviction(num_blocks)
    hit_rates = []
    total_hits = 0
    saved_tokens = 0
    for req in requests:
        acquired = []   # 本次请求 acquire/insert 的 block_id
        hits = 0
        blocks_in_req = 0
        parent_hash = None
        for i in range(0, len(req), block_size):
            chunk = tuple(req[i:i+block_size])
            if len(chunk) < block_size:
                break   # 跳过未满 block
            bh = compute_block_hash(parent_hash, chunk)
            blocks_in_req += 1
            bid = cache.acquire(bh)
            if bid is not None:
                hits += 1
                saved_tokens += block_size
                acquired.append(bid)
            else:
                bid = cache.insert_new(bh, blocks_in_req - 1)
                if bid is not None:
                    cache.acquire(bh)   # 自己也要引用
                    acquired.append(bid)
            parent_hash = bh
        hit_rates.append(hits / max(blocks_in_req, 1))
        total_hits += hits
        # 请求结束 release（简化：全释放）
        for bid in set(acquired):
            cache.release(bid)
    return {"hit_rates": hit_rates, "total_hits": total_hits,
            "saved_tokens": saved_tokens}
"""
