"""
特性 #15 干中学实践：mini KV Cache Offload（GPU↔CPU 两级 + LRU + 异步）

目标：亲手重建 KV offload 核心机制——
  ① 两级存储（GPU 小快 / CPU 大慢）+ LRU 驱逐
  ② 异步换入换出（submit + poll，模拟 CUDA stream 异步拷贝）
  ③ 命中率测量 + 局部性观察（局部性好 vs 差 / thrashing）

参考 vLLM 源码：
  - kv_offload/base.py:168   OffloadingManager（lookup/touch/prepare_load/prepare_store）
  - kv_offload/base.py:450   OffloadingWorker（submit_store/submit_load/get_finished）
  - kv_offload/base.py:512   offload_prompt_only（只 offload prefill block）

简化：用纯 Python dict 模拟两级存储，用队列模拟异步。
依赖：仅标准库（collections）。不需要装 vllm/torch。
运行：python practice_kv_offload.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
from collections import OrderedDict


# ============================================================================
# 实践 1：两级存储 + LRU 驱逐
# ============================================================================
class KVOffloader:
    """mini KV offload：GPU（小快，LRU）+ CPU（大慢）两级。
    参考 OffloadingManager（策略）+ OffloadingWorker（搬数据）的简化合一。
    """
    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
        # GPU 用 OrderedDict（LRU：访问时移到末尾，驱逐从头部）
        self.gpu: OrderedDict[int, str] = OrderedDict()
        self.cpu: dict[int, str] = {}
        # 统计
        self.gpu_hits = 0
        self.cpu_hits = 0      # 换入命中
        self.misses = 0        # 两级都没有

    def _evict_gpu_lru(self) -> int | None:
        """GPU 满时驱逐最久未用的 block 到 CPU。返回被驱逐的 block_id。"""
        # TODO(你): 实现
        # 若 cpu 也满，先驱逐 cpu 的一个（随机/FIFO，简化）
        # 从 gpu 头部 popitem(last=False)，移到 cpu
        pass

    def access(self, block_id: int) -> str | None:
        """访问 block。返回数据或 None（MISS）。
        - GPU 命中：返回，更新 LRU（移到末尾）
        - CPU 命中：换入 GPU（可能触发 GPU 驱逐），返回
        - MISS：返回 None（模拟需要重算 prefill）
        """
        # TODO(你): 实现
        pass

    def store(self, block_id: int, data: str) -> None:
        """存一个新 block 到 GPU（可能触发驱逐）。"""
        # TODO(你): 实现
        pass

    def stats(self) -> dict:
        total = self.gpu_hits + self.cpu_hits + self.misses
        return {
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "misses": self.misses,
            "gpu_hit_rate": self.gpu_hits / total if total else 0,
            "cpu_hit_rate": self.cpu_hits / total if total else 0,
        }


# ============================================================================
# 实践 2：异步换入换出（核心）
# ============================================================================
class AsyncKVOffloader(KVOffloader):
    """支持异步换入换出。submit 立即返回 job_id，poll 模拟完成。
    参考 OffloadingWorker.submit_store/submit_load + get_finished。
    """
    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        super().__init__(gpu_capacity, cpu_capacity)
        # 待完成的异步任务：{job_id: (op, block_id)}
        self.pending: dict[int, tuple[str, int]] = {}
        self._next_job = 0

    def submit_load(self, block_id: int) -> int:
        """提交异步换入（CPU→GPU）。立即返回 job_id，不阻塞。"""
        # TODO(你): 实现（加入 pending 队列）
        pass

    def submit_store(self, block_id: int) -> int:
        """提交异步换出（GPU→CPU）。"""
        # TODO(你): 实现
        pass

    def poll(self) -> list[int]:
        """模拟"过了一段时间"，完成所有 pending 任务。返回完成的 job_id 列表。
        真实系统里这是 CUDA stream 同步点。"""
        # TODO(你): 实现
        # 对每个 pending：执行对应的 load/store，标记完成
        pass

    def is_ready(self, block_id: int) -> bool:
        """block 是否在 GPU 且可用（不在 pending 换入中）。"""
        return block_id in self.gpu


# ============================================================================
# 实践 3：命中率测量 + 局部性
# ============================================================================
def simulate_access_pattern(accesses: list[int], gpu_cap: int,
                            cpu_cap: int) -> dict:
    """跑一个访问序列，返回命中率统计。
    accesses: 要访问的 block_id 序列
    """
    # TODO(你): 实现
    # 用 KVOffloader，预先 store 所有可能的 block 到 cpu
    # 然后按序列 access，统计命中率
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：两级存储 + LRU ===")
    off = KVOffloader(gpu_capacity=3, cpu_capacity=5)
    # 存 5 个 block 到 GPU（会驱逐到 CPU）
    for i in range(5):
        off.store(i, f"data{i}")
    # GPU 容量 3，应有 0,1,2 被驱逐到 CPU，GPU 留 2,3,4
    assert len(off.gpu) == 3, f"GPU 应剩3: {len(off.gpu)}"
    assert set(off.gpu.keys()) == {2, 3, 4}, f"GPU 应留2,3,4: {set(off.gpu.keys())}"
    assert 0 in off.cpu and 1 in off.cpu, f"0,1 应在CPU"

    # 访问 GPU 里的 4（GPU 命中）
    d = off.access(4)
    assert d == "data4"
    # 访问 CPU 里的 0（换入，触发 GPU LRU 驱逐）
    d = off.access(0)
    assert d == "data0", f"应从CPU换入: {d}"
    assert 0 in off.gpu, "0 应换入 GPU"
    # 驱逐了 GPU 里最旧的（2，因为 4 刚访问过最新，3其次）
    assert 2 not in off.gpu, f"应驱逐2: {set(off.gpu.keys())}"

    s = off.stats()
    assert s["gpu_hits"] == 1 and s["cpu_hits"] == 1
    print(f"  GPU(3) CPU(5)，存5个→驱逐0,1到CPU")
    print(f"  access(4) GPU命中, access(0) CPU换入(驱逐2)")
    print(f"  stats: {s} ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：异步换入换出 ===")
    off = AsyncKVOffloader(gpu_capacity=2, cpu_capacity=4)
    # 存 4 个 block
    for i in range(4):
        off.store(i, f"d{i}")
    # GPU 留 2,3，CPU 有 0,1

    # 异步换入 0（submit 立即返回，0 还没在 GPU）
    job = off.submit_load(0)
    assert not off.is_ready(0), "submit 后 0 还没 ready"
    # poll 前访问 0 应该不可用（或等待）
    assert not off.is_ready(0)

    # poll：完成异步任务
    done = off.poll()
    assert job in done, f"job {job} 应完成: {done}"
    assert off.is_ready(0), "poll 后 0 应 ready"

    # 再 submit 多个，一次 poll 全完成
    j1 = off.submit_load(1)
    j2 = off.submit_store(3)
    done = off.poll()
    assert j1 in done and j2 in done, f"两个都应完成: {done}"

    print(f"  submit_load(0) → is_ready=False（未完成）")
    print(f"  poll() → 完成，is_ready(0)=True")
    print(f"  批量 submit + 一次 poll 全完成 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：局部性 vs 命中率 ===")
    gpu_cap, cpu_cap = 3, 8

    # 局部性好：重复访问 0,1,2（都在 GPU 容量内）
    good_locality = [0, 1, 2] * 10
    s_good = simulate_access_pattern(good_locality, gpu_cap, cpu_cap)

    # 局部性差：随机访问 0-7（超出 GPU 容量，频繁换入换出）
    bad_locality = [0, 4, 1, 5, 2, 6, 3, 7] * 4
    s_bad = simulate_access_pattern(bad_locality, gpu_cap, cpu_cap)

    print(f"  局部性好 [0,1,2]*10: GPU命中率 {s_good['gpu_hit_rate']:.0%}")
    print(f"  局部性差 [0-7轮转]*4:  GPU命中率 {s_bad['gpu_hit_rate']:.0%}")
    assert s_good["gpu_hit_rate"] > s_bad["gpu_hit_rate"], \
        f"局部性好应GPU命中率更高: {s_good['gpu_hit_rate']} vs {s_bad['gpu_hit_rate']}"
    print(f"  → 局部性好时 GPU 命中率高（少换入换出）✓")
    print(f"  → 局部性差时频繁换入换出（thrashing）✓\n")


def main():
    print("mini KV Cache Offload 实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 KV offload 两级换页 + 异步 + 命中率测量。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
from collections import OrderedDict

class KVOffloader:
    def __init__(self, gpu_capacity, cpu_capacity):
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
        self.gpu = OrderedDict()
        self.cpu = {}
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.misses = 0

    def _evict_gpu_lru(self):
        if not self.gpu:
            return None
        # GPU 满了，驱逐头部（最旧）到 CPU
        if len(self.cpu) >= self.cpu_capacity:
            # CPU 也满，随便丢一个（FIFO 简化）
            self.cpu.pop(next(iter(self.cpu)))
        block_id, data = self.gpu.popitem(last=False)
        self.cpu[block_id] = data
        return block_id

    def access(self, block_id):
        if block_id in self.gpu:
            self.gpu_hits += 1
            self.gpu.move_to_end(block_id)   # 更新 LRU
            return self.gpu[block_id]
        if block_id in self.cpu:
            self.cpu_hits += 1
            # 换入：若 GPU 满先驱逐
            if len(self.gpu) >= self.gpu_capacity:
                self._evict_gpu_lru()
            data = self.cpu.pop(block_id)
            self.gpu[block_id] = data
            return data
        self.misses += 1
        return None

    def store(self, block_id, data):
        if len(self.gpu) >= self.gpu_capacity:
            self._evict_gpu_lru()
        self.gpu[block_id] = data

    def stats(self):
        total = self.gpu_hits + self.cpu_hits + self.misses
        return {
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "misses": self.misses,
            "gpu_hit_rate": self.gpu_hits / total if total else 0,
            "cpu_hit_rate": self.cpu_hits / total if total else 0,
        }


class AsyncKVOffloader(KVOffloader):
    def __init__(self, gpu_capacity, cpu_capacity):
        super().__init__(gpu_capacity, cpu_capacity)
        self.pending = {}
        self._next_job = 0

    def submit_load(self, block_id):
        job_id = self._next_job
        self._next_job += 1
        self.pending[job_id] = ("load", block_id)
        return job_id

    def submit_store(self, block_id):
        job_id = self._next_job
        self._next_job += 1
        self.pending[job_id] = ("store", block_id)
        return job_id

    def poll(self):
        done = []
        for job_id in list(self.pending.keys()):
            op, block_id = self.pending.pop(job_id)
            if op == "load":
                # 执行换入（同步 access 的换入逻辑）
                if block_id in self.cpu:
                    if len(self.gpu) >= self.gpu_capacity:
                        self._evict_gpu_lru()
                    self.gpu[block_id] = self.cpu.pop(block_id)
            elif op == "store":
                # 执行换出
                if block_id in self.gpu:
                    if len(self.cpu) >= self.cpu_capacity:
                        self.cpu.pop(next(iter(self.cpu)))
                    self.cpu[block_id] = self.gpu.pop(block_id)
            done.append(job_id)
        return done

    def is_ready(self, block_id):
        return block_id in self.gpu


def simulate_access_pattern(accesses, gpu_cap, cpu_cap):
    off = KVOffloader(gpu_cap, cpu_cap)
    # 预存所有 block 到 cpu（模拟"都算过 prefill，在 cpu 里"）
    all_blocks = set(accesses)
    for b in all_blocks:
        off.cpu[b] = f"data{b}"
    for b in accesses:
        off.access(b)
    return off.stats()
"""
