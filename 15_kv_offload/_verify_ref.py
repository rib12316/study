"""验证参考答案。仅标准库。"""
from collections import OrderedDict

class KVOffloader:
    def __init__(self, gc, cc):
        self.gpu_capacity=gc; self.cpu_capacity=cc
        self.gpu=OrderedDict(); self.cpu={}
        self.gpu_hits=0; self.cpu_hits=0; self.misses=0
    def _evict_gpu_lru(self):
        if not self.gpu: return None
        if len(self.cpu)>=self.cpu_capacity: self.cpu.pop(next(iter(self.cpu)))
        bid,data=self.gpu.popitem(last=False); self.cpu[bid]=data; return bid
    def access(self, bid):
        if bid in self.gpu:
            self.gpu_hits+=1; self.gpu.move_to_end(bid); return self.gpu[bid]
        if bid in self.cpu:
            self.cpu_hits+=1
            if len(self.gpu)>=self.gpu_capacity: self._evict_gpu_lru()
            data=self.cpu.pop(bid); self.gpu[bid]=data; return data
        self.misses+=1; return None
    def store(self, bid, data):
        if len(self.gpu)>=self.gpu_capacity: self._evict_gpu_lru()
        self.gpu[bid]=data
    def stats(self):
        t=self.gpu_hits+self.cpu_hits+self.misses
        return {"gpu_hits":self.gpu_hits,"cpu_hits":self.cpu_hits,"misses":self.misses,
                "gpu_hit_rate":self.gpu_hits/t if t else 0,"cpu_hit_rate":self.cpu_hits/t if t else 0}

class AsyncKVOffloader(KVOffloader):
    def __init__(self, gc, cc):
        super().__init__(gc,cc); self.pending={}; self._next_job=0
    def submit_load(self, bid):
        j=self._next_job; self._next_job+=1; self.pending[j]=("load",bid); return j
    def submit_store(self, bid):
        j=self._next_job; self._next_job+=1; self.pending[j]=("store",bid); return j
    def poll(self):
        done=[]
        for j in list(self.pending.keys()):
            op,bid=self.pending.pop(j)
            if op=="load":
                if bid in self.cpu:
                    if len(self.gpu)>=self.gpu_capacity: self._evict_gpu_lru()
                    self.gpu[bid]=self.cpu.pop(bid)
            elif op=="store":
                if bid in self.gpu:
                    if len(self.cpu)>=self.cpu_capacity: self.cpu.pop(next(iter(self.cpu)))
                    self.cpu[bid]=self.gpu.pop(bid)
            done.append(j)
        return done
    def is_ready(self, bid): return bid in self.gpu

def simulate_access_pattern(accesses, gc, cc):
    off=KVOffloader(gc,cc)
    for b in set(accesses): off.cpu[b]=f"data{b}"
    for b in accesses: off.access(b)
    return off.stats()

# 实践1
off=KVOffloader(3,5)
for i in range(5): off.store(i,f"data{i}")
assert len(off.gpu)==3 and set(off.gpu.keys())=={2,3,4}
assert 0 in off.cpu and 1 in off.cpu
assert off.access(4)=="data4"
assert off.access(0)=="data0" and 0 in off.gpu and 2 not in off.gpu
s=off.stats(); assert s["gpu_hits"]==1 and s["cpu_hits"]==1
print("实践1 通过")

# 实践2
off=AsyncKVOffloader(2,4)
for i in range(4): off.store(i,f"d{i}")
job=off.submit_load(0); assert not off.is_ready(0)
done=off.poll(); assert job in done and off.is_ready(0)
j1=off.submit_load(1); j2=off.submit_store(3)
done=off.poll(); assert j1 in done and j2 in done
print("实践2 通过 (异步)")

# 实践3
g=KVOffloader(3,8)
sg=simulate_access_pattern([0,1,2]*10,3,8)
sb=simulate_access_pattern([0,4,1,5,2,6,3,7]*4,3,8)
print(f"  局部性好 GPU命中率 {sg['gpu_hit_rate']:.0%}")
print(f"  局部性差 GPU命中率 {sb['gpu_hit_rate']:.0%}")
assert sg["gpu_hit_rate"]>sb["gpu_hit_rate"],f"{sg['gpu_hit_rate']} vs {sb['gpu_hit_rate']}"
print("实践3 通过 (局部性好命中率更高)")
print("\n全部验证通过")
