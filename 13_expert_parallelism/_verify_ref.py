"""验证参考答案。仅标准库。"""
def make_expert_map(gne, ep_size, ep_rank, strategy="contiguous"):
    em = [-1]*gne
    base = gne // ep_size; rem = gne % ep_size
    if strategy=="contiguous":
        ln = base+1 if ep_rank<rem else base
        start = ep_rank*base + min(ep_rank, rem)
        for li in range(ln): em[start+li] = li
    elif strategy=="round_robin":
        li = 0
        for g in range(ep_rank, gne, ep_size):
            em[g] = li; li += 1
    return em

def global_to_rank(gid, ep_size, strategy, gne):
    for r in range(ep_size):
        if make_expert_map(gne, ep_size, r, strategy)[gid] != -1:
            return r
    return -1

def dispatch(tokens, topk_ids, workers, ep_size, strategy, gne):
    rm = {}
    for t, hidden in enumerate(tokens):
        for gid in topk_ids[t]:
            r = global_to_rank(gid, ep_size, strategy, gne)
            rm.setdefault(r,{}).setdefault(t, (hidden, []))
            rm[r][t][1].append(gid)
    return {r: [(ti,h,ei) for ti,(h,ei) in tm.items()] for r,tm in rm.items()}

def compute_local(worker, dispatched, topk_weights, topk_ids):
    res = {}
    for ti, hidden, eids in dispatched:
        res[ti] = {}
        for gid in eids:
            res[ti][gid] = worker.expert_weights[worker.local_id(gid)](hidden)
    return res

def combine(tc, all_res, topk_weights, topk_ids, out_dim):
    out = [[0.0]*out_dim for _ in range(tc)]
    for t in range(tc):
        for slot, gid in enumerate(topk_ids[t]):
            w = topk_weights[t][slot]
            for rank, res in all_res.items():
                if t in res and gid in res[t]:
                    for i in range(out_dim): out[t][i] += w*res[t][gid][i]
                    break
    return out

def measure_load_balance(tokens, topk_ids, ep_size, strategy, gne):
    load = {r:0 for r in range(ep_size)}
    for t in range(len(tokens)):
        vr = set()
        for gid in topk_ids[t]:
            vr.add(global_to_rank(gid, ep_size, strategy, gne))
        for r in vr: load[r] += 1
    return load

class EPWorker:
    def __init__(self, rank, em, ew): self.rank=rank; self.expert_map=em; self.expert_weights=ew
    def is_local(self, gid): return self.expert_map[gid] != -1
    def local_id(self, gid): return self.expert_map[gid]

# 实践1
em0c = make_expert_map(13,4,0,"contiguous")
em0r = make_expert_map(13,4,0,"round_robin")
assert em0c[:4]==[0,1,2,3] and em0c[4]==-1
nn = [i for i in range(13) if em0r[i]!=-1]
assert nn==[0,4,8,12], f"{nn}"
tc = sum(1 for r in range(4) for x in make_expert_map(13,4,r,"contiguous") if x!=-1)
tr = sum(1 for r in range(4) for x in make_expert_map(13,4,r,"round_robin") if x!=-1)
assert tc==13 and tr==13
print("实践1 通过")

# 实践2
ep,gne,od=2,4,2; strat="contiguous"
em0=make_expert_map(gne,ep,0,strat); em1=make_expert_map(gne,ep,1,strat)
def mkew(gid): return lambda x:[x[0]*(gid+1), x[1]*(gid+1)]
w0={0:mkew(0),1:mkew(1)}; w1={0:mkew(2),1:mkew(3)}
workers=[EPWorker(0,em0,w0),EPWorker(1,em1,w1)]
tokens=[[1.0,2.0],[3.0,4.0]]
topk_ids=[[0,2],[1,3]]; topk_w=[[0.6,0.4],[0.5,0.5]]
disp=dispatch(tokens,topk_ids,workers,ep,strat,gne)
assert 0 in disp and 1 in disp
r0t=[t[0] for t in disp[0]]
assert 0 in r0t and 1 in r0t, f"卡0应收tok0,1: {r0t}"
allr={}
for w in workers: allr[w.rank]=compute_local(w,disp.get(w.rank,[]),topk_w,topk_ids)
out=combine(len(tokens),allr,topk_w,topk_ids,od)
x0=tokens[0]
m0=[0.6*mkew(0)(x0)[i]+0.4*mkew(2)(x0)[i] for i in range(od)]
assert all(abs(a-b)<1e-6 for a,b in zip(out[0],m0)),f"{out[0]} vs {m0}"
print(f"实践2 通过: tok0={[round(x,3) for x in out[0]]}")

# 实践3
ep,gne=2,8
topk_ids=[[0,1]]*5+[[2,3]]*5
tokens=[[0.0]]*10
lc=measure_load_balance(tokens,topk_ids,ep,"contiguous",gne)
lr=measure_load_balance(tokens,topk_ids,ep,"round_robin",gne)
assert lc[0]==10 and lc[1]==0, f"contiguous: {lc}"
assert lr[0]>0 and lr[1]>0, f"round_robin: {lr}"
print(f"实践3 通过: contiguous {lc}, round_robin {lr}")
print("\n全部验证通过")
