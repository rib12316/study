"""验证参考答案。仅标准库。"""
import math, random

def gate(hidden, gw):
    return [sum(w*h for w,h in zip(row,hidden)) for row in gw]

def softmax(logits):
    m=max(logits); exps=[math.exp(x-m) for x in logits]; s=sum(exps)
    return [e/s for e in exps]

def topk_softmax(scores, k, renormalize=True):
    probs=softmax(scores)
    idx=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)[:k]
    w=[probs[i] for i in idx]
    if renormalize:
        s=sum(w); w=[x/s for x in w]
    return w, idx

def expert_ffn(x, ew, eb=None):
    out=[]
    for o in range(len(ew)):
        val=sum(ew[o][i]*x[i] for i in range(len(ew[o])))
        if eb is not None: val+=eb[o]
        out.append(val)
    return [max(0,v) for v in out]

def moe_forward_token(hidden, gw, experts_w, experts_b, k):
    scores=gate(hidden,gw); w,ids=topk_softmax(scores,k)
    out=[0.0]*len(experts_w[0])
    for wi,eid in zip(w,ids):
        eo=expert_ffn(hidden,experts_w[eid],experts_b[eid])
        out=[o+wi*e for o,e in zip(out,eo)]
    return out

def moe_naive(tokens, gw, experts_w, experts_b, k):
    ne=len(experts_w); od=len(experts_w[0])
    output=[[0.0]*od for _ in range(len(tokens))]; loop=0
    for e in range(ne):
        loop+=1
        for ti,token in enumerate(tokens):
            scores=gate(token,gw); w,ids=topk_softmax(scores,k)
            if e in ids:
                wi=w[ids.index(e)]; eo=expert_ffn(token,experts_w[e],experts_b[e])
                output[ti]=[o+wi*x for o,x in zip(output[ti],eo)]
    return output, loop

def moe_sorted(tokens, gw, experts_w, experts_b, k):
    od=len(experts_w[0]); pairs=[]
    for ti,token in enumerate(tokens):
        scores=gate(token,gw); w,ids=topk_softmax(scores,k)
        for wi,eid in zip(w,ids): pairs.append((eid,ti,wi,token))
    pairs.sort(key=lambda p:p[0])
    output=[[0.0]*od for _ in range(len(tokens))]; gc=0; i=0
    while i<len(pairs):
        ce=pairs[i][0]; gc+=1
        while i<len(pairs) and pairs[i][0]==ce:
            _,ti,wi,token=pairs[i]
            eo=expert_ffn(token,experts_w[ce],experts_b[ce])
            output[ti]=[o+wi*x for o,x in zip(output[ti],eo)]; i+=1
    return output, gc

# 实践1
gw=[[1,0,0],[0,1,0],[0,0,1],[1,1,0]]; hidden=[2.0,1.0,0.5]
scores=gate(hidden,gw)
assert len(scores)==4
sm=softmax([1,2,3,0.0]); assert abs(sum(sm)-1)<1e-6
w,ids=topk_softmax(scores,2,True)
assert len(w)==2 and len(ids)==2 and abs(sum(w)-1)<1e-6
si=sorted(range(4),key=lambda i:scores[i],reverse=True)[:2]
assert set(ids)==set(si),f"{ids} vs {si}"
print("实践1 通过")

# 实践2
random.seed(0)
ne,hd,od=4,3,2
gw=[[random.gauss(0,1) for _ in range(hd)] for _ in range(ne)]
ew=[[[random.gauss(0,1) for _ in range(hd)] for _ in range(od)] for _ in range(ne)]
eb=[[0.0]*od]*ne
hidden=[1.0,0.5,-0.3]
out=moe_forward_token(hidden,gw,ew,eb,2)
assert len(out)==od
sc=gate(hidden,gw); w,ids=topk_softmax(sc,2)
manual=[0.0]*od
for wi,ei in zip(w,ids):
    eo=expert_ffn(hidden,ew[ei],eb[ei]); manual=[m+wi*e for m,e in zip(manual,eo)]
assert all(abs(a-b)<1e-6 for a,b in zip(out,manual)),f"{out} vs {manual}"
print("实践2 通过")

# 实践3
random.seed(42)
ne,hd,od,nt,k=4,3,2,5,2
gw=[[random.gauss(0,1) for _ in range(hd)] for _ in range(ne)]
ew=[[[random.gauss(0,1) for _ in range(hd)] for _ in range(od)] for _ in range(ne)]
eb=[[0.0]*od]*ne
tokens=[[random.gauss(0,1) for _ in range(hd)] for _ in range(nt)]
on,ln=moe_naive(tokens,gw,ew,eb,k)
os_,gs=moe_sorted(tokens,gw,ew,eb,k)
for i in range(nt):
    assert all(abs(a-b)<1e-6 for a,b in zip(on[i],os_[i])),f"token{i}: {on[i]} vs {os_[i]}"
print(f"实践3 通过: 朴素循环{ln}次, 排序分组{gs}组, 输出等价")
print("\n全部验证通过")
