"""验证参考答案。仅标准库。"""
import random

def matvec(W, x):
    return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]

def matvec_AB(A, B, x):
    tmp = [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]
    return [sum(B[i][j]*tmp[j] for j in range(len(tmp))) for i in range(len(B))]

def lora_forward(x, W_base, A, B, scaling):
    base = matvec(W_base, x)
    lora = matvec_AB(A, B, x)
    return [base[i] + scaling*lora[i] for i in range(len(base))]

def lora_mixed_batch(tokens, W_base, adapters_A, adapters_B, adapter_indices, scaling):
    output = []
    for t, x in enumerate(tokens):
        idx = adapter_indices[t]
        base = matvec(W_base, x)
        lora = matvec_AB(adapters_A[idx], adapters_B[idx], x)
        output.append([base[i] + scaling*lora[i] for i in range(len(base))])
    return output

class LoRAManager:
    def __init__(self, W_base, max_adapters, scaling=1.0):
        self.W_base=W_base; self.max_adapters=max_adapters; self.scaling=scaling
        self.id_to_index={}; self.slots=[None]*max_adapters; self.ref_cnt=[0]*max_adapters
    def load_adapter(self, adapter_id, A, B):
        if adapter_id in self.id_to_index: return self.id_to_index[adapter_id]
        for i in range(self.max_adapters):
            if self.slots[i] is None:
                self.slots[i]=(A,B); self.id_to_index[adapter_id]=i; return i
        raise RuntimeError("No free slot")
    def unload_adapter(self, adapter_id):
        if adapter_id not in self.id_to_index: return
        idx=self.id_to_index.pop(adapter_id)
        if self.ref_cnt[idx]==0: self.slots[idx]=None
    def serve(self, tokens, adapter_ids):
        indices=[self.id_to_index[aid] for aid in adapter_ids]
        aA=[self.slots[i][0] if self.slots[i] else None for i in range(self.max_adapters)]
        aB=[self.slots[i][1] if self.slots[i] else None for i in range(self.max_adapters)]
        return lora_mixed_batch(tokens, self.W_base, aA, aB, indices, self.scaling)

# 实践1
random.seed(0)
in_d,out_d,r=3,4,2
W=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(out_d)]
A=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(r)]
B=[[random.gauss(0,1) for _ in range(r)] for _ in range(out_d)]
x=[1.0,0.5,-0.3]; scaling=0.5
y0=lora_forward(x,W,A,B,0.0); base=matvec(W,x)
assert all(abs(a-b)<1e-6 for a,b in zip(y0,base))
y=lora_forward(x,W,A,B,scaling)
manual=[base[i]+scaling*matvec_AB(A,B,x)[i] for i in range(out_d)]
assert all(abs(a-b)<1e-6 for a,b in zip(y,manual)),f"{y} vs {manual}"
print("实践1 通过")

# 实践2
random.seed(42)
in_d,out_d,r=3,2,2
W=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(out_d)]
A0=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(r)]
B0=[[random.gauss(0,1) for _ in range(r)] for _ in range(out_d)]
A1=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(r)]
B1=[[random.gauss(0,1) for _ in range(r)] for _ in range(out_d)]
tokens=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(4)]
indices=[0,0,1,1]
out=lora_mixed_batch(tokens,W,[A0,A1],[B0,B1],indices,1.0)
assert len(out)==4
for t in range(4):
    idx=indices[t]; A=[A0,A1][idx]; B=[B0,B1][idx]
    m=[matvec(W,tokens[t])[i]+matvec_AB(A,B,tokens[t])[i] for i in range(out_d)]
    assert all(abs(a-b)<1e-6 for a,b in zip(out[t],m)),f"tok{t}"
assert out[0]!=out[2]
print("实践2 通过 (混合batch)")

# 实践3
random.seed(7)
in_d,out_d,r=3,2,2
W=[[random.gauss(0,1) for _ in range(in_d)] for _ in range(out_d)]
mgr=LoRAManager(W,2,1.0)
def mk():
    return ([[random.gauss(0,1) for _ in range(in_d)] for _ in range(r)],
            [[random.gauss(0,1) for _ in range(r)] for _ in range(out_d)])
A100,B100=mk(); i100=mgr.load_adapter(100,A100,B100); assert i100==0
A200,B200=mk(); i200=mgr.load_adapter(200,A200,B200); assert i200==1
A300,B300=mk()
try: mgr.load_adapter(300,A300,B300); assert False
except RuntimeError: pass
toks=[[1.0]*in_d,[0.5]*in_d]
out=mgr.serve(toks,[100,200])
m0=[matvec(W,toks[0])[i]+matvec_AB(A100,B100,toks[0])[i] for i in range(out_d)]
m1=[matvec(W,toks[1])[i]+matvec_AB(A200,B200,toks[1])[i] for i in range(out_d)]
assert all(abs(a-b)<1e-6 for a,b in zip(out[0],m0))
assert all(abs(a-b)<1e-6 for a,b in zip(out[1],m1))
mgr.unload_adapter(100)
i300=mgr.load_adapter(300,A300,B300); assert i300==0
assert 300 in mgr.id_to_index and 100 not in mgr.id_to_index
print("实践3 通过 (热加载生命周期)")
print("\n全部验证通过")
