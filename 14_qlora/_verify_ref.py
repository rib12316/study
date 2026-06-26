"""验证参考答案。仅标准库。"""
import math, random

def matvec(W,x): return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]
def matvec_AB(A,B,x):
    tmp=[sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]
    return [sum(B[i][j]*tmp[j] for j in range(len(tmp))) for i in range(len(B))]
def measure_error(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def quantize_w(W,bits=4,scale=None):
    if scale is None:
        ma=max(abs(W[i][j]) for i in range(len(W)) for j in range(len(W[0])))
        scale=ma/(2**(bits-1)-1) if ma>0 else 1.0
    return [[round(W[i][j]/scale) for j in range(len(W[0]))] for i in range(len(W))], scale

def dequantize_w(Wq,scale):
    return [[Wq[i][j]*scale for j in range(len(Wq[0]))] for i in range(len(Wq))]

def qlora_forward(x,W_true,Wq,scale,A,B,ls):
    Wd=dequantize_w(Wq,scale); base=matvec(Wd,x); lora=matvec_AB(A,B,x)
    return [base[i]+ls*lora[i] for i in range(len(base))]

def compare_configs(W_true,Wq,scale,A,B,ls,x,bits):
    Wd=dequantize_w(Wq,scale)
    yf=[matvec(W_true,x)[i]+ls*matvec_AB(A,B,x)[i] for i in range(len(W_true))]
    yqb=matvec(Wd,x)
    yq=[yqb[i]+ls*matvec_AB(A,B,x)[i] for i in range(len(W_true))]
    return {"err_base":measure_error(yqb,matvec(W_true,x)),"err_qlora":measure_error(yq,yf)}

def quantize_w_per_group(W,bits,gs):
    od,Id=len(W),len(W[0]); ng=(Id+gs-1)//gs; Wq=[[0]*Id for _ in range(od)]; scales=[]
    for g in range(ng):
        st=g*gs; ed=min(st+gs,Id)
        ma=max(abs(W[i][j]) for i in range(od) for j in range(st,ed))
        s=ma/(2**(bits-1)-1) if ma>0 else 1.0; scales.append(s)
        for i in range(od):
            for j in range(st,ed): Wq[i][j]=round(W[i][j]/s)
    return Wq,scales

def dequantize_w_per_group(Wq,scales,gs):
    od,Id=len(Wq),len(Wq[0]); Wd=[[0.0]*Id for _ in range(od)]
    for g,s in enumerate(scales):
        st=g*gs; ed=min(st+gs,Id)
        for i in range(od):
            for j in range(st,ed): Wd[i][j]=Wq[i][j]*s
    return Wd

def mk(r,c,s):
    rng=random.Random(s); return [[rng.gauss(0,1) for _ in range(c)] for _ in range(r)]

# 实践1
od,Id,r=4,3,2; W=mk(od,Id,0); A=mk(r,Id,1); B=mk(od,r,2); x=[1.0,0.5,-0.3]
Wq,scale=quantize_w(W,4); Wd=dequantize_w(Wq,scale)
err=measure_error(matvec(W,x),matvec(Wd,x)); assert err>0
y=qlora_forward(x,W,Wq,scale,A,B,0.5)
m=[matvec(Wd,x)[i]+0.5*matvec_AB(A,B,x)[i] for i in range(od)]
assert all(abs(a-b)<1e-6 for a,b in zip(y,m)),f"{y} vs {m}"
print(f"实践1 通过: 量化误差 {err:.4f}")

# 实践2
for bits in [4,8,16]:
    Wq,s=quantize_w(W,bits); e=compare_configs(W,Wq,s,A,B,0.5,x,bits)
    print(f"  bits={bits}: err_base={e['err_base']:.4f} err_qlora={e['err_qlora']:.4f}")
    assert abs(e['err_qlora']-e['err_base'])<1e-4, f"{e}"
Wq4,s4=quantize_w(W,4); Wq16,s16=quantize_w(W,16)
e4=compare_configs(W,Wq4,s4,A,B,0.5,x,4); e16=compare_configs(W,Wq16,s16,A,B,0.5,x,16)
assert e16['err_base']<e4['err_base']
print("实践2 通过 (err_qlora≈err_base, bits越高误差越小)")

# 实践3
od,Id=8,16; W=mk(od,Id,0); x=[random.Random(5).gauss(0,1) for _ in range(Id)]
Wqt,st=quantize_w(W,4); errt=measure_error(matvec(W,x),matvec(dequantize_w(Wqt,st),x))
Wqg4,sg4=quantize_w_per_group(W,4,4); errg4=measure_error(matvec(W,x),matvec(dequantize_w_per_group(Wqg4,sg4,4),x))
Wqg8,sg8=quantize_w_per_group(W,4,8); errg8=measure_error(matvec(W,x),matvec(dequantize_w_per_group(Wqg8,sg8,8),x))
print(f"  per-tensor:{errt:.4f} g=8:{errg8:.4f} g=4:{errg4:.4f}")
assert errg4<errg8<errt, f"{errt}/{errg8}/{errg4}"
print("实践3 通过 (group越小误差越小)")
print("\n全部验证通过")
