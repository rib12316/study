"""验证参考答案。仅标准库。"""
import math, random, statistics

NF4_QUANTILES = [
    -1.0, -0.6962, -0.5251, -0.4391, -0.3439, -0.2520, -0.1630, -0.0842,
     0.0842,  0.1630,  0.2520,  0.3439,  0.4391,  0.5251,  0.6962,  1.0,
]

def uniform_quantize_4bit(values):
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0: max_abs = 1.0
    points = [-1.0 + 2.0*i/15 for i in range(16)]
    codes = []
    for v in values:
        nv = v/max_abs
        best = min(range(16), key=lambda i: abs(points[i]-nv))
        codes.append(best)
    return codes, max_abs

def uniform_dequantize_4bit(codes, scale):
    points = [-1.0 + 2.0*i/15 for i in range(16)]
    return [points[c]*scale for c in codes]

def nf4_quantize(values):
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0: max_abs = 1.0
    codes = []
    for v in values:
        nv = v/max_abs
        best = min(range(16), key=lambda i: abs(NF4_QUANTILES[i]-nv))
        codes.append(best)
    return codes, max_abs

def nf4_dequantize(codes, scale):
    return [NF4_QUANTILES[c]*scale for c in codes]

def mse(a,b): return sum((x-y)**2 for x,y in zip(a,b))/len(a)

def compare_on_distribution(values, dist_name):
    uc,us = uniform_quantize_4bit(values); ud = uniform_dequantize_4bit(uc,us)
    nc,ns = nf4_quantize(values); nd = nf4_dequantize(nc,ns)
    return {"dist":dist_name,"uniform_mse":mse(values,ud),"nf4_mse":mse(values,nd)}

# 实践1
values=[0.1,-0.3,0.5,-0.8,0.0]
codes,scale=uniform_quantize_4bit(values)
assert len(codes)==5 and all(0<=c<=15 for c in codes)
assert scale>0
deq=uniform_dequantize_4bit(codes,scale); err=mse(values,deq)
assert err>0
print(f"实践1 通过: scale={scale:.3f} MSE={err:.5f}")

# 实践2
random.seed(0)
nw=[random.gauss(0,0.3) for _ in range(10000)]
r=compare_on_distribution(nw,"正态")
print(f"  正态: 均匀={r['uniform_mse']:.6f} NF4={r['nf4_mse']:.6f}")
assert r["nf4_mse"]<r["uniform_mse"],f"NF4应更优: {r}"
print(f"  NF4 比均匀误差小 {r['uniform_mse']/r['nf4_mse']:.2f}x")
print("实践2 通过 (正态分布下NF4更优)")

# 实践3
random.seed(1)
uw=[random.uniform(-0.5,0.5) for _ in range(10000)]
ru=compare_on_distribution(uw,"均匀")
print(f"  均匀分布权重: 均匀={ru['uniform_mse']:.6f} NF4={ru['nf4_mse']:.6f}")
random.seed(0)
nw2=[random.gauss(0,0.3) for _ in range(10000)]
rn=compare_on_distribution(nw2,"正态")
na=rn["uniform_mse"]/rn["nf4_mse"]; ua=ru["uniform_mse"]/ru["nf4_mse"]
assert na>ua, f"正态优势{na:.2f}应>均匀优势{ua:.2f}"
print(f"  正态优势 {na:.2f}x > 均匀分布优势 {ua:.2f}x")
print("实践3 通过 (分布不匹配优势消失)")
print("\n全部验证通过")
