"""
特性 #16 干中学实践：NF4 量化 vs 均匀 INT4 对比

目标：亲手实现 NF4 和均匀 INT4，实测"为什么正态分布权重下 NF4 精度更高"——
  ① 均匀 INT4 量化（等间距 16 个点）
  ② NF4 量化（正态分位数 16 个点）
  ③ 正态分布权重下对比 MSE（NF4 应更优）
  ④ 均匀分布权重下对比（NF4 优势消失，验证数据感知的前提）

参考：
  - QLoRA 论文（Dettmers 2023）NF4 量化点常量
  - vllm/model_executor/model_loader/bitsandbytes_loader.py:433（vLLM 调 quantize_4bit nf4）
  - 第14讲 group_size 实践

依赖：仅标准库（math, random, statistics）。不需要装 vllm/bitsandbytes。
运行：python practice_nf4.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import math
import random
import statistics


# ============================================================================
# NF4 的 16 个量化点（QLoRA 论文固定常量，归一化到 [-1,1]）
# ============================================================================
NF4_QUANTILES = [
    -1.0, -0.6962, -0.5251, -0.4391, -0.3439, -0.2520, -0.1630, -0.0842,
     0.0842,  0.1630,  0.2520,  0.3439,  0.4391,  0.5251,  0.6962,  1.0,
]


# ============================================================================
# 实践 1：均匀 INT4 量化
# ============================================================================
def uniform_quantize_4bit(values: list[float]) -> tuple[list[int], float]:
    """均匀 INT4 量化。
    1. 找 max_abs，归一化到 [-1,1]
    2. 均匀摆 16 个点（等间距，从 -1 到 1）
    3. 每个值找最近点，存编号 0-15
    返回 (codes, scale)，scale = max_abs。
    """
    # TODO(你): 实现
    pass


def uniform_dequantize_4bit(codes: list[int], scale: float) -> list[float]:
    """均匀 INT4 反量化。"""
    # TODO(你): 实现
    # 重建 16 个均匀点，codes[i] → 点值 × scale
    pass


# ============================================================================
# 实践 2：NF4 量化 + 对比（核心）
# ============================================================================
def nf4_quantize(values: list[float]) -> tuple[list[int], float]:
    """NF4 量化。
    1. 找 max_abs，归一化到 [-1,1]
    2. 用 NF4_QUANTILES 的 16 个点
    3. 每个值找最近 NF4 点，存编号 0-15
    返回 (codes, scale)。
    """
    # TODO(你): 实现
    pass


def nf4_dequantize(codes: list[int], scale: float) -> list[float]:
    """NF4 反量化。"""
    # TODO(你): 实现
    pass


def mse(a: list[float], b: list[float]) -> float:
    """均方误差。"""
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


# ============================================================================
# 实践 3：分布不匹配（均匀分布权重）
# ============================================================================
def compare_on_distribution(values: list[float], dist_name: str) -> dict:
    """对一组权重，对比均匀 INT4 和 NF4 的量化 MSE。"""
    u_codes, u_scale = uniform_quantize_4bit(values)
    u_deq = uniform_dequantize_4bit(u_codes, u_scale)
    n_codes, n_scale = nf4_quantize(values)
    n_deq = nf4_dequantize(n_codes, n_scale)
    u_mse = mse(values, u_deq)
    n_mse = mse(values, n_deq)
    return {"dist": dist_name, "uniform_mse": u_mse, "nf4_mse": n_mse}


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：均匀 INT4 ===")
    values = [0.1, -0.3, 0.5, -0.8, 0.0]
    codes, scale = uniform_quantize_4bit(values)
    assert len(codes) == 5 and all(0 <= c <= 15 for c in codes), f"codes 非法: {codes}"
    assert scale > 0, f"scale 应正: {scale}"
    deq = uniform_dequantize_4bit(codes, scale)
    err = mse(values, deq)
    assert err > 0, "量化应有误差"
    # 检查量化点是否等间距（均匀）
    max_abs = max(abs(v) for v in values)
    normalized = [v / max_abs for v in values]
    # 均匀点间距应相等
    print(f"  values={values}")
    print(f"  scale={scale:.3f}, codes={codes}, deq={[round(x,3) for x in deq]}")
    print(f"  MSE={err:.5f} (量化误差存在) ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：NF4 vs 均匀（正态分布权重）===")
    random.seed(0)
    # 10000 个正态分布权重（模拟 LLM 权重）
    normal_weights = [random.gauss(0, 0.3) for _ in range(10000)]
    result = compare_on_distribution(normal_weights, "正态分布 N(0,0.3)")
    print(f"  {result['dist']}:")
    print(f"    均匀 INT4 MSE = {result['uniform_mse']:.6f}")
    print(f"    NF4       MSE = {result['nf4_mse']:.6f}")
    # 核心断言：正态分布下 NF4 应比均匀准（MSE 更小）
    assert result["nf4_mse"] < result["uniform_mse"], \
        f"正态分布下 NF4 应更优: NF4={result['nf4_mse']} vs 均匀={result['uniform_mse']}"
    ratio = result["uniform_mse"] / result["nf4_mse"]
    print(f"    NF4 比均匀误差小 {ratio:.2f}x ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：分布不匹配（均匀分布权重）===")
    random.seed(1)
    # 10000 个均匀分布权重（不是正态）
    uniform_weights = [random.uniform(-0.5, 0.5) for _ in range(10000)]
    result = compare_on_distribution(uniform_weights, "均匀分布 U(-0.5,0.5)")
    print(f"  {result['dist']}:")
    print(f"    均匀 INT4 MSE = {result['uniform_mse']:.6f}")
    print(f"    NF4       MSE = {result['nf4_mse']:.6f}")
    # 核心断言：均匀分布下 NF4 优势消失（甚至变差）
    # 不要求 NF4 一定更差，但差距应远小于正态分布的情况
    ratio = result["nf4_mse"] / result["uniform_mse"]
    print(f"    NF4/均匀 = {ratio:.2f} (接近1甚至>1，优势消失)")
    # 对比正态分布的倍数差距
    random.seed(0)
    normal_w = [random.gauss(0, 0.3) for _ in range(10000)]
    normal_r = compare_on_distribution(normal_w, "正态")
    normal_advantage = normal_r["uniform_mse"] / normal_r["nf4_mse"]
    uniform_advantage = result["uniform_mse"] / result["nf4_mse"]
    assert normal_advantage > uniform_advantage, \
        f"正态分布下NF4优势({normal_advantage:.2f}x)应大于均匀分布({uniform_advantage:.2f}x)"
    print(f"  → 正态分布: NF4 优势 {normal_advantage:.2f}x")
    print(f"  → 均匀分布: NF4 优势 {uniform_advantage:.2f}x (优势消失) ✓\n")


def main():
    print("NF4 量化 vs 均匀 INT4 对比实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手验证 NF4 在正态分布权重下的精度优势。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def uniform_quantize_4bit(values):
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0:
        max_abs = 1.0
    # 16 个均匀点，从 -1 到 1，等间距
    points = [-1.0 + 2.0 * i / 15 for i in range(16)]
    codes = []
    for v in values:
        nv = v / max_abs  # 归一化到 [-1,1]
        # 找最近点
        best = min(range(16), key=lambda i: abs(points[i] - nv))
        codes.append(best)
    return codes, max_abs


def uniform_dequantize_4bit(codes, scale):
    points = [-1.0 + 2.0 * i / 15 for i in range(16)]
    return [points[c] * scale for c in codes]


def nf4_quantize(values):
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0:
        max_abs = 1.0
    codes = []
    for v in values:
        nv = v / max_abs
        best = min(range(16), key=lambda i: abs(NF4_QUANTILES[i] - nv))
        codes.append(best)
    return codes, max_abs


def nf4_dequantize(codes, scale):
    return [NF4_QUANTILES[c] * scale for c in codes]


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def compare_on_distribution(values, dist_name):
    u_codes, u_scale = uniform_quantize_4bit(values)
    u_deq = uniform_dequantize_4bit(u_codes, u_scale)
    n_codes, n_scale = nf4_quantize(values)
    n_deq = nf4_dequantize(n_codes, n_scale)
    return {"dist": dist_name,
            "uniform_mse": mse(values, u_deq),
            "nf4_mse": mse(values, n_deq)}
"""
