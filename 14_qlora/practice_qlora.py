"""
特性 #14 干中学实践：mini QLoRA（量化 base + LoRA 交互 + 误差传播测量）

目标：亲手重建 QLoRA 核心并实测误差传播——
  ① 量化 base（均匀量化/反量化）+ LoRA 叠加前向
  ② 测量量化误差传播：全精度 vs 量化base无LoRA vs QLoRA
  ③ 量化粒度对比：per-tensor vs per-group（group_size 影响）

参考 vLLM 源码：
  - lora/layers/base_linear.py:186  _get_quant_method（量化对 LoRA 透明）
  - lora/layers/base_linear.py:207  quant_method.apply（base 量化前向）
  - lora/layers/base_linear.py:227  add_lora_linear（LoRA 叠加）

依赖：仅标准库（random, math）。不需要装 vllm/torch。
运行：python practice_qlora.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import math
import random


# ============================================================================
# 辅助：矩阵/向量运算
# ============================================================================
def matvec(W, x):
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

def matvec_AB(A, B, x):
    tmp = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    return [sum(B[i][j] * tmp[j] for j in range(len(tmp))) for i in range(len(B))]


# ============================================================================
# 实践 1：量化 base + LoRA 叠加
# ============================================================================
def quantize_w(W, bits=4, scale=None):
    """对称均匀量化。返回 (W_quant[int], scale)。
    W_quant = round(W / scale)，范围 [-2^(bits-1), 2^(bits-1)-1]
    scale = max(abs(W)) / (2^(bits-1) - 1)
    """
    # TODO(你): 实现 per-tensor 对称量化
    pass


def dequantize_w(W_quant, scale):
    """反量化：W_deq = W_quant * scale。"""
    # TODO(你): 实现
    pass


def qlora_forward(x, W_true, W_quant, scale, A, B, lora_scaling):
    """QLoRA 前向：y = dequant(W_quant) @ x + lora_scaling * B @ A @ x。
    参考 base_linear.py:204（量化 base + LoRA 叠加）。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 2：量化误差传播测量（核心）
# ============================================================================
def measure_error(vec_a, vec_b):
    """L2 范数误差。"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def compare_configs(W_true, W_quant, scale, A, B, lora_scaling, x, bits):
    """对比三种配置，返回误差 dict。
    - full: W_true@x + s*B@A@x（全精度 ground truth）
    - q_base_no_lora: dequant(W_q)@x（量化 base 无 LoRA）
    - qlora: dequant(W_q)@x + s*B@A@x（QLoRA）
    测量：
    - err_base: ‖q_base_no_lora - W_true@x‖
    - err_qlora: ‖qlora - full‖
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 3：量化粒度对比（per-tensor vs per-group）
# ============================================================================
def quantize_w_per_group(W, bits, group_size):
    """per-group 量化：每 group_size 列一组，各自 scale。
    W[out][in]，按 in 维度切 group。
    返回 (W_quant, scales) 其中 scales[g] 是第 g 组的 scale。
    """
    # TODO(你): 实现
    pass


def dequantize_w_per_group(W_quant, scales, group_size):
    """per-group 反量化。"""
    # TODO(你): 实现
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def make_random_matrix(rows, cols, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


def verify_practice1():
    print("=== 实践 1 验证：量化 base + LoRA ===")
    out_dim, in_dim, r = 4, 3, 2
    W = make_random_matrix(out_dim, in_dim, 0)
    A = make_random_matrix(r, in_dim, 1)
    B = make_random_matrix(out_dim, r, 2)
    x = [1.0, 0.5, -0.3]

    Wq, scale = quantize_w(W, bits=4)
    Wdeq = dequantize_w(Wq, scale)
    # 量化误差应存在但不大
    err = measure_error(matvec(W, x), matvec(Wdeq, x))
    assert err > 0, "量化应有误差"

    # QLoRA 前向
    y = qlora_forward(x, W, Wq, scale, A, B, lora_scaling=0.5)
    # 手算
    manual = [matvec(Wdeq, x)[i] + 0.5 * matvec_AB(A, B, x)[i] for i in range(out_dim)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(y, manual)), f"{y} vs {manual}"
    print(f"  bits=4 量化 base 误差: {err:.4f}")
    print(f"  QLoRA 前向与手算一致 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：误差传播 ===")
    out_dim, in_dim, r = 4, 3, 2
    W = make_random_matrix(out_dim, in_dim, 0)
    A = make_random_matrix(r, in_dim, 1)
    B = make_random_matrix(out_dim, r, 2)
    x = [1.0, 0.5, -0.3]

    for bits in [4, 8, 16]:
        Wq, scale = quantize_w(W, bits=bits)
        errs = compare_configs(W, Wq, scale, A, B, 0.5, x, bits)
        print(f"  bits={bits}: err_base={errs['err_base']:.4f}, "
              f"err_qlora={errs['err_qlora']:.4f}")
        # err_qlora ≈ err_base（LoRA 叠加不放大 base 误差）
        assert abs(errs['err_qlora'] - errs['err_base']) < 1e-4, \
            f"err_qlora({errs['err_qlora']}) 应≈err_base({errs['err_base']})"
        # bits 越高误差越小
    # bits=16 误差应 << bits=4
    Wq4, s4 = quantize_w(W, 4)
    Wq16, s16 = quantize_w(W, 16)
    e4 = compare_configs(W, Wq4, s4, A, B, 0.5, x, 4)
    e16 = compare_configs(W, Wq16, s16, A, B, 0.5, x, 16)
    assert e16['err_base'] < e4['err_base'], "bits=16 误差应更小"
    print(f"  → err_qlora ≈ err_base（LoRA 不放大 base 误差）✓")
    print(f"  → bits 越高误差越小 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：量化粒度对比 ===")
    out_dim, in_dim = 8, 16   # in_dim=16 方便分组
    W = make_random_matrix(out_dim, in_dim, 0)
    x = [random.Random(5).gauss(0, 1) for _ in range(in_dim)]

    # per-tensor
    Wq_t, st = quantize_w(W, bits=4)
    Wdeq_t = dequantize_w(Wq_t, st)
    err_t = measure_error(matvec(W, x), matvec(Wdeq_t, x))

    # per-group group_size=4
    Wq_g4, sg4 = quantize_w_per_group(W, bits=4, group_size=4)
    Wdeq_g4 = dequantize_w_per_group(Wq_g4, sg4, 4)
    err_g4 = measure_error(matvec(W, x), matvec(Wdeq_g4, x))

    # per-group group_size=8
    Wq_g8, sg8 = quantize_w_per_group(W, bits=4, group_size=8)
    Wdeq_g8 = dequantize_w_per_group(Wq_g8, sg8, 8)
    err_g8 = measure_error(matvec(W, x), matvec(Wdeq_g8, x))

    print(f"  per-tensor (1组):     err={err_t:.4f}")
    print(f"  per-group (g=8, 2组): err={err_g8:.4f}")
    print(f"  per-group (g=4, 4组): err={err_g4:.4f}")
    assert err_g4 < err_g8 < err_t, \
        f"per-group 应优于 per-tensor，g越小越好: {err_t}/{err_g8}/{err_g4}"
    print(f"  → group 越小（组越多）量化误差越小，NF4 用 g=64 的依据 ✓\n")


def main():
    print("mini QLoRA 误差传播实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 QLoRA 并实测量化误差传播。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def quantize_w(W, bits=4, scale=None):
    if scale is None:
        max_abs = max(abs(W[i][j]) for i in range(len(W)) for j in range(len(W[0])))
        scale = max_abs / (2**(bits-1) - 1) if max_abs > 0 else 1.0
    Wq = [[round(W[i][j] / scale) for j in range(len(W[0]))] for i in range(len(W))]
    return Wq, scale


def dequantize_w(W_quant, scale):
    return [[W_quant[i][j] * scale for j in range(len(W_quant[0]))] for i in range(len(W_quant))]


def qlora_forward(x, W_true, W_quant, scale, A, B, lora_scaling):
    Wdeq = dequantize_w(W_quant, scale)
    base = matvec(Wdeq, x)
    lora = matvec_AB(A, B, x)
    return [base[i] + lora_scaling * lora[i] for i in range(len(base))]


def measure_error(vec_a, vec_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def compare_configs(W_true, W_quant, scale, A, B, lora_scaling, x, bits):
    Wdeq = dequantize_w(W_quant, scale)
    y_full = [matvec(W_true, x)[i] + lora_scaling * matvec_AB(A, B, x)[i]
              for i in range(len(W_true))]
    y_q_base = matvec(Wdeq, x)
    y_qlora = [y_q_base[i] + lora_scaling * matvec_AB(A, B, x)[i]
               for i in range(len(W_true))]
    err_base = measure_error(y_q_base, matvec(W_true, x))
    err_qlora = measure_error(y_qlora, y_full)
    return {"err_base": err_base, "err_qlora": err_qlora}


def quantize_w_per_group(W, bits, group_size):
    out_dim, in_dim = len(W), len(W[0])
    num_groups = (in_dim + group_size - 1) // group_size
    Wq = [[0] * in_dim for _ in range(out_dim)]
    scales = []
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_dim)
        # 取这组的 max abs
        max_abs = max(abs(W[i][j]) for i in range(out_dim) for j in range(start, end))
        s = max_abs / (2**(bits-1) - 1) if max_abs > 0 else 1.0
        scales.append(s)
        for i in range(out_dim):
            for j in range(start, end):
                Wq[i][j] = round(W[i][j] / s)
    return Wq, scales


def dequantize_w_per_group(W_quant, scales, group_size):
    out_dim, in_dim = len(W_quant), len(W_quant[0])
    Wdeq = [[0.0] * in_dim for _ in range(out_dim)]
    for g, s in enumerate(scales):
        start = g * group_size
        end = min(start + group_size, in_dim)
        for i in range(out_dim):
            for j in range(start, end):
                Wdeq[i][j] = W_quant[i][j] * s
    return Wdeq
"""
