"""
特性 #10 干中学实践：mini MoE（专家路由 + 稀疏激活 + 加权融合）

目标：亲手重建 MoE 的算法逻辑——
  ① gate 线性层打分 + topk_softmax 路由（softmax 后选 top-k）
  ② MoE 前向：每个 token 选 k 个专家、算 FFN、加权融合
  ③ 朴素逐专家 vs 按专家排序批量（体会 fused MoE 为什么快）

参考 vLLM 源码：
  - router/fused_topk_router.py:69  fused_topk（softmax/sigmoid routing）
  - fused_moe.py:1460              _prepare_expert_assignment（token 按专家排序）
  - layer.py                       FusedMoE 层（gate + experts）

依赖：仅标准库（math）。不需要装 vllm/torch。
运行：python practice_moe.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import math


# ============================================================================
# 实践 1：Gate 打分 + Top-k Softmax 路由
# ============================================================================
def gate(hidden: list[float], gate_weights: list[list[float]]) -> list[float]:
    """Gate 线性层打分。hidden=[hidden_dim], gate_weights=[num_experts][hidden_dim]。
    返回 [num_experts] 个分数（logits）。
    参考：layer.py 的 gate 是一个 Linear(hidden_dim, num_experts)。
    """
    # TODO(你): 实现（矩阵向量乘）
    pass


def softmax(logits: list[float]) -> list[float]:
    """标准 softmax。"""
    # TODO(你): 实现
    pass


def topk_softmax(scores: list[float], k: int,
                 renormalize: bool = True) -> tuple[list[float], list[int]]:
    """softmax 后选 top-k。返回 (topk权重, topk专家ID)。
    参考 fused_topk_router.py:69 + vllm_topk_softmax。
    算法：先对 scores softmax 得概率 → 选最大的 k 个 → (可选)权重归一化。
    注意：是先 softmax 再选 topk，不是选 topk 再 softmax。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 2：MoE 前向（核心）
# ============================================================================
def expert_ffn(x: list[float], expert_weights: list[list[float]],
               expert_bias: list[float] | None = None) -> list[float]:
    """模拟单个专家的 FFN：一个线性变换 + 简单激活。
    expert_weights=[out_dim, in_dim]。
    简化：用线性变换 + relu（真实 MoE 是 gate_up + silu + down，这里简化）。
    """
    # TODO(你): 实现（y = relu(W @ x + b)）
    pass


def moe_forward_token(hidden: list[float],
                      gate_weights: list[list[float]],
                      experts_w: list[list[list[float]]],
                      experts_b: list[list[float] | None],
                      k: int) -> list[float]:
    """单个 token 的 MoE 前向。
    1. gate 打分 → topk_softmax 选 k 个专家及权重
    2. 对这 k 个专家各算 FFN
    3. 按权重加权融合
    """
    # TODO(你): 实现
    pass


def moe_forward(tokens: list[list[float]],
                gate_weights: list[list[float]],
                experts_w: list[list[list[float]]],
                experts_b: list[list[float] | None],
                k: int) -> list[list[float]]:
    """批量 MoE 前向（朴素版：逐 token）。"""
    return [moe_forward_token(t, gate_weights, experts_w, experts_b, k) for t in tokens]


# ============================================================================
# 实践 3：朴素逐专家 vs 按专家排序批量
# ============================================================================
def moe_naive(tokens, gate_weights, experts_w, experts_b, k):
    """朴素版：遍历每个专家，收集分配给它的 token，逐专家算 FFN。
    返回 (output, expert_loop_count) —— expert_loop_count 是专家外循环次数。
    """
    # TODO(你): 实现
    # num_experts = len(experts_w)
    # 对每个专家 e：找出哪些 (token_idx, slot) 路由到了 e，算它们的 FFN，累加到输出
    pass


def moe_sorted(tokens, gate_weights, experts_w, experts_b, k):
    """按专家排序版（模拟 fused MoE）：先收集所有 (token, expert) 对并按 expert 排序，
    再按专家分组批量算。返回 (output, expert_group_count)。
    数学上和 moe_naive 完全等价，但模拟了"排序后批量"的思路。
    """
    # TODO(你): 实现
    # 1. 对所有 token 算 topk，得 (token_idx, expert_id, weight) 三元组列表
    # 2. 按 expert_id 排序
    # 3. 按连续相同 expert_id 分组，每组批量算 FFN
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：gate + topk_softmax ===")
    # 4 专家，hidden_dim=3
    gw = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]   # gate 权重
    hidden = [2.0, 1.0, 0.5]
    scores = gate(hidden, gw)
    assert len(scores) == 4, f"gate 应返回4个分数: {scores}"
    # softmax 和为 1
    sm = softmax([1.0, 2.0, 3.0, 0.0])
    assert abs(sum(sm) - 1.0) < 1e-6, f"softmax 和应为1: {sum(sm)}"
    # topk_softmax：k=2
    w, ids = topk_softmax(scores, k=2, renormalize=True)
    assert len(w) == 2 and len(ids) == 2
    assert abs(sum(w) - 1.0) < 1e-6, f"renormalize 后权重和应=1: {sum(w)}"
    # 选的应是分数最高的两个专家
    sorted_ids = sorted(range(4), key=lambda i: scores[i], reverse=True)[:2]
    assert set(ids) == set(sorted_ids), f"topk选错: {ids} vs top2 {sorted_ids}"
    print(f"  hidden={hidden}")
    print(f"  gate scores={[round(s,3) for s in scores]}")
    print(f"  topk(k=2): weights={[round(x,3) for x in w]}, experts={ids} ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：MoE 前向 ===")
    import random
    random.seed(0)
    num_experts, hidden_dim, out_dim = 4, 3, 2
    gw = [[random.gauss(0, 1) for _ in range(hidden_dim)] for _ in range(num_experts)]
    experts_w = [[[random.gauss(0, 1) for _ in range(hidden_dim)] for _ in range(out_dim)]
                 for _ in range(num_experts)]
    experts_b = [[0.0] * out_dim] * num_experts

    hidden = [1.0, 0.5, -0.3]
    out = moe_forward_token(hidden, gw, experts_w, experts_b, k=2)
    assert len(out) == out_dim, f"输出维度应={out_dim}: {len(out)}"
    # 手动验证：选 top2 专家，加权融合
    scores = gate(hidden, gw)
    w, ids = topk_softmax(scores, k=2)
    manual = [0.0] * out_dim
    for wi, ei in zip(w, ids):
        e_out = expert_ffn(hidden, experts_w[ei], experts_b[ei])
        manual = [m + wi * e for m, e in zip(manual, e_out)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, manual)), \
        f"MoE前向与手算不符: {out} vs {manual}"
    print(f"  single token MoE(k=2) 输出={[round(x,4) for x in out]}")
    print(f"  与手算加权融合一致 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：朴素 vs 排序 ===")
    import random
    random.seed(42)
    num_experts, hidden_dim, out_dim, num_tokens, k = 4, 3, 2, 5, 2
    gw = [[random.gauss(0, 1) for _ in range(hidden_dim)] for _ in range(num_experts)]
    experts_w = [[[random.gauss(0, 1) for _ in range(hidden_dim)] for _ in range(out_dim)]
                 for _ in range(num_experts)]
    experts_b = [[0.0] * out_dim] * num_experts
    tokens = [[random.gauss(0, 1) for _ in range(hidden_dim)] for _ in range(num_tokens)]

    out_naive, loop_naive = moe_naive(tokens, gw, experts_w, experts_b, k)
    out_sorted, group_sorted = moe_sorted(tokens, gw, experts_w, experts_b, k)

    # 数学等价
    for i in range(num_tokens):
        assert all(abs(a - b) < 1e-6 for a, b in zip(out_naive[i], out_sorted[i])), \
            f"token{i} 朴素与排序结果不符: {out_naive[i]} vs {out_sorted[i]}"
    # 朴素外循环 = num_experts（遍历每个专家）
    # 排序分组数 <= num_experts（只处理被选中的专家）
    print(f"  朴素版: 专家外循环 {loop_naive} 次（遍历全部 {num_experts} 专家）")
    print(f"  排序版: 专家分组 {group_sorted} 组（只处理被选中的专家）")
    print(f"  两者输出完全一致（数学等价）✓")
    print(f"  → 排序版跳过未选中专家，且同专家 token 聚集利于批量 ✓\n")


def main():
    print("mini MoE 专家路由实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 MoE 路由与稀疏专家计算核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def gate(hidden, gate_weights):
    return [sum(w * h for w, h in zip(row, hidden)) for row in gate_weights]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def topk_softmax(scores, k, renormalize=True):
    probs = softmax(scores)
    # 选 top-k：按概率降序取前 k 个的索引
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    weights = [probs[i] for i in idx]
    if renormalize:
        s = sum(weights)
        weights = [w / s for w in weights]
    return weights, idx


def expert_ffn(x, expert_weights, expert_bias=None):
    out_dim = len(expert_weights)
    in_dim = len(expert_weights[0])
    out = []
    for o in range(out_dim):
        val = sum(expert_weights[o][i] * x[i] for i in range(in_dim))
        if expert_bias is not None:
            val += expert_bias[o]
        out.append(val)
    return [max(0, v) for v in out]   # relu


def moe_forward_token(hidden, gate_weights, experts_w, experts_b, k):
    scores = gate(hidden, gate_weights)
    weights, ids = topk_softmax(scores, k)
    out_dim = len(experts_w[0])
    output = [0.0] * out_dim
    for w, eid in zip(weights, ids):
        e_out = expert_ffn(hidden, experts_w[eid], experts_b[eid])
        output = [o + w * e for o, e in zip(output, e_out)]
    return output


def moe_naive(tokens, gate_weights, experts_w, experts_b, k):
    num_experts = len(experts_w)
    out_dim = len(experts_w[0])
    output = [[0.0] * out_dim for _ in range(len(tokens))]
    expert_loop_count = 0
    # 朴素：外循环遍历每个专家
    for e in range(num_experts):
        expert_loop_count += 1
        for t_idx, token in enumerate(tokens):
            scores = gate(token, gate_weights)
            weights, ids = topk_softmax(scores, k)
            if e in ids:
                w = weights[ids.index(e)]
                e_out = expert_ffn(token, experts_w[e], experts_b[e])
                output[t_idx] = [o + w * x for o, x in zip(output[t_idx], e_out)]
    return output, expert_loop_count


def moe_sorted(tokens, gate_weights, experts_w, experts_b, k):
    out_dim = len(experts_w[0])
    # 1. 收集所有 (token_idx, expert_id, weight)
    pairs = []
    for t_idx, token in enumerate(tokens):
        scores = gate(token, gate_weights)
        weights, ids = topk_softmax(scores, k)
        for w, eid in zip(weights, ids):
            pairs.append((eid, t_idx, w, token))
    # 2. 按专家排序
    pairs.sort(key=lambda p: p[0])
    # 3. 按连续相同 expert 分组批量算
    output = [[0.0] * out_dim for _ in range(len(tokens))]
    group_count = 0
    i = 0
    while i < len(pairs):
        cur_expert = pairs[i][0]
        group_count += 1
        # 同专家的所有 (token, weight)
        while i < len(pairs) and pairs[i][0] == cur_expert:
            _, t_idx, w, token = pairs[i]
            e_out = expert_ffn(token, experts_w[cur_expert], experts_b[cur_expert])
            output[t_idx] = [o + w * x for o, x in zip(output[t_idx], e_out)]
            i += 1
    return output, group_count
"""
