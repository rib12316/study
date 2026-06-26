"""
特性 #13 干中学实践：mini Expert Parallelism（专家分片 + dispatch/compute/combine）

目标：亲手重建 EP 的核心机制——
  ① expert_map 生成（contiguous / round_robin 两种放置）
  ② dispatch-compute-combine 三阶段（模拟多卡 token 路由）
  ③ 负载均衡观察（contiguous vs round_robin）

参考 vLLM 源码：
  - fused_moe/expert_map_manager.py:22   determine_expert_map（专家分片）
  - fused_moe/all2all_utils.py           dispatch/compute/combine（all2all 通信）
  - fused_moe/fused_moe.py:161           off_experts==-1 跳过非本地专家

简化：用纯 Python 模拟多卡（每"卡"是 dict），不涉及真实通信。
依赖：仅标准库。不需要装 vllm/torch。
运行：python practice_expert_parallelism.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations


# ============================================================================
# 实践 1：expert_map 生成
# ============================================================================
def make_expert_map(global_num_experts: int, ep_size: int, ep_rank: int,
                    strategy: str = "contiguous") -> list[int]:
    """生成 expert_map：[global_num_experts]，本卡专家填 local_id，其他填 -1。
    参考 expert_map_manager.py:22 determine_expert_map。
    
    strategy:
      - "contiguous": 专家 0..base-1 在卡0，base..2*base-1 在卡1...
      - "round_robin": 专家 0,ep_size,2*ep_size... 在卡0；1,1+ep_size... 在卡1...
    """
    # TODO(你): 实现
    # base = global_num_experts // ep_size; remainder = global_num_experts % ep_size
    # contiguous: 本卡负责 [rank*base+min(rank,remainder) : ...+local_num]
    # round_robin: 本卡负责 [rank, rank+ep_size, rank+2*ep_size, ...]
    pass


# 辅助：根据 expert_map 反查 global_id 属于哪个 rank
def global_to_rank(global_id: int, ep_size: int, strategy: str,
                   global_num_experts: int) -> int:
    """全局专家 ID 在哪个 rank。"""
    for rank in range(ep_size):
        em = make_expert_map(global_num_experts, ep_size, rank, strategy)
        if em[global_id] != -1:
            return rank
    return -1


# ============================================================================
# 实践 2：Dispatch-Compute-Combine
# ============================================================================
class EPWorker:
    """模拟一张卡，持有部分专家。"""
    def __init__(self, rank: int, expert_map: list[int], expert_weights: dict):
        self.rank = rank
        self.expert_map = expert_map   # 本卡的 expert_map
        # expert_weights: {local_id: weight_fn}，weight_fn(x) -> y（模拟专家 FFN）
        self.expert_weights = expert_weights

    def is_local(self, global_expert_id: int) -> bool:
        return self.expert_map[global_expert_id] != -1

    def local_id(self, global_expert_id: int) -> int:
        return self.expert_map[global_expert_id]


def dispatch(tokens: list[list[float]], topk_ids: list[list[int]],
             workers: list[EPWorker], ep_size: int, strategy: str,
             global_num_experts: int) -> dict[int, list[tuple[int, list[float], list[int]]]]:
    """Dispatch 阶段：按 topk_ids 把 token 分发到目标卡。
    返回 {rank: [(token_idx, hidden, expert_ids_on_this_rank), ...]}
    
    去重：一个 token 发往同一目标卡只发一次（即使它在那个卡有多个专家）。
    """
    # TODO(你): 实现
    # 对每个 token t，看它的 topk_ids[t]，对每个 global_expert_id：
    #   查它在哪个 rank → 把 (t, tokens[t], [该rank上的expert_ids]) 加入该 rank 的列表
    # 注意：同 token 同 rank 只发一次，但 expert_ids 要收集全
    pass


def compute_local(worker: EPWorker,
                  dispatched: list[tuple[int, list[float], list[int]]],
                  topk_weights: list[list[float]], topk_ids: list[list[int]]
                  ) -> dict[int, dict[int, list[float]]]:
    """Compute 阶段：本地算路由到本地专家的部分。
    返回 {token_idx: {global_expert_id: expert_output}}
    """
    # TODO(你): 实现
    # 对 dispatched 里每个 (token_idx, hidden, expert_ids)：
    #   对每个 global_expert_id：取 local_id，调 worker.expert_weights[local_id](hidden)
    pass


def combine(token_count: int,
            all_compute_results: dict[int, dict[int, dict[int, list[float]]]],
            topk_weights: list[list[float]], topk_ids: list[list[int]],
            out_dim: int) -> list[list[float]]:
    """Combine 阶段：汇总各卡结果，按 topk_weights 加权融合。
    all_compute_results: {rank: {token_idx: {global_expert_id: output}}}
    返回 [token_count][out_dim] 最终输出。
    """
    # TODO(你): 实现
    # 对每个 token：找它选的 k 个专家（topk_ids），从各卡结果里取对应 output，加权求和
    pass


# ============================================================================
# 实践 3：负载均衡
# ============================================================================
def measure_load_balance(tokens: list, topk_ids: list[list[int]],
                         ep_size: int, strategy: str,
                         global_num_experts: int) -> dict[int, int]:
    """统计每卡 dispatch 后收到的 token 数。返回 {rank: count}。"""
    # TODO(你): 实现
    # 对每个 token 的 topk_ids，查每个专家在哪个 rank，统计各 rank 被访问次数
    # 注意：一个 token 可能访问同 rank 多次（多个专家），但 dispatch 只算1次
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：expert_map ===")
    # 13 专家，EP=4
    em0_c = make_expert_map(13, 4, 0, "contiguous")
    em0_r = make_expert_map(13, 4, 0, "round_robin")
    # contiguous rank0: base=3, remainder=1, rank0 得4个 → [0,1,2,3,-1,...]
    assert em0_c[:4] == [0, 1, 2, 3], f"contiguous rank0 前4应非-1: {em0_c[:4]}"
    assert em0_c[4] == -1, f"contiguous rank0 第5应-1: {em0_c[4]}"
    # round_robin rank0: 0,4,8,12 → 这些位置非-1
    non_neg_r0 = [i for i in range(13) if em0_r[i] != -1]
    assert non_neg_r0 == [0, 4, 8, 12], f"round_robin rank0 应持 0,4,8,12: {non_neg_r0}"
    # 所有 rank 的专家数之和 == global
    total_c = sum(1 for r in range(4) for x in make_expert_map(13, 4, r, "contiguous") if x != -1)
    total_r = sum(1 for r in range(4) for x in make_expert_map(13, 4, r, "round_robin") if x != -1)
    assert total_c == 13 and total_r == 13, f"专家总数应=13: c={total_c} r={total_r}"
    print(f"  contiguous rank0: {[i for i,v in enumerate(em0_c) if v!=-1]}")
    print(f"  round_robin rank0: {non_neg_r0}")
    print(f"  两种策略各 rank 专家数之和都=13 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：dispatch/compute/combine ===")
    # EP=2, 4 专家, out_dim=2
    ep_size, gne, out_dim = 2, 4, 2
    strategy = "contiguous"
    # 构造 worker：每卡持2专家
    em0 = make_expert_map(gne, ep_size, 0, strategy)  # [0,1,-1,-1]
    em1 = make_expert_map(gne, ep_size, 1, strategy)  # [-1,-1,0,1]
    # 专家权重：用简单线性变换模拟 FFN，每专家不同（用 global_id 区分）
    def make_ew(gid):
        return lambda x: [x[0] * (gid + 1), x[1] * (gid + 1)]  # 缩放系数=gid+1
    w0 = {0: make_ew(0), 1: make_ew(1)}   # local 0,1 → global 0,1
    w1 = {0: make_ew(2), 1: make_ew(3)}   # local 0,1 → global 2,3
    workers = [EPWorker(0, em0, w0), EPWorker(1, em1, w1)]

    # 2 个 token，各选 2 个专家
    tokens = [[1.0, 2.0], [3.0, 4.0]]
    topk_ids = [[0, 2], [1, 3]]   # tok0→expert0(卡0)+expert2(卡1)；tok1→expert1(卡0)+expert3(卡1)
    topk_weights = [[0.6, 0.4], [0.5, 0.5]]

    dispatched = dispatch(tokens, topk_ids, workers, ep_size, strategy, gne)
    # tok0 应发往卡0(expert0)和卡1(expert2)
    assert 0 in dispatched and 1 in dispatched
    # 卡0 收到 tok0（expert0）和 tok1（expert1）
    rank0_tokens = [t[0] for t in dispatched[0]]
    assert 0 in rank0_tokens and 1 in rank0_tokens, f"卡0应收tok0,1: {rank0_tokens}"

    all_results = {}
    for w in workers:
        all_results[w.rank] = compute_local(w, dispatched.get(w.rank, []),
                                            topk_weights, topk_ids)
    out = combine(len(tokens), all_results, topk_weights, topk_ids, out_dim)

    # 手算验证 tok0: 0.6*expert0(x) + 0.4*expert2(x)
    x0 = tokens[0]
    manual0 = [0.6 * make_ew(0)(x0)[i] + 0.4 * make_ew(2)(x0)[i] for i in range(out_dim)]
    assert all(abs(a-b) < 1e-6 for a,b in zip(out[0], manual0)), \
        f"tok0 EP结果与手算不符: {out[0]} vs {manual0}"
    print(f"  EP=2, 4专家, 2 token 跨卡路由")
    print(f"  tok0 → expert0(卡0)+expert2(卡1), 加权融合 = {[round(x,3) for x in out[0]]}")
    print(f"  与手算 0.6*exp0 + 0.4*exp2 一致 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：负载均衡 ===")
    # 8 专家，EP=2，构造偏斜路由（前4专家高频被选）
    ep_size, gne = 2, 8
    # 10 个 token，都选 expert 0,1,2,3（都在卡0，contiguous）
    topk_ids = [[0, 1]] * 5 + [[2, 3]] * 5   # 全选卡0的专家
    tokens = [[0.0]] * 10
    load_c = measure_load_balance(tokens, topk_ids, ep_size, "contiguous", gne)
    load_r = measure_load_balance(tokens, topk_ids, ep_size, "round_robin", gne)
    # contiguous: 卡0持expert0-3，全压卡0
    assert load_c[0] == 10 and load_c[1] == 0, f"contiguous应全压卡0: {load_c}"
    # round_robin: expert0,2,4,6在卡0；1,3,5,7在卡1。topk=[[0,1]]→expert0(卡0)+expert1(卡1)
    # 所以 round_robin 下卡0卡1各收10（每token去两卡）
    assert load_r[0] > 0 and load_r[1] > 0, f"round_robin应分散: {load_r}"
    print(f"  偏斜路由(全选expert0-3):")
    print(f"    contiguous: {load_c} （全压卡0）")
    print(f"    round_robin: {load_r} （分散到两卡）")
    print(f"  → round_robin 负载更均衡 ✓\n")


def main():
    print("mini Expert Parallelism 实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 EP 专家分片 + dispatch/compute/combine。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def make_expert_map(global_num_experts, ep_size, ep_rank, strategy="contiguous"):
    em = [-1] * global_num_experts
    base = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    if strategy == "contiguous":
        local_num = base + 1 if ep_rank < remainder else base
        start = ep_rank * base + min(ep_rank, remainder)
        for local_id in range(local_num):
            em[start + local_id] = local_id
    elif strategy == "round_robin":
        local_idx = 0
        for g in range(ep_rank, global_num_experts, ep_size):
            em[g] = local_idx
            local_idx += 1
    return em


def dispatch(tokens, topk_ids, workers, ep_size, strategy, global_num_experts):
    # {rank: {token_idx: (hidden, [global_expert_ids on this rank])}}
    rank_map = {}
    for t, hidden in enumerate(tokens):
        for gid in topk_ids[t]:
            r = global_to_rank(gid, ep_size, strategy, global_num_experts)
            rank_map.setdefault(r, {}).setdefault(t, (hidden, []))
            rank_map[r][t][1].append(gid)
    # 转成 {rank: [(token_idx, hidden, expert_ids), ...]}
    result = {}
    for r, tm in rank_map.items():
        result[r] = [(tidx, h, eids) for tidx, (h, eids) in tm.items()]
    return result


def compute_local(worker, dispatched, topk_weights, topk_ids):
    results = {}  # {token_idx: {global_expert_id: output}}
    for token_idx, hidden, expert_ids in dispatched:
        results[token_idx] = {}
        for gid in expert_ids:
            lid = worker.local_id(gid)
            results[token_idx][gid] = worker.expert_weights[lid](hidden)
    return results


def combine(token_count, all_compute_results, topk_weights, topk_ids, out_dim):
    output = [[0.0] * out_dim for _ in range(token_count)]
    for t in range(token_count):
        for slot, gid in enumerate(topk_ids[t]):
            w = topk_weights[t][slot]
            # 找哪个 rank 算了这个 token 的这个 expert
            for rank, res in all_compute_results.items():
                if t in res and gid in res[t]:
                    for i in range(out_dim):
                        output[t][i] += w * res[t][gid][i]
                    break
    return output


def measure_load_balance(tokens, topk_ids, ep_size, strategy, global_num_experts):
    load = {r: 0 for r in range(ep_size)}
    for t in range(len(tokens)):
        visited_ranks = set()
        for gid in topk_ids[t]:
            r = global_to_rank(gid, ep_size, strategy, global_num_experts)
            visited_ranks.add(r)
        for r in visited_ranks:
            load[r] += 1
    return load
"""
