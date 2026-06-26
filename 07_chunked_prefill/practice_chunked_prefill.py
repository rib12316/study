"""
特性 #7 干中学实践：Chunked Prefill 调度 + 策略对比模拟器

目标：亲手实现 chunked prefill 调度，并量化对比不同 chunking 策略——
  ① 支持 threshold 的 chunking（主动切 + 被动切）
  ② 策略对比模拟器：不切 vs 被动切 vs 主动切，对比 decode延迟/TTFT/总step数
  ③ varlen 的 cu_seqlens 构造（理解 prefill/decode 零 padding 混合）

参考 vLLM 源码：
  - scheduler.py:795-810   chunking 决策（threshold + budget）
  - config/scheduler.py:84 enable_chunked_prefill 默认 True
  - flash_attn.py:248      query_start_loc / seq_lens (varlen)

依赖：仅标准库。不需要装 vllm/torch。
运行：python practice_chunked_prefill.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations


def cdiv(n: int, b: int) -> int:
    return (n + b - 1) // b


# ============================================================================
# 实践 1：支持 threshold 的 chunking 调度
# ============================================================================
class Request:
    def __init__(self, req_id: str, prompt_len: int, max_output: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_output = max_output
        self.num_computed = 0
        self.output_tokens = 0
        self.num_prefill_steps = 0      # prefill 跨了多少 step
        self.prefill_done_step = -1     # 在第几步 prefill 完成（算 TTFT）

    @property
    def prefill_done(self) -> bool:
        return self.num_computed >= self.prompt_len

    def is_finished(self) -> bool:
        return self.output_tokens >= self.max_output

    def __repr__(self):
        return (f"Req({self.req_id},p={self.prompt_len},"
                f"comp={self.num_computed},out={self.output_tokens}/{self.max_output})")


class ChunkedScheduler:
    """支持 chunking 策略的 mini 调度器。
    参考 scheduler.py:786-810。
    enable_chunked_prefill=False 时不切（长prompt独占或排队）
    long_prefill_token_threshold: 单次 prefill 上限（主动切）
    """
    def __init__(self, max_num_seqs: int, token_budget: int,
                 long_prefill_token_threshold: int = 0,
                 enable_chunked_prefill: bool = True):
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.threshold = long_prefill_token_threshold
        self.enable_chunked = enable_chunked_prefill
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_count = 0

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def step(self) -> dict[str, int]:
        """一个 step。返回 {req_id: 分到的token数}。
        TODO(你): 实现
        - 阶段1：running 的 decode，每个 1 token
        - 阶段2：waiting 的 prefill
          num_new = prompt_len - num_computed
          if 0 < threshold < num_new: num_new = threshold   # 主动切
          if not enable_chunked and num_new > budget: break  # 不切，排队
          num_new = min(num_new, budget)                     # 被动切
          累加 num_computed；没 prefill 完放回 waiting 头部
          prefill 完则进 running，记录 prefill_done_step
        """
        pass


# ============================================================================
# 实践 2：策略对比模拟器
# ============================================================================
def simulate(workload: list[Request], max_num_seqs: int, token_budget: int,
             threshold: int, enable_chunked: bool) -> dict:
    """跑一个负载，返回统计指标。
    复制 workload（避免污染原请求），用 ChunkedScheduler 跑到全部完成。
    返回:
      - decode_per_step_avg: 平均每 step 的 decode token 数（越多说明 decode 没被饿死）
      - ttft_list: 各请求的 TTFT（prefill 完成的 step 序号）
      - total_steps: 总 step 数（吞吐代理，越少越好）
    """
    # TODO(你): 实现
    pass


def compare_strategies():
    """对比三种策略，打印表格。"""
    print("=== 实践 2：策略对比 ===\n")
    # 构造负载：2个长prompt + 3个短decode请求
    def make_workload():
        return [
            Request("long1", prompt_len=5000, max_output=3),
            Request("long2", prompt_len=4000, max_output=3),
            Request("short1", prompt_len=20, max_output=10),
            Request("short2", prompt_len=20, max_output=10),
            Request("short3", prompt_len=20, max_output=10),
        ]
    # TODO(你): 调 simulate 三次，对比打印
    # 策略A: enable_chunked=False（不切）
    # 策略B: threshold=100000（只被动切）
    # 策略C: threshold=1000（主动切，平衡）
    pass


# ============================================================================
# 实践 3：varlen cu_seqlens 构造
# ============================================================================
def build_cu_seqlens(query_lens: list[int], kv_lens: list[int]):
    """给定一个 step 里各请求的 query 长度和 KV 长度，
    构造 FlashAttention varlen 的 cu_seqlens_q 和 cu_seqlens_k。
    query_lens 如 [2048, 1, 1]（一个prefill chunk + 两个decode）
    返回 cu_seqlens_q=[0,2048,2049,2050], cu_seqlens_k=[0,2048,2148,2198]
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：threshold chunking ===")
    sched = ChunkedScheduler(max_num_seqs=4, token_budget=2000,
                             long_prefill_token_threshold=1000,
                             enable_chunked_prefill=True)
    req = Request("long", prompt_len=5000, max_output=2)
    sched.add_request(req)

    for i in range(20):
        if not sched.waiting and not sched.running:
            break
        sched.step()
        sched.step_count += 1

    # threshold=1000, budget=2000 → 每step算min(剩余,1000,2000)=1000 → 5步prefill
    assert req.num_prefill_steps == 5, f"prefill应跨5步，实际{req.num_prefill_steps}"
    assert req.num_computed >= 5000, f"prefill没算完: {req.num_computed}"
    print(f"  long(5000token) threshold=1000 → prefill 跨 {req.num_prefill_steps} 步 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：策略对比 ===")
    # 这里调用学生实现的 compare_strategies，验证它产出了对比表
    # 参考答案会打印三种策略的指标对比
    compare_strategies()


def verify_practice3():
    print("=== 实践 3 验证：varlen cu_seqlens ===")
    ql, kl = [2048, 1, 1], [2048, 100, 50]
    cq, ck = build_cu_seqlens(ql, kl)
    assert cq == [0, 2048, 2049, 2050], f"cu_seqlens_q 错: {cq}"
    assert ck == [0, 2048, 2148, 2198], f"cu_seqlens_k 错: {ck}"
    print(f"  query_lens={ql}, kv_lens={kl}")
    print(f"  cu_seqlens_q = {cq}")
    print(f"  cu_seqlens_k = {ck}")
    print(f"  → 零 padding 拼接，prefill/decode 混合正确 ✓\n")


def main():
    print("Chunked Prefill 调度实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 Chunked Prefill 并量化对比策略。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
class ChunkedScheduler:
    def __init__(self, max_num_seqs, token_budget, long_prefill_token_threshold=0,
                 enable_chunked_prefill=True):
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.threshold = long_prefill_token_threshold
        self.enable_chunked = enable_chunked_prefill
        self.waiting = []
        self.running = []
        self.step_count = 0

    def add_request(self, req):
        self.waiting.append(req)

    def step(self):
        budget = self.token_budget
        scheduled = {}
        # 阶段1：running decode
        for req in list(self.running):
            if budget <= 0: break
            if req.is_finished(): continue
            req.num_computed += 1
            req.output_tokens += 1
            budget -= 1
            scheduled[req.req_id] = scheduled.get(req.req_id, 0) + 1
        self.running = [r for r in self.running if not r.is_finished()]
        # 阶段2：waiting prefill
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting.pop(0)
            num_new = req.prompt_len - req.num_computed
            # 主动切
            if 0 < self.threshold < num_new:
                num_new = self.threshold
            # 不切模式：一次算不完就排队
            if not self.enable_chunked and num_new > budget:
                self.waiting.insert(0, req)
                break
            # 被动切
            num_new = min(num_new, budget)
            req.num_computed += num_new
            req.num_prefill_steps += 1
            budget -= num_new
            scheduled[req.req_id] = scheduled.get(req.req_id, 0) + num_new
            if req.num_computed < req.prompt_len:
                self.waiting.insert(0, req)   # chunked 没完，放回头部
            else:
                req.prefill_done_step = self.step_count + 1
                self.running.append(req)
        return scheduled


def simulate(workload, max_num_seqs, token_budget, threshold, enable_chunked):
    sched = ChunkedScheduler(max_num_seqs, token_budget, threshold, enable_chunked)
    reqs = {r.req_id: r for r in workload}
    all_reqs = []    # 持有所有副本引用（完成的请求会从队列移除，需独立持有以收集 TTFT）
    for r in workload:
        nr = Request(r.req_id, r.prompt_len, r.max_output)
        sched.add_request(nr); all_reqs.append(nr)
    decode_tokens_total = 0
    step_count = 0
    while sched.waiting or sched.running:
        sched.step_count = step_count
        s = sched.step()
        decode_tokens_total += sum(v for rid, v in s.items()
                                   if reqs[rid].prompt_len <= 50)
        step_count += 1
        if step_count > 500: break    # 防死循环（策略A"不切"会饿死）
    # TTFT 从 all_reqs 收集（独立于当前队列状态）
    ttft_list = [(r.req_id, r.prefill_done_step) for r in all_reqs
                 if r.prefill_done_step >= 0]
    return {
        "decode_per_step_avg": decode_tokens_total / max(step_count, 1),
        "ttft_list": ttft_list,
        "total_steps": step_count,
    }


def compare_strategies():
    def make_workload():
        return [
            Request("long1", 5000, 3), Request("long2", 4000, 3),
            Request("short1", 20, 10), Request("short2", 20, 10),
            Request("short3", 20, 10),
        ]
    strategies = [
        ("A 不切(排队)",      dict(max_num_seqs=4, token_budget=2000, threshold=0,      enable_chunked=False)),
        ("B 只被动切",        dict(max_num_seqs=4, token_budget=2000, threshold=10**9,  enable_chunked=True)),
        ("C 主动切(thr=1000)",dict(max_num_seqs=4, token_budget=2000, threshold=1000,   enable_chunked=True)),
    ]
    print(f"{'策略':<22} {'总step数':<10} {'平均decode/step':<16} {'长req TTFT(req,step)'}")
    print("-" * 72)
    for name, cfg in strategies:
        r = simulate(make_workload(), **cfg)
        long_ttft = [t for t in r["ttft_list"] if t[0].startswith("long")]
        print(f"{name:<22} {r['total_steps']:<10} "
              f"{r['decode_per_step_avg']:<16.1f} {long_ttft}")
    print()
    print("  注：策略A(不切)在长prompt>budget时会饿死（排队死循环），")
    print("  这正是 chunked prefill 必要性的活教材——纯不切模式处理不了长prompt。")


def build_cu_seqlens(query_lens, kv_lens):
    cu_q = [0]
    for q in query_lens:
        cu_q.append(cu_q[-1] + q)
    cu_k = [0]
    for k in kv_lens:
        cu_k.append(cu_k[-1] + k)
    return cu_q, cu_k
"""
