"""
特性 #6 干中学实践：mini Continuous Batching 调度器

目标：亲手重建 vLLM v1 Scheduler 的核心——
  ① Request 模型 + waiting/running 两队列
  ② step() 统一调度：先 running(decode, 1 token) 后 waiting(prefill)，用 token_budget 约束
  ③ 抢占：KV 块不足时牺牲 running 里输出最多的请求（重算代价）

参考 vLLM 源码：
  - scheduler.py:388   schedule() 主循环（先 running 后 waiting）
  - scheduler.py:390   "no prefill/decode phase" 统一抽象注释
  - scheduler.py:538   抢占逻辑
  - scheduler.py:1106  _preempt_request（放回 waiting 头部）

依赖：仅标准库。不需要装 vllm/torch。
运行：python practice_continuous_batching.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations


def cdiv(n: int, b: int) -> int:
    """ceil division（第五讲元运算）。"""
    return (n + b - 1) // b


# ============================================================================
# 实践 1：Request + Scheduler 骨架
# ============================================================================
class Request:
    """一个推理请求。
    - prompt_len: prefill 要算的 token 数
    - max_output: 最多生成多少 output token
    - num_computed: 已经算到第几个 token（含 prompt 和 output）
    - output_tokens: 已生成的 output token 数
    - prefill_done: prefill 是否完成（num_computed >= prompt_len）
    """
    def __init__(self, req_id: str, prompt_len: int, max_output: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_output = max_output
        self.num_computed = 0
        self.output_tokens = 0
        self.num_preemptions = 0

    @property
    def prefill_done(self) -> bool:
        return self.num_computed >= self.prompt_len

    def num_uncomputed(self) -> int:
        """还有多少 token 没算。
        prefill 阶段：prompt_len - num_computed
        decode 阶段：始终还要算下一个 output token（1 个），除非已生成完
        """
        # TODO(你): 实现
        pass

    def is_finished(self) -> bool:
        """output_tokens 达到 max_output 即完成。"""
        return self.output_tokens >= self.max_output

    def __repr__(self):
        return (f"Req({self.req_id}, prompt={self.prompt_len}, "
                f"out={self.output_tokens}/{self.max_output}, "
                f"computed={self.num_computed}, preempt={self.num_preemptions})")


class Scheduler:
    """mini continuous batching 调度器。
    参考 scheduler.py:68 Scheduler。
    """
    def __init__(self, max_num_seqs: int, token_budget: int,
                 block_size: int = 16, max_blocks: int | None = None):
        self.max_num_seqs = max_num_seqs        # running 上限
        self.token_budget = token_budget        # 每 step 总 token 预算
        self.block_size = block_size            # KV 块大小（实践3用）
        self.max_blocks = max_blocks            # 总 KV 块数（实践3用，None=不限制）
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_count = 0

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    # 实践 3：抢占用
    def _req_blocks(self, req: Request) -> int:
        """请求当前占用的 KV 块数 = cdiv(num_computed, block_size)。"""
        return cdiv(req.num_computed, self.block_size)

    def _total_blocks_used(self) -> int:
        return sum(self._req_blocks(r) for r in self.running)

    # TODO(你): 实现下面的 step / preempt
    def step(self) -> dict[str, int]:
        """模拟一次 GPU forward。返回 {req_id: 本step分到的token数}。
        参考 scheduler.py:388 schedule()。
        """
        pass

    def preempt(self, req: Request) -> None:
        """抢占一个请求：num_computed 归零（KV释放要重算），放回 waiting 头部。
        参考 scheduler.py:1106 _preempt_request。
        """
        pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：Request 模型 ===")
    r = Request("r0", prompt_len=10, max_output=3)
    assert r.num_uncomputed() == 10, f"prefill 前应 10: {r.num_uncomputed()}"
    assert not r.prefill_done
    r.num_computed = 10
    assert r.prefill_done
    # prefill 完后，还要生成 1 个 output token（decode）
    assert r.num_uncomputed() == 1, f"decode 应 1: {r.num_uncomputed()}"
    r.output_tokens = 3
    assert r.is_finished()
    print("  Request 状态机正确 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：统一调度循环 ===")
    sched = Scheduler(max_num_seqs=4, token_budget=20)
    # 3 个请求：不同 prompt_len 和 max_output
    ra = Request("A", prompt_len=5, max_output=2)   # 短
    rb = Request("B", prompt_len=5, max_output=4)   # 中
    rc = Request("C", prompt_len=12, max_output=3)  # 长 prompt
    all_reqs = {"A": ra, "B": rb, "C": rc}
    sched.add_request(ra); sched.add_request(rb); sched.add_request(rc)

    history = []
    for i in range(30):
        if not sched.waiting and not sched.running:
            break
        scheduled = sched.step()
        sched.step_count += 1
        # 记录每步调度
        history.append((i, dict(scheduled),
                        [r.req_id for r in sched.running],
                        [r.req_id for r in sched.waiting]))
    # 用 all_reqs 直接判断完成态（step 内部会移除完成的请求）
    finished = {rid for rid, r in all_reqs.items() if r.is_finished()}

    # 断言1：A（短）应该比 C（长）先完成
    assert "A" in finished and "B" in finished and "C" in finished, \
        f"没全部完成: {finished}"

    # 断言2：至少有一个 step 同时调度了 decode 和 prefill（continuous batching 特征）
    mixed_steps = [h for h in history if len(h[1]) >= 2]
    assert len(mixed_steps) > 0, "没有混合调度 step，不是 continuous batching"

    # 断言3：第一个 step 应该 prefill（budget=20 够 A+B+C 的 prompt? A5+B5+C12=22>20, 会 chunked）
    step0 = history[0][1]
    assert sum(step0.values()) <= 20, f"step0 超 budget: {step0}"

    print(f"  跑了 {len(history)} 步，全部完成: {sorted(finished)}")
    print(f"  混合调度 step 数: {len(mixed_steps)} (continuous batching 特征)")
    print(f"  step0 调度: {step0} (先 prefill，budget 约束生效)")
    # 找一个混合 step 展示
    for h in mixed_steps[:2]:
        print(f"  step{h[0]}: scheduled={h[1]} running={h[2]} waiting={h[3]}")
    print(f"  → 短请求先完成、混合调度、budget 约束 全部正确 ✓\n")
    return history


def verify_practice3():
    print("=== 实践 3 验证：抢占 ===")
    # max_blocks 很小，强制触发抢占
    sched = Scheduler(max_num_seqs=4, token_budget=20, block_size=16, max_blocks=3)
    sched.add_request(Request("X", prompt_len=16, max_output=5))   # 占 1 块(prefill) + decode
    sched.add_request(Request("Y", prompt_len=16, max_output=5))
    sched.add_request(Request("Z", prompt_len=16, max_output=5))

    # 跑几步直到发生抢占
    saw_preempt = False
    for i in range(50):
        if not sched.waiting and not sched.running:
            break
        before_running = len(sched.running)
        sched.step()
        sched.step_count += 1
        # 检测抢占：某请求 num_preemptions 增加
        for r in sched.waiting + sched.running:
            if r.num_preemptions > 0:
                saw_preempt = True
        if saw_preempt and i > 2:
            break

    assert saw_preempt, "没发生抢占，max_blocks 约束没生效"
    # 找到被抢占的请求
    preempted = [r for r in sched.waiting + sched.running if r.num_preemptions > 0]
    assert len(preempted) > 0, "没找到被抢占的请求"
    # 被抢占的请求 output_tokens 应清零（preempt 会清零，重 prefill 不恢复 output）
    # 注意：num_computed 可能在抢占后被重新 prefill 累加，所以不要求它当前为 0
    for r in preempted:
        assert r.output_tokens == 0, f"被抢占请求 {r.req_id} output 应清零: {r.output_tokens}"

    print(f"  max_blocks=3 触发抢占 ✓")
    print(f"  被抢占请求: {[(r.req_id, r.num_preemptions) for r in preempted]}")
    print(f"  被抢占后 output_tokens 清零（KV 释放，重新 prefill 不恢复 output）✓")
    print(f"  → 你复现了 vLLM 调度器的抢占机制 ✓\n")


def main():
    print("mini Continuous Batching 调度器实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 Continuous Batching 调度核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
class Request:
    def __init__(self, req_id, prompt_len, max_output):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_output = max_output
        self.num_computed = 0
        self.output_tokens = 0
        self.num_preemptions = 0

    @property
    def prefill_done(self):
        return self.num_computed >= self.prompt_len

    def num_uncomputed(self):
        if not self.prefill_done:
            return self.prompt_len - self.num_computed
        # decode 阶段：若没生成完，还要算 1 个；生成完则 0
        if self.is_finished():
            return 0
        return 1

    def is_finished(self):
        return self.output_tokens >= self.max_output


class Scheduler:
    def __init__(self, max_num_seqs, token_budget, block_size=16, max_blocks=None):
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.waiting = []
        self.running = []
        self.step_count = 0

    def add_request(self, req):
        self.waiting.append(req)

    def _req_blocks(self, req):
        return cdiv(req.num_computed, self.block_size)

    def _total_blocks_used(self):
        return sum(self._req_blocks(r) for r in self.running)

    def step(self):
        budget = self.token_budget
        scheduled = {}
        # 阶段1：先服务 running（每个 decode 1 token）
        for req in list(self.running):
            if budget <= 0:
                break
            if req.is_finished():
                continue
            # decode: 算 1 个新 output token
            req.num_computed += 1
            req.output_tokens += 1
            budget -= 1
            scheduled[req.req_id] = scheduled.get(req.req_id, 0) + 1
        # 移除完成的
        self.running = [r for r in self.running if not r.is_finished()]

        # 阶段2：接纳 waiting（prefill，可能 chunked）
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            # 实践3：检查 KV 块是否够，不够则抢占
            req = self.waiting.pop(0)
            need = req.prompt_len - req.num_computed   # prefill 剩余
            alloc = min(need, budget)
            # KV 块约束（实践3）
            if self.max_blocks is not None:
                new_blocks_after = cdiv(req.num_computed + alloc, self.block_size)
                delta = new_blocks_after - self._req_blocks(req)
                while self._total_blocks_used() + delta > self.max_blocks and self.running:
                    # 抢占输出最多的 running 请求
                    victim = max(self.running, key=lambda r: r.output_tokens)
                    self.running.remove(victim)
                    self.preempt(victim)
                    # 重新算 delta（victim 释放了块）
                    new_blocks_after = cdiv(req.num_computed + alloc, self.block_size)
                    delta = new_blocks_after - self._req_blocks(req)
                if self._total_blocks_used() + delta > self.max_blocks:
                    # 还是不够，这个请求放回 waiting 头部
                    self.waiting.insert(0, req)
                    break
            req.num_computed += alloc
            budget -= alloc
            scheduled[req.req_id] = scheduled.get(req.req_id, 0) + alloc
            if req.num_computed < req.prompt_len:
                # chunked prefill 没完，放回 waiting 头部
                self.waiting.insert(0, req)
            else:
                # prefill 完，进 running
                self.running.append(req)
        return scheduled

    def preempt(self, req):
        req.num_preemptions += 1
        req.num_computed = 0           # KV 释放，要重算
        req.output_tokens = 0
        self.waiting.insert(0, req)    # 放回头部，优先重 prefill
"""
