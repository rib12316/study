"""
特性 #8 干中学实践：Speculative Decoding draft-verify 循环

目标：亲手重建投机解码的核心算法——
  ① 模拟 target/draft + accept/reject（连续匹配前缀 + bonus）
  ② 完整 draft-verify 循环 + 实际/理论加速比对比
  ③ N-gram draft proposer（真实模式匹配，不靠概率）

参考 vLLM 源码：
  - spec_decode/ngram_proposer.py:207  N-gram draft (KMP/LPS)
  - spec_decode/metrics.py:114         mean_acceptance_length = 1 + accepted/drafts
  - sampler.py:358                     spec token 组合

依赖：仅标准库（random）。不需要装 vllm/torch。
运行：python practice_speculative_decoding.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import random


# ============================================================================
# 实践 1：模拟 target/draft + accept/reject
# ============================================================================
def mock_target_next(context: tuple) -> int:
    """模拟大模型：基于 context 确定地预测下一个 token。
    用 hash 保证同一 context 总是同一 token（确定性）。
    """
    return hash(context) % 1000


def mock_draft_next(context: tuple, k: int, p_accept: float,
                    rng: random.Random) -> list[int]:
    """模拟 draft：以概率 p_accept 猜对（返回 target 的 token），否则随机错 token。
    返回 k 个 draft token。
    """
    # TODO(你): 实现
    # 对每个位置：以 p_accept 概率返回 mock_target_next 的结果，否则返回一个不同的随机 token
    pass


def verify(real_logits_token: int, draft_tokens: list[int]) -> tuple[int, int]:
    """验证 draft token，返回 (accepted_count, bonus_token)。
    
    算法（exact match）：
    - real_logits_token 是 target 对"真实上一 token 位置"的预测（应 == draft_tokens[0] 才接受第1个）
      为简化：我们直接比较 draft_tokens[i] 和 target 在该位置的预测。
    - 从左到右：第 i 个 draft token 若 == target 预测，接受；第一个不等则截断。
    - bonus = 截断处 target 的预测（肯定接受）。
    
    简化模型：假设 target 的预测就是 mock_target_next，draft 是否猜对由 mock_draft_next 的 p_accept 决定。
    所以这里只需检查 draft_tokens 里"哪些是正确的"（连续前缀）。
    
    返回 (接受的 draft 数, bonus token)。
    """
    # TODO(你): 实现 accept/reject
    # 提示：遍历 draft_tokens，找到第一个"猜错"的位置 j
    # accepted = j，bonus = target 对该位置的预测
    pass


# ============================================================================
# 实践 2：完整 draft-verify 循环 + 加速比
# ============================================================================
def speculative_decode(num_tokens: int, K: int, p_accept: float,
                       seed: int = 0) -> dict:
    """跑完整的投机解码循环，生成 num_tokens 个 token。
    返回统计：
      - target_forwards: target forward 总次数（每次 verify 算 1 次）
      - drafts: draft 轮数
      - total_accepted: 接受的 draft token 总数
      - actual_speedup: 实际加速比 = num_tokens / target_forwards
        （普通 decode 要 num_tokens 次 forward）
      - theoretical_speedup: 理论 = (1 - p^(K+1)) / (1 - p)，p=p_accept
    """
    rng = random.Random(seed)
    context = (0,)   # 初始 context（简化）
    target_forwards = 0
    drafts = 0
    total_accepted = 0
    produced = 0

    # TODO(你): 实现主循环
    # while produced < num_tokens:
    #     draft_tokens = mock_draft_next(context, K, p_accept, rng)   # draft
    #     drafts += 1
    #     target_forwards += 1                                         # verify 一次 forward
    #     accepted, bonus = verify(...)                                # 验证
    #     total_accepted += accepted
    #     produced += accepted + 1   # 接受的 draft + 1 bonus
    #     context = context + tuple(draft_tokens[:accepted]) + (bonus,)
    pass


def theoretical_speedup(p_accept: float, K: int) -> float:
    """理论加速比 = (1 - p^(K+1)) / (1 - p)。p=1 时返回 K+1。"""
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 3：N-gram draft proposer
# ============================================================================
class NgramProposer:
    """真实的 N-gram draft：维护已见 n-gram → 后续 token 序列，按模式匹配 draft。
    参考 ngram_proposer.py:207。
    """
    def __init__(self, n: int = 3, k: int = 4):
        self.n = n          # n-gram 长度
        self.k = k          # draft token 数
        self.history: list[int] = []          # 已生成的 token 历史
        self.ngram_table: dict[tuple, list[int]] = {}  # n-gram → 后续 tokens

    def _update_table(self):
        """从 history 里提取所有 n-gram 及其后继，存进 table。"""
        # TODO(你): 实现
        # 对 history 里每个长度 n 的窗口，记录它后面的 token
        pass

    def draft(self) -> list[int]:
        """用最后 n 个 token 查表，预测接下来 k 个 token。无匹配返回空。"""
        # TODO(你): 实现
        pass

    def observe(self, token: int):
        """记录新 token，更新历史和 table。"""
        self.history.append(token)
        self._update_table()


def run_ngram_test(text_tokens: list[int], n: int, k: int) -> dict:
    """用一段 token 序列测试 N-gram draft 的接受率。
    逐 token：先 observe 当前 token，再用历史 draft 预测下一个，和真实下一个比。
    """
    # TODO(你): 实现
    # 用 NgramProposer，target 用"真实下一个 token"（text_tokens[i+1]）
    # 统计：draft 次数、总 draft token、被接受的 token（draft[0]==true_next 即接受）
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：accept/reject ===")
    rng = random.Random(42)
    # p=1.0 全对：draft 应全部 == target 预测
    ctx = (1, 2, 3)
    d = mock_draft_next(ctx, 4, 1.0, rng)
    t = mock_target_next(ctx)
    assert all(x == t for x in d[:1]), "p=1 第一个应等于 target"
    # p=0.0 全错：draft 应全 != target（极大概率）
    d0 = mock_draft_next(ctx, 4, 0.0, rng)
    assert all(x != t for x in d0), f"p=0 应全错: {d0} vs {t}"
    # verify：全对应接受全部 + bonus
    accepted, bonus = verify(t, [t, t, t, t])  # 全 == target
    assert accepted == 4, f"全对应接受4: {accepted}"
    assert bonus == t, "bonus 应是 target 预测"
    print(f"  p=1.0 draft={d}, verify 全对 → accepted={accepted}, bonus={bonus} ✓")
    print(f"  p=0.0 draft={d0} (全错) ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：加速比对比 ===")
    for p in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        K = 4
        r = speculative_decode(num_tokens=2000, K=K, p_accept=p, seed=0)
        theo = theoretical_speedup(p, K)
        print(f"  p={p:.1f} K={K}: 实际加速={r['actual_speedup']:.2f}x "
              f"(理论 {theo:.2f}x), 接受率={r['total_accepted']/max(r['drafts']*K,1)*100:.0f}%")
    # 关键断言：p=0 时加速比应接近 1（不亏）
    r0 = speculative_decode(num_tokens=500, K=4, p_accept=0.0, seed=0)
    assert 0.95 <= r0["actual_speedup"] <= 1.05, f"p=0 应≈1.0x: {r0['actual_speedup']}"
    # p=1 时应接近 K+1=5
    r1 = speculative_decode(num_tokens=500, K=4, p_accept=1.0, seed=0)
    assert r1["actual_speedup"] >= 4.5, f"p=1 应≈5x: {r1['actual_speedup']}"
    print(f"  → p=0 不亏({r0['actual_speedup']:.2f}x≈1), p=1 接近 K+1({r1['actual_speedup']:.2f}x≈5) ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：N-gram draft ===")
    # 重复模式文本：A B C A B C A B C ...（n-gram=3 应高度匹配）
    repeating = [1, 2, 3] * 20
    r_rep = run_ngram_test(repeating, n=3, k=4)
    # 随机文本：接受率应低
    rng = random.Random(0)
    random_tokens = [rng.randint(0, 100) for _ in range(60)]
    r_rand = run_ngram_test(random_tokens, n=3, k=4)
    print(f"  重复模式 [1,2,3]*20: draft {r_rep['drafts']}次, "
          f"接受率 {r_rep['accepted']/max(r_rep['drafted'],1)*100:.0f}%")
    print(f"  随机文本: draft {r_rand['drafts']}次, "
          f"接受率 {r_rand['accepted']/max(r_rand['drafted'],1)*100:.0f}%")
    assert r_rep["accepted"] > r_rand["accepted"], "重复文本接受率应更高"
    print(f"  → N-gram 对重复模式接受率显著更高 ✓\n")


def main():
    print("Speculative Decoding 投机解码实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现投机解码 draft-verify 核心并验证加速比。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def mock_draft_next(context, k, p_accept, rng):
    target_tok = mock_target_next(context)
    result = []
    for i in range(k):
        if rng.random() < p_accept:
            result.append(target_tok)
        else:
            # 随机一个 != target 的 token
            wrong = target_tok
            while wrong == target_tok:
                wrong = rng.randint(0, 999)
            result.append(wrong)
    return result


def verify(real_token, draft_tokens):
    # 简化：draft_tokens 里"正确"的 = == real_token 的连续前缀
    accepted = 0
    for d in draft_tokens:
        if d == real_token:
            accepted += 1
        else:
            break
    bonus = real_token   # target 对截断处的预测
    return accepted, bonus


def speculative_decode(num_tokens, K, p_accept, seed=0):
    rng = random.Random(seed)
    context = (0,)
    target_forwards = 0
    drafts = 0
    total_accepted = 0
    produced = 0
    while produced < num_tokens:
        draft_tokens = mock_draft_next(context, K, p_accept, rng)
        drafts += 1
        target_forwards += 1
        real_tok = mock_target_next(context)   # target 对当前位置的预测
        accepted, bonus = verify(real_tok, draft_tokens)
        total_accepted += accepted
        produced += accepted + 1
        context = context + tuple(draft_tokens[:accepted]) + (bonus,)
    theo = theoretical_speedup(p_accept, K)
    return {
        "target_forwards": target_forwards,
        "drafts": drafts,
        "total_accepted": total_accepted,
        "actual_speedup": num_tokens / target_forwards,
        "theoretical_speedup": theo,
    }


def theoretical_speedup(p_accept, K):
    if p_accept >= 1.0:
        return float(K + 1)
    return (1 - p_accept ** (K + 1)) / (1 - p_accept)


class NgramProposer:
    def __init__(self, n=3, k=4):
        self.n = n; self.k = k
        self.history = []
        self.ngram_table = {}

    def _update_table(self):
        h = self.history
        for i in range(len(h) - self.n):
            gram = tuple(h[i:i+self.n])
            nxt = h[i+self.n]
            self.ngram_table.setdefault(gram, []).append(nxt)

    def draft(self):
        if len(self.history) < self.n:
            return []
        gram = tuple(self.history[-self.n:])
        if gram not in self.ngram_table:
            return []
        # 取第一次匹配后的 k 个（简化：用 table 里这个 gram 后见过的序列）
        seq = self.ngram_table[gram]
        return seq[:self.k]

    def observe(self, token):
        self.history.append(token)
        self._update_table()


def run_ngram_test(text_tokens, n, k):
    prop = NgramProposer(n=n, k=k)
    drafts = 0
    drafted = 0
    accepted = 0
    for i in range(len(text_tokens) - 1):
        prop.observe(text_tokens[i])        # 先观察当前
        true_next = text_tokens[i+1]
        draft = prop.draft()
        if not draft:
            continue
        drafts += 1
        drafted += len(draft)
        if draft[0] == true_next:           # draft[0] 预测对就算接受
            accepted += 1
    return {"drafts": drafts, "drafted": drafted, "accepted": accepted}
"""
