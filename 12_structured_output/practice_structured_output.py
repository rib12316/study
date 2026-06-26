"""
特性 #12 干中学实践：mini Grammar Constrained Sampler

目标：亲手重建结构化输出的核心机制——
  ① 简单 grammar FSM（数字串）+ allowed_next_chars
  ② bitmask 应用到 logits（非法置 -inf）+ 受约束采样
  ③ 完整 JSON 键值 grammar FSM + 端到端生成

参考 vLLM 源码：
  - structured_output/__init__.py    _fill_bitmasks（bitmask 收集）
  - backend_xgrammar.py:136          XgrammarGrammar（fill_bitmask/accept_tokens/rollback）
  - sample/sampler.py                masked_fill_(~mask, -inf)（bitmask 应用）

简化：用 token=字符 模型（词表=ASCII），避免 BPE 复杂性。
依赖：仅标准库（math, random）。不需要装 vllm/torch/xgrammar。
运行：python practice_structured_output.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import math
import random


# ============================================================================
# 实践 1：数字串 grammar FSM
# ============================================================================
class NumberGrammar:
    """匹配纯数字串（如 '12345'）的 FSM。
    状态：START → DIGIT → DIGIT → ... → END
    参考 backend_xgrammar.py 的 grammar matcher 思路（状态机）。
    """
    START = "START"
    DIGIT = "DIGIT"
    END = "END"

    def __init__(self):
        self.state = self.START

    def allowed_next_chars(self) -> set[str]:
        """返回当前状态允许的字符集（白名单）。"""
        # TODO(你): 实现
        # START: 允许 '0'-'9'
        # DIGIT: 允许 '0'-'9' 和 '\n'（触发结束）
        # END: 空集
        pass

    def accept(self, ch: str) -> str:
        """接受一个字符，推进状态，返回新状态。"""
        # TODO(你): 实现
        pass

    def is_terminated(self) -> bool:
        return self.state == self.END


# ============================================================================
# 实践 2：Bitmask + 受约束采样（核心）
# ============================================================================
def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def constrained_sample(logits: list[float], grammar, vocab: list[str],
                       temperature: float = 1.0,
                       rng: random.Random | None = None) -> tuple[int, str]:
    """受 grammar 约束的采样。
    参考 sampler 的 masked_fill_(~mask, -inf) + 采样。
    
    1. mask = grammar.allowed_next_chars()  # 白名单字符集
    2. 对 logits：非白名单字符位置置 -inf
    3. softmax（带 temperature）→ 概率
    4. 按概率采样一个字符 index
    返回 (char_index, char)。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 3：JSON 键值 grammar（进阶）
# ============================================================================
class JsonKeyValueGrammar:
    """匹配 {"<字母键>": <数字>} 的简化 JSON FSM。
    状态序列：
      LBRACE(期望'{') → QUOTE_OPEN(期望'"') → KEY(键名字母)
      → (KEY接受'"'后) EXPECT_COLON(期望':') → VALUE(数字) → END(接受'}')
    """
    LBRACE = "LBRACE"
    QUOTE_OPEN = "QUOTE_OPEN"
    KEY = "KEY"
    EXPECT_COLON = "EXPECT_COLON"   # 键名引号闭合后，期望冒号
    VALUE = "VALUE"
    END = "END"

    def __init__(self):
        self.state = self.LBRACE

    def allowed_next_chars(self) -> set[str]:
        """每个状态允许的字符。"""
        # TODO(你): 实现
        # LBRACE: {'{'}
        # QUOTE_OPEN: {'"'}
        # KEY: {字母 a-z A-Z} + {'"'}（可继续键名或结束）
        # EXPECT_COLON: {':'}（键名闭合后期望冒号）
        # VALUE: {数字 0-9} + {'}'}（值可多位数字或结束）
        # END: {} （已结束）
        # 注意：状态转移要和 accept 一致，这里只返回"当前允许字符"
        pass

    def accept(self, ch: str) -> str:
        """接受字符，推进状态。"""
        # TODO(你): 实现
        # 根据当前状态和字符决定下一状态：
        # LBRACE+'{' → QUOTE_OPEN
        # QUOTE_OPEN+'"' → KEY
        # KEY+'"' → EXPECT_COLON；KEY+字母 → 保持KEY
        # EXPECT_COLON+':' → VALUE
        # VALUE+'}' → END；VALUE+数字 → 保持VALUE
        pass

    def is_terminated(self) -> bool:
        return self.state == self.END


def generate_constrained(grammar, logits_fn, vocab: list[str],
                         max_steps: int = 50,
                         rng: random.Random | None = None) -> str:
    """端到端：用 grammar 约束，循环采样直到 terminated。
    logits_fn(step) -> list[float]：模拟每步的 logits（可以是随机的，grammar 会约束）。
    """
    # TODO(你): 实现
    # output = ""
    # while not grammar.is_terminated() and len(output) < max_steps:
    #     logits = logits_fn(len(output))
    #     idx, ch = constrained_sample(logits, grammar, vocab, rng=rng)
    #     output += ch
    #     grammar.accept(ch)
    # return output
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def make_digit_vocab():
    return list("0123456789\nabcXYZ{}\": ")  # 词表含数字+非法字符


def verify_practice1():
    print("=== 实践 1 验证：数字串 FSM ===")
    g = NumberGrammar()
    assert g.state == NumberGrammar.START
    allowed = g.allowed_next_chars()
    assert allowed == set("0123456789"), f"START 应只允许数字: {allowed}"
    g.accept("1")
    assert g.state == NumberGrammar.DIGIT
    allowed = g.allowed_next_chars()
    assert set("0123456789").issubset(allowed), f"DIGIT 应允许数字: {allowed}"
    assert "\n" in allowed, "DIGIT 应允许\\n 结束"
    print(f"  START→allowed={sorted(allowed)[:5]}... ✓")
    print(f"  accept('1')→DIGIT, allowed含数字+\\n ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：受约束采样 ===")
    vocab = make_digit_vocab()
    g = NumberGrammar()
    rng = random.Random(0)
    # 构造 logits 强烈偏向字母（非法），看约束后是否只出数字
    # 给 'a','b','c' 极高 logits，数字极低
    logits = [0.0] * len(vocab)
    for i, ch in enumerate(vocab):
        if ch in "abc":
            logits[i] = 10.0   # 字母极高
        elif ch in "0123456789":
            logits[i] = -10.0  # 数字极低
    # 约束采样 5 次
    samples = []
    for _ in range(5):
        idx, ch = constrained_sample(logits, g, vocab, temperature=1.0, rng=rng)
        samples.append(ch)
        g.accept(ch)
    assert all(c in "0123456789" for c in samples), \
        f"约束后应只采样数字: {samples}"
    print(f"  logits 强烈偏向字母(a,b,c=10, 数字=-10)")
    print(f"  约束采样 5 次: {samples}（全是数字）✓")
    print(f"  → 非法字符 logit 被 -inf 屏蔽，即使概率高也选不到 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：JSON 键值 FSM ===")
    vocab = list('abc0123456789{}": ')
    rng = random.Random(42)

    def random_logits(step):
        return [rng.gauss(0, 1) for _ in vocab]

    g = JsonKeyValueGrammar()
    output = generate_constrained(g, random_logits, vocab, max_steps=30, rng=rng)
    # 验证：输出必须是合法的 {"<字母>": <数字>} 形式
    assert output.startswith("{"), f"应以{{开头: {output!r}"
    assert output.endswith("}"), f"应以}}结尾: {output!r}"
    assert '"' in output, f"应含引号: {output!r}"
    assert ":" in output, f"应含冒号: {output!r}"
    # 冒号后应是数字
    colon_idx = output.index(":")
    after_colon = output[colon_idx+1:].strip()
    # 去掉结尾 }，剩下的应是数字
    value_part = after_colon.rstrip("}").strip()
    assert value_part and all(c in "0123456789" for c in value_part), \
        f"冒号后应是数字: {value_part!r} (完整 {output!r})"
    print(f"  随机 logits + grammar 约束生成: {output!r}")
    print(f"  无论 logits 如何随机，输出始终合法 ✓\n")


def main():
    print("mini Grammar Constrained Sampler 实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 Grammar Constrained Sampling 核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
class NumberGrammar:
    START = "START"; DIGIT = "DIGIT"; END = "END"
    def __init__(self): self.state = self.START
    def allowed_next_chars(self):
        if self.state == self.START: return set("0123456789")
        if self.state == self.DIGIT: return set("0123456789\\n")
        return set()
    def accept(self, ch):
        if self.state == self.START and ch in "0123456789":
            self.state = self.DIGIT
        elif self.state == self.DIGIT:
            if ch == "\\n": self.state = self.END
            # 数字则保持 DIGIT
        return self.state
    def is_terminated(self): return self.state == self.END


def constrained_sample(logits, grammar, vocab, temperature=1.0, rng=None):
    if rng is None: rng = random.Random()
    allowed = grammar.allowed_next_chars()
    # 应用 mask：非白名单置 -inf
    masked = [logits[i] / temperature if vocab[i] in allowed else float("-inf")
              for i in range(len(vocab))]
    probs = softmax(masked)
    # 按概率采样
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i, vocab[i]
    return len(vocab)-1, vocab[-1]


class JsonKeyValueGrammar:
    LBRACE="LBRACE"; QUOTE_OPEN="QUOTE_OPEN"; KEY="KEY"
    EXPECT_COLON="EXPECT_COLON"; VALUE="VALUE"; END="END"
    def __init__(self): self.state = self.LBRACE
    def allowed_next_chars(self):
        s = self.state
        if s == self.LBRACE: return {"{"}
        if s == self.QUOTE_OPEN: return {'"'}
        if s == self.KEY: return set("abc") | {'"'}
        if s == self.EXPECT_COLON: return {":"}
        if s == self.VALUE: return set("0123456789") | {"}"}
        return set()
    def accept(self, ch):
        s = self.state
        if s == self.LBRACE and ch == "{": self.state = self.QUOTE_OPEN
        elif s == self.QUOTE_OPEN and ch == '"': self.state = self.KEY
        elif s == self.KEY:
            if ch == '"': self.state = self.EXPECT_COLON
            # 字母则保持 KEY
        elif s == self.EXPECT_COLON and ch == ":": self.state = self.VALUE
        elif s == self.VALUE:
            if ch == "}": self.state = self.END
            # 数字则保持 VALUE
        return self.state
    def is_terminated(self): return self.state == self.END


def generate_constrained(grammar, logits_fn, vocab, max_steps=50, rng=None):
    if rng is None: rng = random.Random()
    output = ""
    steps = 0
    while not grammar.is_terminated() and steps < max_steps:
        logits = logits_fn(steps)
        idx, ch = constrained_sample(logits, grammar, vocab, rng=rng)
        output += ch
        grammar.accept(ch)
        steps += 1
    return output
"""
