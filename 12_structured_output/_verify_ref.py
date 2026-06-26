"""验证参考答案。仅标准库。"""
import math, random

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x-m) for x in logits]
    s = sum(exps)
    return [e/s for e in exps]

class NumberGrammar:
    START="START"; DIGIT="DIGIT"; END="END"
    def __init__(self): self.state=self.START
    def allowed_next_chars(self):
        if self.state==self.START: return set("0123456789")
        if self.state==self.DIGIT: return set("0123456789\n")
        return set()
    def accept(self, ch):
        if self.state==self.START and ch in "0123456789": self.state=self.DIGIT
        elif self.state==self.DIGIT:
            if ch=="\n": self.state=self.END
        return self.state
    def is_terminated(self): return self.state==self.END

def constrained_sample(logits, grammar, vocab, temperature=1.0, rng=None):
    if rng is None: rng=random.Random()
    allowed = grammar.allowed_next_chars()
    masked = [logits[i]/temperature if vocab[i] in allowed else float("-inf") for i in range(len(vocab))]
    probs = softmax(masked)
    r = rng.random(); cum = 0.0
    for i,p in enumerate(probs):
        cum += p
        if r <= cum: return i, vocab[i]
    return len(vocab)-1, vocab[-1]

class JsonKeyValueGrammar:
    LBRACE="LBRACE"; QUOTE_OPEN="QUOTE_OPEN"; KEY="KEY"
    EXPECT_COLON="EXPECT_COLON"; VALUE="VALUE"; END="END"
    def __init__(self): self.state=self.LBRACE
    def allowed_next_chars(self):
        s=self.state
        if s==self.LBRACE: return {"{"}
        if s==self.QUOTE_OPEN: return {'"'}
        if s==self.KEY: return set("abc")|{'"'}
        if s==self.EXPECT_COLON: return {":"}          # 键名结束后期望冒号
        if s==self.VALUE: return set("0123456789")|{"}"}
        return set()
    def accept(self, ch):
        s=self.state
        if s==self.LBRACE and ch=="{": self.state=self.QUOTE_OPEN
        elif s==self.QUOTE_OPEN and ch=='"': self.state=self.KEY
        elif s==self.KEY:
            if ch=='"': self.state=self.EXPECT_COLON   # 键名引号闭合→期望冒号
        elif s==self.EXPECT_COLON and ch==":": self.state=self.VALUE  # 冒号→值
        elif s==self.VALUE:
            if ch=="}": self.state=self.END
        return self.state
    def is_terminated(self): return self.state==self.END

def generate_constrained(grammar, logits_fn, vocab, max_steps=50, rng=None):
    if rng is None: rng=random.Random()
    output=""; steps=0
    while not grammar.is_terminated() and steps<max_steps:
        logits=logits_fn(steps)
        idx,ch=constrained_sample(logits,grammar,vocab,rng=rng)
        output+=ch; grammar.accept(ch); steps+=1
    return output

# 实践1
g=NumberGrammar()
assert g.state==NumberGrammar.START
assert g.allowed_next_chars()==set("0123456789")
g.accept("1"); assert g.state==NumberGrammar.DIGIT
al=g.allowed_next_chars()
assert set("0123456789").issubset(al) and "\n" in al
print("实践1 通过")

# 实践2
vocab=list("0123456789\nabcXYZ{}\": ")
g=NumberGrammar(); rng=random.Random(0)
logits=[0.0]*len(vocab)
for i,ch in enumerate(vocab):
    if ch in "abc": logits[i]=10.0
    elif ch in "0123456789": logits[i]=-10.0
samples=[]
for _ in range(5):
    if g.is_terminated(): break   # grammar 结束就停（END 状态无合法 token）
    idx,ch=constrained_sample(logits,g,vocab,1.0,rng); samples.append(ch); g.accept(ch)
# 采到的都应是数字（grammar 保证），\n 是结束符不计入"输出字符"断言
digit_samples=[c for c in samples if c!="\n"]
assert all(c in "0123456789" for c in digit_samples),f"{samples}"
print(f"实践2 通过: 约束采样 {samples}")

# 实践3
vocab=list('abc0123456789{}": ')
rng=random.Random(42)
def rl(step): return [rng.gauss(0,1) for _ in vocab]
g=JsonKeyValueGrammar()
out=generate_constrained(g,rl,vocab,30,rng)
assert out.startswith("{") and out.endswith("}") and '"' in out and ":" in out, f"{out!r}"
ci=out.index(":"); ac=out[ci+1:].strip(); vp=ac.rstrip("}").strip()
assert vp and all(c in "0123456789" for c in vp), f"value {vp!r} in {out!r}"
print(f"实践3 通过: 生成 {out!r}")
print("\n全部验证通过")
