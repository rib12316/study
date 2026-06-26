"""
特性 #11 干中学实践：mini LoRA（低秩叠加 + 多 adapter 混合 batch + 热加载）

目标：亲手重建 LoRA 的核心机制——
  ① 单 LoRA 叠加：y = W_base @ x + scaling * B @ A @ x
  ② 多 adapter 混合 batch：一个 batch 里不同 token 用不同 adapter（punica 思路）
  ③ 热加载生命周期：load/unload adapter，index 槽位复用

参考 vLLM 源码：
  - lora/layers/base_linear.py:194   apply（base + lora 叠加）
  - lora/layers/base_linear.py:129   lora_a/b_stacked（预分配 stack）
  - lora/layers/base_linear.py:158   set_lora（按 index 装载）
  - lora/punica_wrapper              add_lora_linear（混合 batch 分组计算）

依赖：仅标准库（random）。不需要装 vllm/torch。
运行：python practice_lora.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from __future__ import annotations
import random


# ============================================================================
# 辅助：矩阵运算（纯 Python）
# ============================================================================
def matvec(W: list[list[float]], x: list[float]) -> list[float]:
    """矩阵向量乘 W[M][N] @ x[N] -> [M]。"""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def matvec_AB(A: list[list[float]], B: list[list[float]], x: list[float]) -> list[float]:
    """算 B @ (A @ x)。A[r][N], B[M][r]。
    先 A@x 得 [r]，再 B@得 [M]。模拟 LoRA 的低秩路径。
    """
    # TODO(你): 实现（先 A@x 再 B@中间结果）
    pass


# ============================================================================
# 实践 1：单 LoRA 叠加
# ============================================================================
def lora_forward(x: list[float], W_base: list[list[float]],
                 A: list[list[float]], B: list[list[float]],
                 scaling: float) -> list[float]:
    """y = W_base @ x + scaling * B @ A @ x。
    参考 base_linear.py:204 _apply_sync（base_output + add_lora）。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 2：多 adapter 混合 batch（核心）
# ============================================================================
def lora_mixed_batch(tokens: list[list[float]],
                     W_base: list[list[float]],
                     adapters_A: list[list[list[float]]],
                     adapters_B: list[list[list[float]]],
                     adapter_indices: list[int],
                     scaling: float) -> list[list[float]]:
    """混合 batch：每个 token 用自己的 adapter。
    adapters_A[i]、adapters_B[i] 是第 i 个 adapter 的 A/B（模拟 stacked 张量）。
    adapter_indices[t] 是 token t 用的 adapter index。
    参考 punica_wrapper.add_lora_linear（分组按 adapter 计算）。
    """
    # TODO(你): 实现
    # 对每个 token t：y[t] = W_base @ x[t] + scaling * adapters_B[idx] @ adapters_A[idx] @ x[t]
    pass


# ============================================================================
# 实践 3：LoRA 热加载生命周期
# ============================================================================
class LoRAManager:
    """管理 adapter 的加载/卸载，index 槽位复用。
    参考 lora/worker_manager.py + base_linear.py:set_lora。
    """
    def __init__(self, W_base, max_adapters: int, scaling: float = 1.0):
        self.W_base = W_base
        self.max_adapters = max_adapters
        self.scaling = scaling
        # adapter_id -> index 槽位
        self.id_to_index: dict[int, int] = {}
        # index -> (A, B) 或 None（空闲）
        self.slots: list[tuple | None] = [None] * max_adapters
        # index -> ref_cnt
        self.ref_cnt: list[int] = [0] * max_adapters

    def load_adapter(self, adapter_id: int, A, B) -> int:
        """加载 adapter 到一个空闲 index 槽。返回 index。
        参考 set_lora。若无空闲槽 raise。
        """
        # TODO(你): 实现
        pass

    def unload_adapter(self, adapter_id: int) -> None:
        """卸载 adapter，释放 index 槽（仅当 ref_cnt=0）。
        参考 worker_manager 的 unload 逻辑。
        """
        # TODO(你): 实现
        pass

    def serve(self, tokens: list[list[float]], adapter_ids: list[int]) -> list[list[float]]:
        """服务一批混合请求。每个 token 的 adapter_id 查 index，混合计算。
        参考 add_lora_linear 的分组计算。
        """
        # TODO(你): 实现
        # 1. 把 adapter_ids 转成 index
        # 2. 调 lora_mixed_batch
        pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：单 LoRA 叠加 ===")
    random.seed(0)
    in_dim, out_dim, r = 3, 4, 2
    W = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(out_dim)]
    A = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(r)]
    B = [[random.gauss(0,1) for _ in range(r)] for _ in range(out_dim)]
    x = [1.0, 0.5, -0.3]
    scaling = 0.5

    # scaling=0 → 输出 = base
    y0 = lora_forward(x, W, A, B, scaling=0.0)
    base = matvec(W, x)
    assert all(abs(a-b) < 1e-6 for a,b in zip(y0, base)), f"scaling=0 应=base: {y0} vs {base}"

    # 手算验证
    y = lora_forward(x, W, A, B, scaling)
    manual = [base[i] + scaling * matvec_AB(A, B, x)[i] for i in range(out_dim)]
    assert all(abs(a-b) < 1e-6 for a,b in zip(y, manual)), f"LoRA前向与手算不符: {y} vs {manual}"
    print(f"  scaling=0: 输出=base ✓")
    print(f"  scaling={scaling}: base + LoRA修正 ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：多 adapter 混合 batch ===")
    random.seed(42)
    in_dim, out_dim, r = 3, 2, 2
    W = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(out_dim)]
    # 2 个 adapter
    A0 = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(r)]
    B0 = [[random.gauss(0,1) for _ in range(r)] for _ in range(out_dim)]
    A1 = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(r)]
    B1 = [[random.gauss(0,1) for _ in range(r)] for _ in range(out_dim)]
    # 4 个 token：前2用adapter0，后2用adapter1
    tokens = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(4)]
    indices = [0, 0, 1, 1]
    scaling = 1.0

    out = lora_mixed_batch(tokens, W, [A0, A1], [B0, B1], indices, scaling)
    assert len(out) == 4
    # 逐 token 手算验证
    for t in range(4):
        idx = indices[t]
        A = [A0, A1][idx]; B = [B0, B1][idx]
        manual = [matvec(W, tokens[t])[i] + scaling * matvec_AB(A, B, tokens[t])[i]
                  for i in range(out_dim)]
        assert all(abs(a-b) < 1e-6 for a,b in zip(out[t], manual)), \
            f"token{t}(adapter{idx}) 不符: {out[t]} vs {manual}"
    # 确认前2个 token 用了 adapter0、后2个用 adapter1（输出应不同，因为 adapter 不同）
    assert out[0] != out[2], "不同 adapter 应产生不同输出"
    print(f"  4 token 混合 batch: tok0,1→adapter0, tok2,3→adapter1")
    print(f"  每个 token 用了正确 adapter，与逐 token 手算一致 ✓")
    print(f"  不同 adapter 产生不同输出 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：热加载生命周期 ===")
    random.seed(7)
    in_dim, out_dim, r = 3, 2, 2
    W = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(out_dim)]
    mgr = LoRAManager(W, max_adapters=2, scaling=1.0)
    # adapter 构造器
    def make_ab():
        A = [[random.gauss(0,1) for _ in range(in_dim)] for _ in range(r)]
        B = [[random.gauss(0,1) for _ in range(r)] for _ in range(out_dim)]
        return A, B

    # 加载 adapter 100 → index 0
    A100, B100 = make_ab()
    idx100 = mgr.load_adapter(100, A100, B100)
    assert idx100 == 0, f"第一个应得 index0: {idx100}"
    # 加载 adapter 200 → index 1
    A200, B200 = make_ab()
    idx200 = mgr.load_adapter(200, A200, B200)
    assert idx200 == 1, f"第二个应得 index1: {idx200}"
    # 满了，第三个应失败
    A300, B300 = make_ab()
    try:
        mgr.load_adapter(300, A300, B300)
        assert False, "应 raise（槽满）"
    except (RuntimeError, ValueError):
        pass

    # 服务混合请求
    tokens = [[1.0]*in_dim, [0.5]*in_dim]
    out = mgr.serve(tokens, [100, 200])
    assert len(out) == 2
    # 验证：token0 用 adapter100、token1 用 adapter200
    manual0 = [matvec(W, tokens[0])[i] + matvec_AB(A100,B100,tokens[0])[i] for i in range(out_dim)]
    manual1 = [matvec(W, tokens[1])[i] + matvec_AB(A200,B200,tokens[1])[i] for i in range(out_dim)]
    assert all(abs(a-b)<1e-6 for a,b in zip(out[0], manual0)), f"token0(adapter100)不符"
    assert all(abs(a-b)<1e-6 for a,b in zip(out[1], manual1)), f"token1(adapter200)不符"

    # 卸载 adapter100，加载 adapter300 应复用 index 0
    mgr.unload_adapter(100)
    idx300 = mgr.load_adapter(300, A300, B300)
    assert idx300 == 0, f"卸载后应复用 index0: {idx300}"
    assert 300 in mgr.id_to_index and 100 not in mgr.id_to_index

    print(f"  load(100)→idx0, load(200)→idx1, 满了拒绝300 ✓")
    print(f"  serve 混合: token0→adapter100, token1→adapter200 正确 ✓")
    print(f"  unload(100) 后 load(300) 复用 idx0 ✓\n")


def main():
    print("mini LoRA 热加载实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 LoRA 低秩叠加 + 多 adapter 混合 + 热加载。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def matvec_AB(A, B, x):
    # A[r][N] @ x[N] -> tmp[r]
    tmp = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    # B[M][r] @ tmp[r] -> [M]
    return [sum(B[i][j] * tmp[j] for j in range(len(tmp))) for i in range(len(B))]


def lora_forward(x, W_base, A, B, scaling):
    base = matvec(W_base, x)
    lora = matvec_AB(A, B, x)
    return [base[i] + scaling * lora[i] for i in range(len(base))]


def lora_mixed_batch(tokens, W_base, adapters_A, adapters_B, adapter_indices, scaling):
    output = []
    for t, x in enumerate(tokens):
        idx = adapter_indices[t]
        base = matvec(W_base, x)
        lora = matvec_AB(adapters_A[idx], adapters_B[idx], x)
        output.append([base[i] + scaling * lora[i] for i in range(len(base))])
    return output


class LoRAManager:
    def __init__(self, W_base, max_adapters, scaling=1.0):
        self.W_base = W_base
        self.max_adapters = max_adapters
        self.scaling = scaling
        self.id_to_index = {}
        self.slots = [None] * max_adapters
        self.ref_cnt = [0] * max_adapters

    def load_adapter(self, adapter_id, A, B):
        if adapter_id in self.id_to_index:
            return self.id_to_index[adapter_id]
        # 找空闲槽
        for i in range(self.max_adapters):
            if self.slots[i] is None:
                self.slots[i] = (A, B)
                self.id_to_index[adapter_id] = i
                return i
        raise RuntimeError(f"No free LoRA slot (max={self.max_adapters})")

    def unload_adapter(self, adapter_id):
        if adapter_id not in self.id_to_index:
            return
        idx = self.id_to_index.pop(adapter_id)
        if self.ref_cnt[idx] == 0:
            self.slots[idx] = None

    def serve(self, tokens, adapter_ids):
        # adapter_id -> index
        indices = []
        for aid in adapter_ids:
            if aid not in self.id_to_index:
                raise RuntimeError(f"adapter {aid} not loaded")
            indices.append(self.id_to_index[aid])
        adapters_A = [self.slots[i][0] if self.slots[i] else None for i in range(self.max_adapters)]
        adapters_B = [self.slots[i][1] if self.slots[i] else None for i in range(self.max_adapters)]
        return lora_mixed_batch(tokens, self.W_base, adapters_A, adapters_B, indices, self.scaling)
"""
