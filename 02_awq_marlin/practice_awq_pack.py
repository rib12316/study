"""
特性 #2 干中学实践：AWQ INT4 权重打包格式转换

目标：亲手实现 INT4 的 pack/unpack 和 AWQ 非标准位序转换，
     用 vLLM 源码（auto_awq.py:_convert_awq_to_standard_format）一模一样的算法验证。
     只有能复现 vLLM 的行为，才算真正理解了"数据布局"。

【核心认知】_REVERSE_AWQ_PACK_ORDER = [0,4,1,5,2,6,3,7] 的语义
  ——它是"AWQ 槽位 j 里存的是第几个标准序值"的索引表：
    槽位0 存标准序第0个值，槽位1 存标准序第4个值，槽位2 存标准序第1个值...
  即 AWQ 格式 int32 的 8 个槽位（bit[0:4]..bit[28:32]），槽位 j 装的是
  标准序值 reverse[j]。
  · unpack AWQ：取出 8 个槽位的值，按 reverse 重排，得到标准序。
  · pack 成 AWQ：标准序值 v_i 要放进满足 reverse[j]==i 的那个槽位 j（即逆置换）。

依赖：torch（CPU 版即可）、标准库。不需要 GPU、不需要装 vllm。
运行：python practice_awq_pack.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO 部分，跑通后取消注释对照。
"""
import torch

PACK_FACTOR = 8          # 32 // 4：一个 int32 装 8 个 4bit 值
MASK = 0xF               # 4bit 掩码
# AWQ 的非标准 pack 顺序（来自 auto_awq.py:76）：槽位j → 标准序值 reverse[j]
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
# 它的逆置换：标准序值i → 进哪个AWQ槽位
_INV_AWQ_PACK_ORDER = [0, 0, 0, 0, 0, 0, 0, 0]
for _j, _i in enumerate(_REVERSE_AWQ_PACK_ORDER):
    _INV_AWQ_PACK_ORDER[_i] = _j
# 结果 _INV_AWQ_PACK_ORDER == [0, 2, 4, 6, 1, 3, 5, 7]


# ============================================================================
# 实践 1：标准 INT4 pack/unpack（热身）
# ============================================================================
def pack_standard(values: list[int]) -> int:
    """把 8 个 0~15 的值按【标准顺序】pack 成一个 int32。
    标准顺序：v_i 放在 bit [4*i : 4*i+4]，即 v0|v1<<4|v2<<8|...
    """
    assert len(values) == PACK_FACTOR
    # TODO(你): 实现
    pass


def unpack_standard(packed: int) -> list[int]:
    """逆操作：从一个 int32 取出 8 个 4bit 值（标准顺序）。"""
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 2：AWQ 位序 pack/unpack + 转换（核心）
# ============================================================================
def pack_awq_order(values: list[int]) -> int:
    """把 8 个【标准序】值 pack 成 AWQ 格式的 int32。
    规则：标准序值 values[i] 要放进 AWQ 槽位 _INV_AWQ_PACK_ORDER[i]。
    """
    assert len(values) == PACK_FACTOR
    # TODO(你): 实现
    pass


def unpack_awq_order(packed: int) -> list[int]:
    """从一个 AWQ 格式 int32 取出 8 个值，还原成【标准序】。
    规则：AWQ 槽位 j 装的是标准序值 _REVERSE_AWQ_PACK_ORDER[j]。
    """
    # TODO(你): 实现
    pass


def awq_to_standard(packed_awq: int) -> int:
    """把一个 AWQ 格式 int32 转成标准格式 int32。
    要求：和 vLLM auto_awq.py 的位序修正逻辑等价。
    提示：先 unpack_awq_order 得标准序，再 pack_standard。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 3：矩阵级转换（进阶）
# ============================================================================
def convert_awq_qweight_matrix(qw_awq: torch.Tensor) -> torch.Tensor:
    """把 AWQ 格式的 qweight 矩阵转成标准（Marlin 友好）格式。

    输入:  qw_awq  shape [K, N//8], int32, AWQ 位序, pack 在输出维 N
    输出:  new_qw  shape [K//8, N], int32, 标准位序, pack 在输入维 K

    要求：用向量化操作（broadcasting），不要逐 int32 的 for 循环。
    这是 auto_awq.py:112-126 的复刻。
    """
    K, N_packed = qw_awq.shape
    N = N_packed * PACK_FACTOR
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    reverse_order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)

    # TODO(你): 按下面步骤实现（参考讲义第四节）
    # 1. unpack: (K, N//8) -> (K, N//8, 8) 每个值是 4bit，再按 reverse 修位序 -> 标准序
    # 2. reshape 成 (K, N)
    # 3. 沿输入维重新 pack 成 (K//8, N)
    pass


# ============================================================================
# 验证函数（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：标准 pack/unpack ===")
    v = [1, 2, 3, 4, 5, 6, 7, 8]
    p = pack_standard(v)
    u = unpack_standard(p)
    assert u == v, f"round-trip 失败: {u} != {v}"
    assert all(0 <= x <= 15 for x in u), "值超出 4bit 范围"
    assert (p >> 0) & 0xF == 1
    assert (p >> 4) & 0xF == 2
    print(f"  pack_standard({v}) = 0x{p:08x}")
    print(f"  unpack -> {u}  ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：AWQ 位序转换 ===")
    v = [1, 2, 3, 4, 5, 6, 7, 8]
    pa = pack_awq_order(v)
    ua = unpack_awq_order(pa)
    assert ua == v, f"AWQ round-trip 失败: {ua} != {v}"
    ps = pack_standard(v)
    converted = awq_to_standard(pa)
    assert converted == ps, (
        f"位序转换与标准 pack 不一致!\n"
        f"  pack_standard(v)      = 0x{ps:08x}\n"
        f"  awq_to_standard(pack) = 0x{converted:08x}\n"
        f"  说明你的 awq_to_standard 和 vLLM 的位序修正不等价"
    )
    print(f"  原始 values      : {v}")
    print(f"  pack_standard    = 0x{ps:08x}")
    print(f"  pack_awq_order   = 0x{pa:08x}")
    print(f"  awq_to_standard  = 0x{converted:08x}  == pack_standard ✓")
    print(f"  → 你的位序转换与 vLLM _convert_awq_to_standard_format 等价 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：矩阵级转换 ===")
    torch.manual_seed(0)
    K, N = 256, 128       # K, N 都要是 8 的倍数
    raw = torch.randint(0, 16, (K, N), dtype=torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)

    # 1) 模拟 AWQ 格式矩阵 (K, N//8)：raw 的【标准序】值按 AWQ 位序 pack 到输出维 N
    inv = torch.tensor(_INV_AWQ_PACK_ORDER, dtype=torch.long)
    awq_vals = raw.reshape(K, N // PACK_FACTOR, PACK_FACTOR)[:, :, inv]
    qw_awq = (awq_vals << shifts[None, None, :]).sum(dim=-1, dtype=torch.int32)
    assert qw_awq.shape == (K, N // PACK_FACTOR)

    # 2) 用你的函数转成标准格式
    qw_std = convert_awq_qweight_matrix(qw_awq)
    assert qw_std.shape == (K // PACK_FACTOR, N), f"shape 错: {qw_std.shape}"

    # 3) 从标准格式 unpack 回 (K, N)，应 == raw
    #    qw_std (K//8, N) 每 int32 装【K方向连续8个】标准序值
    unpacked = (qw_std.unsqueeze(-1) >> shifts) & 0xF   # (K//8, N, 8)
    unpacked = unpacked.permute(0, 2, 1).reshape(K, N)  # (K, N)
    assert torch.equal(unpacked, raw), "矩阵转换 round-trip 失败"
    print(f"  输入 AWQ 格式: shape {tuple(qw_awq.shape)}, int32, pack 在输出维 N")
    print(f"  输出标准格式: shape {tuple(qw_std.shape)}, int32, pack 在输入维 K")
    print(f"  round-trip 还原原始 4bit 值 ✓")
    print(f"  → 你复现了 vLLM _convert_awq_to_standard_format 的核心逻辑 ✓\n")


def main():
    print("AWQ INT4 打包格式转换实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已经亲手实现了 vLLM AWQ 权重格式转换的核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def pack_standard(values):
    p = 0
    for i, v in enumerate(values):
        p |= (v & 0xF) << (4 * i)
    return p

def unpack_standard(packed):
    return [(packed >> (4 * i)) & 0xF for i in range(PACK_FACTOR)]

def pack_awq_order(values):
    # 标准序值 values[i] 进 _INV_AWQ_PACK_ORDER[i] 号 AWQ 槽位
    p = 0
    for i, v in enumerate(values):
        slot = _INV_AWQ_PACK_ORDER[i]
        p |= (v & 0xF) << (4 * slot)
    return p

def unpack_awq_order(packed):
    # AWQ 槽位 j 装的是标准序值 _REVERSE_AWQ_PACK_ORDER[j]
    slots = [(packed >> (4 * j)) & 0xF for j in range(PACK_FACTOR)]
    out = [0] * PACK_FACTOR
    for j in range(PACK_FACTOR):
        out[_REVERSE_AWQ_PACK_ORDER[j]] = slots[j]
    return out

def awq_to_standard(packed_awq):
    return pack_standard(unpack_awq_order(packed_awq))

def convert_awq_qweight_matrix(qw_awq):
    K, N_packed = qw_awq.shape
    N = N_packed * PACK_FACTOR
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    reverse_order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
    # unpack + 修 AWQ 位序 -> 标准序
    unpacked = (qw_awq.unsqueeze(-1) >> shifts) & 0xF   # (K, N//8, 8)
    unpacked = unpacked[:, :, reverse_order]            # 修正位序
    unpacked = unpacked.reshape(K, N)                   # (K, N) 标准序
    # 沿输入维重新 pack
    unpacked = unpacked.reshape(K // PACK_FACTOR, PACK_FACTOR, N)
    new_qw = (unpacked.to(torch.int32) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32)
    return new_qw.contiguous()
"""
