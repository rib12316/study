"""
特性 #4 干中学实践：实现一个 mini ScalarType + 基于 ScalarType 的通用 pack/unpack

目标：亲手重建 vLLM 的 ScalarType 类型系统核心——
  ① 实现 ScalarType 四字段模型（exponent/mantissa/signed/bias）+ 派生属性
  ② 用 ScalarType 驱动通用 pack/unpack（从类型派生 pack_factor/mask，不硬编码）
  ③ 验证 GPTQ 的 bias 机制（uint4b8：存无符号 0~15，表带符号 -8~7）

参考 vLLM 源码：
  - scalar_type.py:22        ScalarType 类
  - quant_utils.py:461       pack_quantized_values_into_int32
  - quant_utils.py:483       unpack_quantized_values_into_int32

依赖：仅标准库。不需要装 vllm/torch。
运行：python practice_scalar_type.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
from dataclasses import dataclass


# ============================================================================
# 实践 1：实现 mini ScalarType（热身）
# ============================================================================
@dataclass(frozen=True)
class ScalarType:
    """vLLM ScalarType 的最小复刻。四字段描述任意数值类型。
    参考 scalar_type.py:22。
    """
    exponent: int          # 指数位（浮点用；整数 0）
    mantissa: int          # 尾数位（浮点）或有效位（整数，不含符号位）
    signed: bool           # 是否有符号位
    bias: int              # 偏置：stored = actual + bias

    # TODO(你): 实现以下属性和方法
    @property
    def size_bits(self) -> int:
        """总位数 = exponent + mantissa + (1 if signed else 0)。"""
        pass

    def is_floating_point(self) -> bool:
        """exponent != 0 即为浮点。"""
        pass

    def _raw_max(self) -> int:
        """无符号整数情况下的最大存储值 = (1 << mantissa) - 1。
        （本实践只处理整数类型，浮点的 max 略过）"""
        pass

    def min(self) -> int:
        """考虑 bias 的最小【实际】值 = _raw_min() - bias。
        无符号整数 _raw_min = 0。"""
        pass

    def max(self) -> int:
        """考虑 bias 的最大【实际】值 = _raw_max() - bias。"""
        pass

    def __str__(self) -> str:
        """整数类型：[u]int<size>[b<bias>]，如 uint4 / int8 / uint4b8（bias 为 0 不显示 b）。
        浮点类型：float<size>_e<exp>m<man>，如 float8_e4m3。"""
        pass

    # 工厂方法
    @classmethod
    def uint(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """无符号整数：exponent=0, mantissa=size_bits, signed=False。"""
        pass

    @classmethod
    def int_(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """有符号整数：exponent=0, mantissa=size_bits-1（去掉符号位）, signed=True。"""
        pass

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
        """IEEE754 浮点：signed=True, bias=0。"""
        pass


# ============================================================================
# 实践 2：基于 ScalarType 的通用 pack/unpack（核心）
# ============================================================================
def pack_into_int32(values: list[int], wtype: ScalarType) -> int:
    """把多个值按【标准位序】pack 进一个 int32。
    pack_factor、mask 都从 wtype 派生，不许硬编码！
    参考 quant_utils.py:461。
    注意：带符号值要先用 & mask 转成无符号（补码）再 pack。
    """
    pack_factor = 32 // wtype.size_bits      # TODO(你): 用这行，理解它
    mask = (1 << wtype.size_bits) - 1
    # TODO(你): 实现打包逻辑
    pass


def unpack_from_int32(packed: int, wtype: ScalarType) -> list[int]:
    """从一个 int32 unpack 出多个值（标准位序）。
    参考 quant_utils.py:483。
    返回的值是【存储值】（无符号 0~2^size-1），不是实际值。
    """
    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1
    # TODO(你): 实现解包逻辑
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
def verify_practice1():
    print("=== 实践 1 验证：mini ScalarType ===")
    u4 = ScalarType.uint(4, None)
    i4 = ScalarType.int_(4, None)
    u4b8 = ScalarType.uint(4, 8)
    fp8 = ScalarType.float_IEEE754(4, 3)

    assert u4.size_bits == 4, f"uint4 size_bits: {u4.size_bits}"
    assert i4.size_bits == 4, f"int4 size_bits: {i4.size_bits}"
    assert i4.mantissa == 3, f"int4 mantissa 应为 3: {i4.mantissa}"
    assert u4.mantissa == 4, f"uint4 mantissa 应为 4: {u4.mantissa}"
    assert u4.min() == 0 and u4.max() == 15
    assert u4b8.min() == -8 and u4b8.max() == 7, f"uint4b8 值域: {u4b8.min()}~{u4b8.max()}"
    assert fp8.size_bits == 8 and fp8.is_floating_point()
    assert str(u4) == "uint4", str(u4)
    assert str(i4) == "int4", str(i4)
    assert str(u4b8) == "uint4b8", str(u4b8)
    assert str(fp8) == "float8_e4m3", str(fp8)

    print(f"  uint4     : size={u4.size_bits}, 值域 {u4.min()}~{u4.max()}, mantissa={u4.mantissa}")
    print(f"  int4      : size={i4.size_bits}, mantissa={i4.mantissa} (含1符号位)")
    print(f"  uint4b8   : size={u4b8.size_bits}, 值域 {u4b8.min()}~{u4b8.max()} ← GPTQ!")
    print(f"  float8_e4m3: size={fp8.size_bits}, is_fp={fp8.is_floating_point()}")
    print("  ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：通用 pack/unpack ===")
    # uint4: 每 int32 装 8 个
    wtype = ScalarType.uint(4, None)
    v = [1, 2, 3, 4, 5, 6, 7, 8]
    assert len(v) == 32 // wtype.size_bits
    p = pack_into_int32(v, wtype)
    u = unpack_from_int32(p, wtype)
    assert u == v, f"uint4 round-trip 失败: {u} != {v}"

    # int8: 每 int32 装 4 个，带符号要转补码
    wtype2 = ScalarType.int_(8, None)
    v2 = [-4, -1, 0, 3]                      # 实际带符号值
    mask8 = (1 << 8) - 1
    v2_unsigned = [x & mask8 for x in v2]    # 补码转无符号: -4→252, -1→255, 0→0, 3→3
    p2 = pack_into_int32(v2, wtype2)
    u2 = unpack_from_int32(p2, wtype2)
    assert u2 == v2_unsigned, f"int8 unpack 应为无符号存储值: {u2} != {v2_unsigned}"

    print(f"  uint4: pack {v}")
    print(f"         -> int32 = 0x{p:08x}, unpack -> {u} ✓")
    print(f"  int8 : pack(带符号) {v2}")
    print(f"         -> 存储为无符号 {v2_unsigned}, unpack -> {u2} ✓")
    print(f"  → pack_factor/mask 全部从 ScalarType 派生，未硬编码 ✓\n")


def verify_practice3():
    print("=== 实践 3 验证：GPTQ bias 机制（uint4b8）===")
    wtype = ScalarType.uint(4, 8)            # GPTQ 4bit
    # GPTQ 量化后的实际带符号权重值
    actual = [-8, -1, 0, 3, 7]

    # bias 机制：存储值 = 实际值 + bias
    bias = wtype.bias
    stored = [a + bias for a in actual]      # [0, 7, 8, 11, 15]，全是无符号 0~15
    assert stored == [0, 7, 8, 11, 15], f"存储值错: {stored}"

    # 补齐到 pack_factor 个值（uint4 → 8 个）
    while len(stored) < 32 // wtype.size_bits:
        stored.append(0)

    # pack → unpack → 减回 bias，应还原 actual
    packed = pack_into_int32(stored, wtype)
    unpacked_stored = unpack_from_int32(packed, wtype)
    restored_actual = [s - bias for s in unpacked_stored[:len(actual)]]
    assert restored_actual == actual, f"bias round-trip 失败: {restored_actual} != {actual}"

    # 同时验证 ScalarType 的 min/max 与 bias 一致
    assert wtype.min() == min(range(wtype.min(), wtype.max()+1))  # -8
    assert wtype.max() == 7

    print(f"  GPTQ uint4b8, bias={bias}")
    print(f"  实际带符号值: {actual}")
    print(f"  存储无符号值: {stored[:len(actual)]}  ( = 实际 + {bias})")
    print(f"  pack→unpack→减bias 还原: {restored_actual} == 原始 ✓")
    print(f"  ScalarType.min()={wtype.min()}, max()={wtype.max()} ✓")
    print(f"  → 你证明了 GPTQ『存无符号+bias还原带符号』机制，且 ScalarType 正确编码 bias ✓\n")


def main():
    print("mini ScalarType 类型系统实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 vLLM ScalarType 类型系统的核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int

    @property
    def size_bits(self):
        return self.exponent + self.mantissa + int(self.signed)

    def is_floating_point(self):
        return self.exponent != 0

    def _raw_max(self):
        return (1 << self.mantissa) - 1

    def min(self):
        return 0 - self.bias            # 无符号整数 _raw_min=0

    def max(self):
        return self._raw_max() - self.bias

    def __str__(self):
        if self.is_floating_point():
            return f"float{self.size_bits}_e{self.exponent}m{self.mantissa}"
        name = ("int" if self.signed else "uint") + str(self.size_bits)
        if self.bias != 0:
            name += "b" + str(self.bias)
        return name

    @classmethod
    def uint(cls, size_bits, bias):
        return cls(0, size_bits, False, bias if bias else 0)

    @classmethod
    def int_(cls, size_bits, bias):
        return cls(0, size_bits - 1, True, bias if bias else 0)

    @classmethod
    def float_IEEE754(cls, exponent, mantissa):
        return cls(exponent, mantissa, True, 0)


def pack_into_int32(values, wtype):
    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1
    assert len(values) == pack_factor
    res = 0
    for i, v in enumerate(values):
        res |= (v & mask) << (wtype.size_bits * i)
    return res


def unpack_from_int32(packed, wtype):
    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1
    return [(packed >> (wtype.size_bits * i)) & mask for i in range(pack_factor)]
"""
