"""
特性 #1 实践任务 2：模拟 vllm 的量化方法注册表 + 工厂 + 装饰器

目标：亲手实现一次 "注册表 + 工厂 + 装饰器" 三件套，
理解 vllm/model_executor/layers/quantization/__init__.py 的核心机制。

运行：python practice_register.py
（纯标准库，不需要安装 vllm）

请按 TODO 提示补全代码，然后运行验证。参考答案在文件末尾（先自己写！）。
"""
from abc import ABC, abstractmethod


# ============================================================================
# 1. 抽象基类（对应 vllm 的 QuantizationConfig）
# ============================================================================
class QuantizationConfig(ABC):
    """所有量化方法配置的基类。"""

    @abstractmethod
    def get_name(self) -> str:
        ...


# ============================================================================
# 2. 注册表数据结构
# ============================================================================
# TODO(你): 模仿 vllm，维护两个全局结构：
#   - QUANTIZATION_METHODS: list[str]  （所有合法方法名，类似"菜单"）
#   - _CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type]  （名字 -> Config类）
# 初始化为空。

QUANTIZATION_METHODS: list[str] = []
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG: dict[str, type] = {}


# ============================================================================
# 3. 注册装饰器（对应 vllm 的 register_quantization_config）
# ============================================================================
def register_quantization_config(quantization: str):
    """装饰器：把一个 QuantizationConfig 子类注册进注册表。

    要求（模仿 vllm 的行为）：
      - 如果 quantization 已存在于 QUANTIZATION_METHODS，打印一条 debug 信息
        说明将被覆盖（vllm 用 logger.debug，这里用 print）。
      - 如果不存在，追加到 QUANTIZATION_METHODS。
      - 校验被装饰的类必须是 QuantizationConfig 的子类，否则 ValueError。
      - 写入 _CUSTOMIZED_METHOD_TO_QUANT_CONFIG。
      - 返回原类（装饰器惯例）。
    """
    def _wrapper(quant_config_cls):
        # TODO(你): 实现上述逻辑
        pass
    return _wrapper


# ============================================================================
# 4. 工厂函数（对应 vllm 的 get_quantization_config）
# ============================================================================
def get_quantization_config(quantization: str) -> type:
    """根据名字返回对应的 QuantizationConfig 子类。

    要求：
      - 若 quantization 不在 QUANTIZATION_METHODS 里，raise ValueError。
      - （真实 vllm 在这里做 lazy import 并构建一个大 dict；
         这里我们简化：直接从 _CUSTOMIZED_METHOD_TO_QUANT_CONFIG 查。）
      - 返回对应的类。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 5. 用装饰器注册两个 fake 方法
# ============================================================================
# TODO(你): 注册 my_w4a16 和 my_fp8
# @register_quantization_config("my_w4a16")
# class MyW4A16Config(QuantizationConfig):
#     def get_name(self): return "my_w4a16"
#
# @register_quantization_config("my_fp8")
# class MyFP8Config(QuantizationConfig):
#     def get_name(self): return "my_fp8"


# ============================================================================
# 6. 验证
# ============================================================================
def main():
    print("=== 注册表内容 ===")
    print("QUANTIZATION_METHODS =", QUANTIZATION_METHODS)

    print("\n=== 工厂查询 ===")
    cls1 = get_quantization_config("my_w4a16")
    print("get_quantization_config('my_w4a16') ->", cls1.__name__)
    assert cls1.__name__ == "MyW4A16Config", "查询 my_w4a16 失败"

    cls2 = get_quantization_config("my_fp8")
    print("get_quantization_config('my_fp8')   ->", cls2.__name__)
    assert cls2.__name__ == "MyFP8Config", "查询 my_fp8 失败"

    print("\n=== 覆盖测试 ===")
    # TODO(你): 再注册一个同名 my_fp8，验证覆盖行为
    # @register_quantization_config("my_fp8")
    # class MyFP8ConfigV2(QuantizationConfig):
    #     def get_name(self): return "my_fp8_v2"
    # 然后断言 get_quantization_config("my_fp8") 现在返回 V2

    print("\n=== 非法名字测试 ===")
    try:
        get_quantization_config("not_exist")
        print("错误：应该抛 ValueError")
    except ValueError as e:
        print("正确捕获 ValueError:", e)

    print("\n✅ 全部通过！")


if __name__ == "__main__":
    # 取消下面的注释来跑你的实现
    # main()
    print("请先完成 TODO，然后取消 main() 的注释运行。")


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
QUANTIZATION_METHODS = []
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    def _wrapper(quant_config_cls):
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("must be subclass of QuantizationConfig")
        if quantization in QUANTIZATION_METHODS:
            print(f"[debug] 方法 '{quantization}' 已存在，将被 {quant_config_cls.__name__} 覆盖")
        else:
            QUANTIZATION_METHODS.append(quantization)
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls
    return _wrapper


def get_quantization_config(quantization: str) -> type:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]
"""
