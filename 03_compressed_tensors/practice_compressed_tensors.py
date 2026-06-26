"""
特性 #3 干中学实践：compressed-tensors config 解析器 + target 匹配器

目标：亲手重建 vLLM 的 compressed-tensors 核心解析逻辑——
  ① parse_config_groups：把 config_groups 解析成 target → scheme 映射
  ② find_matched_target：实现精确/正则/类名三级 target 匹配
  ③ get_scheme_for_layer：端到端，给一层找出它的量化方案

参考 vLLM 源码：
  - compressed_tensors.py:297  _quantization_scheme_map_from_config
  - utils.py:113               find_matched_target
  - utils.py:175               _is_equal_or_regex_match

依赖：仅标准库（re, dataclasses）。不需要装 vllm/torch/compressed-tensors。
运行：python practice_compressed_tensors.py

设计：参考答案在文件末尾（被注释）。请先自己实现 TODO，跑通后取消注释对照。
"""
import re
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# 简化的 QuantizationArgs（对应 compressed-tensors 库的 QuantizationArgs）
# ============================================================================
@dataclass
class QuantArgs:
    """量化的参数描述。None 表示该侧不量化（如 input_activations=None 即 W?A16）。"""
    num_bits: int | None = None
    type: str | None = None              # "int" | "float"
    symmetric: bool | None = None
    strategy: str | None = None          # "tensor" | "channel" | "group" | "block"
    group_size: int | None = None
    dynamic: bool | None = None          # 仅激活用：True=动态量化

    @classmethod
    def from_dict(cls, d: dict | None) -> "QuantArgs | None":
        if d is None:
            return None
        return cls(
            num_bits=d.get("num_bits"),
            type=d.get("type"),
            symmetric=d.get("symmetric"),
            strategy=d.get("strategy"),
            group_size=d.get("group_size"),
            dynamic=d.get("dynamic"),
        )


# ============================================================================
# 实践 1：解析 config_groups → target_scheme_map（热身）
# ============================================================================
def parse_config_groups(config: dict) -> dict[str, dict]:
    """把 compressed-tensors 的 config_groups 解析成扁平的 target → scheme_dict。

    输入 config 形如（参考讲义第一节示例）:
      {"config_groups": {
          "group_0": {"targets": [...], "weights": {...}, "input_activations": {...}|null},
          ...
      }}
    输出: { target_name: {"weights": QuantArgs, "input_activations": QuantArgs|None, "format": str} }

    要求：扁平化——遍历每个 group 的 targets，把每个 target 映射到该 group 的 scheme。
    参考 compressed_tensors.py:320-366。
    """
    # TODO(你): 实现
    pass


# ============================================================================
# 实践 2：target 匹配（核心）
# ============================================================================
def _is_equal_or_regex_match(value: str, target: str, check_contains: bool = False) -> bool:
    """复刻 utils.py:175。判断 value 是否匹配 target。
    - target 以 "re:" 开头：用 re.match 匹配正则（注意 match 从头匹配！）
    - check_contains=True：target 是否是 value 的子串（大小写不敏感）
    - 否则：精确相等
    """
    # TODO(你): 实现
    pass


def _find_first_match(value: str, targets: list[str], check_contains: bool = False) -> str | None:
    """返回 targets 里第一个匹配 value 的元素，无则 None。"""
    for t in targets:
        if _is_equal_or_regex_match(value, t, check_contains):
            return t
    return None


def find_matched_target(layer_name: str, module_class_name: str,
                        targets: list[str]) -> str | None:
    """复刻 utils.py:113。三级优先匹配:
      ① layer_name 精确/正则匹配
      ② module_class_name 包含匹配（check_contains=True）
    返回匹配到的 target 字符串，无则 None。
    （融合层匹配 _match_fused_layer 本实践略过，留作选做）
    """
    # TODO(你): 实现三级匹配（前两级即可）
    pass


# ============================================================================
# 实践 3：端到端——给一层找出它的 scheme
# ============================================================================
def get_scheme_for_layer(layer_name: str, module_class_name: str,
                         config: dict) -> dict | None:
    """端到端：解析 config，再为指定 layer 找出它的量化 scheme_dict。
    不匹配（包括在 ignore 列表里）返回 None，表示该层不量化。
    """
    # TODO(你): 实现（调 parse_config_groups + find_matched_target）
    pass


# ============================================================================
# 验证（不要改）
# ============================================================================
SAMPLE_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": False,
                        "strategy": "group", "group_size": 128},
            "input_activations": None,    # → W4A16
        },
        "group_1": {
            "targets": ["re:.*lm_head"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "tensor"},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        },
    },
    "ignore": ["lm_head.language_model"],
}


def verify_practice1():
    print("=== 实践 1 验证：parse_config_groups ===")
    tsm = parse_config_groups(SAMPLE_CONFIG)
    assert "Linear" in tsm, "target 'Linear' 缺失"
    assert "re:.*lm_head" in tsm, "target 're:.*lm_head' 缺失"
    assert tsm["Linear"]["weights"].num_bits == 4, "weights.num_bits 错"
    assert tsm["Linear"]["input_activations"] is None, "W4A16 应 input=None"
    assert tsm["re:.*lm_head"]["input_activations"].dynamic is True, "lm_head 应动态量化"
    print(f"  解析出 {len(tsm)} 个 target")
    for t, s in tsm.items():
        w, a = s["weights"], s["input_activations"]
        print(f"  {t:20s} -> weights={w.num_bits}bit {w.type}, act={'None(A16)' if a is None else f'{a.num_bits}bit {a.type}'}")
    print("  ✓\n")


def verify_practice2():
    print("=== 实践 2 验证：find_matched_target 三级匹配 ===")
    targets = ["model.layers.0.self_attn.q_proj", "re:.*lm_head", "Linear"]

    # ① 精确匹配
    t = find_matched_target("model.layers.0.self_attn.q_proj", "QKVParallelLinear", targets)
    assert t == "model.layers.0.self_attn.q_proj", f"精确匹配失败: {t}"

    # ② 正则匹配（re.match 从头匹配）
    t = find_matched_target("model.lm_head", "Linear", targets)
    assert t == "re:.*lm_head", f"正则匹配失败: {t}"

    # ③ 类名包含匹配（module 类名含 "Linear"）
    t = find_matched_target("model.layers.1.mlp.gate_proj", "ReplicatedLinear", targets)
    assert t == "Linear", f"类名包含匹配失败: {t}"

    # ④ 不匹配
    t = find_matched_target("model.norm", "LayerNorm", targets)
    assert t is None, f"不应匹配: {t}"

    # ⑤ 易错点：re.match 匹配【前缀】而非全串，不锚定结尾！
    #   "re:.*lm_head" 能匹配 "model.layers.0.lm_head_extra"：
    #   .* 吃掉 "model.layers.0."，lm_head 匹配，剩余 "_extra" 不影响（match 不要求到尾）
    t = find_matched_target("model.layers.0.lm_head_extra", "SomeLayer", targets)
    assert t == "re:.*lm_head", f"re.match 前缀匹配语义错误: {t}"

    # ⑥ 反例：re.match 从头匹配，"lm_head" 出现在中间但开头不匹配则失败
    t = find_matched_target("prefix.lm_head.suffix", "SomeLayer", targets)
    # 这里 ".*lm_head" 仍能从头匹配（.* 吃 prefix.），所以仍是 re:.*lm_head
    assert t == "re:.*lm_head", f"意外: {t}"

    print("  ① 精确匹配 ✓")
    print("  ② 正则匹配 ✓")
    print("  ③ 类名包含匹配 ✓")
    print("  ④ 不匹配返回 None ✓")
    print("  ⑤ re.match 前缀语义（不锚定结尾）✓\n")


def verify_practice3():
    print("=== 实践 3 验证：get_scheme_for_layer 端到端 ===")
    # 普通 Linear → W4A16
    s = get_scheme_for_layer("model.layers.0.mlp.gate_proj", "ReplicatedLinear", SAMPLE_CONFIG)
    assert s is not None and s["weights"].num_bits == 4, f"普通 Linear 应 W4A16: {s}"

    # lm_head → W8A8
    s = get_scheme_for_layer("model.lm_head", "Linear", SAMPLE_CONFIG)
    assert s is not None and s["weights"].num_bits == 8, f"lm_head 应 W8A8: {s}"

    # LayerNorm → 不量化
    s = get_scheme_for_layer("model.norm", "LayerNorm", SAMPLE_CONFIG)
    assert s is None, f"norm 应不量化: {s}"

    print("  model.layers.0.mlp.gate_proj (ReplicatedLinear) -> W4A16 ✓")
    print("  model.lm_head (Linear)                       -> W8A8  ✓")
    print("  model.norm (LayerNorm)                       -> None  ✓")
    print("  → 你复现了 compressed-tensors 混合精度解析的核心 ✓\n")


def main():
    print("compressed-tensors config 解析实践\n" + "=" * 50 + "\n")
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
    print("🎉 全部通过！你已亲手实现 compressed-tensors 的解析核心。")


if __name__ == "__main__":
    main()


# ============================================================================
# 参考答案（自己写完再对照！）
# ============================================================================
"""
def parse_config_groups(config):
    target_scheme_map = {}
    quant_format = config.get("format")
    for gname, gcfg in config.get("config_groups", {}).items():
        targets = gcfg.get("targets", [])
        weights = QuantArgs.from_dict(gcfg.get("weights"))
        input_act = QuantArgs.from_dict(gcfg.get("input_activations"))
        fmt = gcfg.get("format", quant_format)
        for target in targets:
            target_scheme_map[target] = {
                "weights": weights,
                "input_activations": input_act,
                "format": fmt,
            }
    return target_scheme_map

def _is_equal_or_regex_match(value, target, check_contains=False):
    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False

def find_matched_target(layer_name, module_class_name, targets):
    # 三级优先：① 层名精确/正则 ② 类名包含
    return (
        _find_first_match(layer_name, targets)
        or _find_first_match(module_class_name, targets, check_contains=True)
    )

def get_scheme_for_layer(layer_name, module_class_name, config):
    tsm = parse_config_groups(config)
    ignore = config.get("ignore", [])
    if any(layer_name == ig or ig in layer_name for ig in ignore):
        return None
    matched = find_matched_target(layer_name, module_class_name, list(tsm.keys()))
    if matched is None:
        return None
    return tsm[matched]
"""
