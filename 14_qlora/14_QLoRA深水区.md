# 特性 #14：QLoRA 深水区 —— 量化 base + LoRA 的交互与误差传播

> 学习阶段：AI Infra / 量化方向（你的主场，量化主线收口）
> 对应源码：`vllm/lora/layers/base_linear.py:186/204/227`（量化 base + LoRA 叠加 + dtype 处理）+ `vllm/model_executor/layers/quantization/`（各量化方法的 apply）
> 本讲定位：第1~4讲量化、第11讲 LoRA 都讲过了。这一讲把它们**交叉**——QLoRA（Quantized base + LoRA）。回答量化研究者必须搞懂的：**base 是 4bit 量化的，LoRA 微调/推理时，量化误差怎么影响 LoRA？dtype 不一致怎么处理？QLoRA 训练和推理的量化粒度有何不同？** 这是你面试量化岗的差异化深水区。
> 干中学原则：本讲你要**亲手实现 mini QLoRA**——量化 base（模拟 W4 反量化）+ LoRA 叠加，实测量化误差如何传播到 LoRA 输出，对比全精度 vs 量化 base 的精度差。

---

## 一、为什么 QLoRA 是量化研究的高价值方向？（背景）

### 1.1 大模型微调的显存墙

想在 70B 模型上微调。全参数微调：base 70B（fp16，140GB）+ 优化器状态（2×70B，280GB）+ 梯度（140GB）≈ 560GB。**单卡根本不可能**。

LoRA 把可训练参数降到 r×维度（几十 MB），但 base 仍要 140GB fp16 加载（前向需要）。还是装不下。

### 1.2 QLoRA：base 量化 + LoRA 微调

QLoRA（Dettmers 2023）的洞察：**base 用 4bit 量化加载（NF4），只 LoRA 部分参与梯度/优化器**。
- base 70B → 4bit ≈ 35GB（装得下单卡）
- LoRA 参数 fp16（几十 MB），正常反向传播
- 优化器状态只针对 LoRA 参数（小）

**这是大模型单卡微调的事实标准**。你量化方向在这里的核心贡献：**NF4 量化精度、量化误差对 LoRA 训练的影响、double quantization（对 scale 再量化）**。

### 1.3 推理时的 QLoRA

训练完的 QLoRA = 4bit base + fp16 LoRA。部署时（vLLM）：
- base 走第1~4讲的量化推理路径（AWQ/GPTQ/FP8 kernel）
- LoRA 走第11讲的热加载路径
- 两者在 forward 时叠加

> 💡 面试一句话答：**QLoRA 用 4bit 量化 base（省显存，装得下单卡）+ fp16 LoRA 微调（只训小参数），前向时 base 走量化 kernel 算出 fp16 输出，LoRA 在 fp16 域叠加 B@A@x——量化对 LoRA 透明（LoRA 只看 base 的 fp16 输出，不碰量化权重），但量化误差会通过 base 输出传播到 LoRA。**

---

## 二、核心机制：量化对 LoRA 透明（base_linear.py:186/204）

看 vLLM 的 LoRA 层怎么和量化 base 协作：

```python
# base_linear.py:186
def _get_quant_method(self) -> QuantizeMethodBase:
    quant_method = self.base_layer.quant_method   # base 的量化方法（第1讲）
    return quant_method

# base_linear.py:204
def _apply_sync(self, x, bias=None):
    output = self._get_quant_method().apply(self.base_layer, x, bias)  # 量化 base 前向
    return self._apply_lora_to_output(x, output)                       # 叠加 LoRA

# base_linear.py:227
def _apply_lora_to_output(self, x, output):
    lora_output = self.punica_wrapper.add_lora_linear(
        output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, ...)
    # output = base_output(量化算的,fp16) + B@A@x(LoRA,fp16)
```

**关键洞察：LoRA 层不直接碰量化的权重**。它只调 `quant_method.apply`（base 内部反量化到 fp16 算），拿到 fp16 的 `output`，再把 fp16 的 LoRA 修正加上去。**dtype 在 fp16 域统一，没有冲突**。

这就是"量化对 LoRA 透明"——LoRA 不知道 base 是 AWQ 还是 FP8 还是 NF4，它只看到一个 fp16 的 base 输出。

---

## 三、核心难点：量化误差的传播

"透明"不等于"无影响"。base 量化的误差会**通过 output 传播到 LoRA**。

### 3.1 误差来源

`quant_method.apply` 内部：`output = dequant(W_quant) @ x`。`dequant(W_quant) ≈ W_true + ε`（ε 是量化误差）。所以：
```
output_quant ≈ (W_true + ε) @ x = W_true @ x + ε @ x = output_true + ε @ x
```
base 输出带了 `ε @ x` 的误差。

### 3.2 误差如何影响 LoRA

QLoRA 训练时，梯度通过 base（frozen）回传到 LoRA：
```
∂L/∂(B@A) = ∂L/∂output · xᵀ  （LoRA 的梯度）
```
base 的 `output` 带 `ε@x` 误差，但**梯度计算用的是 output 的局部值**。如果 ε@x 较大，LoRA 学到的 A/B 会"补偿"这个误差——某种程度上 LoRA 能**吸收部分量化误差**（这是 QLoRA 论文的发现之一）。

但推理时，如果 base 量化误差大，`output = W_quant@x + B@A@x` 里 `W_quant@x` 偏离真值，即使 LoRA 完美，总输出仍有偏差。

### 3.3 量化粒度的影响

- **per-tensor 量化**：误差大，LoRA 难补偿
- **per-channel/group 量化**（NF4 用 group_size=64）：误差小，QLoRA 精度接近全精度
- **double quantization**（QLoRA 原创对 scale 再量化）：scale 也量化省显存，但引入二阶误差

---

## 四、QLoRA 训练 vs 推理的量化差异（重要！）

这是面试常被追问的深水区：

| | 训练（QLoRA 原文） | 推理（vLLM） |
|---|---|---|
| **base 量化格式** | NF4（4bit normal float） | AWQ/GPTQ/FP8（第1~4讲） |
| **反量化时机** | 前向时实时反量化到 bf16 | kernel 内 fused（不显式反量化） |
| **LoRA dtype** | bf16 | fp16（lora_dtype） |
| **量化粒度** | NF4 group=64 + double quant | 取决于具体方法 |

**训练用的 NF4 不一定等于推理用的 AWQ**。实际部署时，常见做法：QLoRA 训练 → 合并 LoRA 到 base → 重新量化成 AWQ/GPTQ 推理。或者保留 LoRA 分离，base 用训练时的量化格式。

vLLM 的 QLoRA 推理支持 base 是任意量化格式（通过 `quant_method.apply` 抽象），LoRA 用 fp16。**两者解耦**——你可以 AWQ base + LoRA，或 GPTQ base + LoRA，或 NF4 base + LoRA。

---

## 五、把第十四讲和前十三讲连起来

| 讲次 | 关系 |
|------|------|
| 第1~4讲（量化） | QLoRA 的 base 就是这些量化方法 |
| 第11讲（LoRA） | QLoRA 的 LoRA 部分就是第11讲的热加载 |
| 第4讲（scalar_type） | NF4 是一种 float 类型，scalar_type 能描述 |
| **第14讲（QLoRA）** | **量化 + LoRA 的交叉收口** |

**第14讲是量化主线（1~4）和 LoRA（11）的交汇点**。你量化方向的全套知识（量化方法选择、误差分析、kernel）在这里都有应用。面试被问"量化模型怎么做 LoRA 微调/推理"，答案就是这一讲串联前 13 讲。

---

## 六、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：量化-LoRA 协作（基础）
1. 读 `base_linear.py:186 _get_quant_method`。它返回什么？为什么 LoRA 层能支持任意量化 base？（提示：抽象 = QuantizeMethodBase）
2. `base_linear.py:207`：`output = quant_method.apply(base_layer, x, bias)`。这个 output 是什么 dtype？为什么不是 int4？
3. `lora_config.lora_dtype`（135行）默认是什么？为什么 LoRA 不量化（保持 fp16）？

### 任务 B：误差传播（核心）
4. 假设 base 是 AWQ 4bit，量化误差 ε。写出 `quant_method.apply` 输出的表达式（含 ε）。LoRA 叠加后总输出含什么误差项？
5. QLoRA 训练时，LoRA 的梯度 `∂L/∂(B@A)` 依赖 base 的 output。如果 output 带 ε@x 误差，LoRA 学到的 A/B 会偏向什么？（提示：补偿误差）
6. NF4 的 group_size=64 vs per-tensor 量化，哪个对 QLoRA 精度更友好？为什么？

### 任务 C：训练 vs 推理（机制）
7. QLoRA 训练用 NF4，推理用 AWQ——两者数值范围/分布不同。直接把 NF4 训练的 LoRA 接到 AWQ base 推理，会有什么问题？
8. `double quantization`（QLoRA 原创）：对 scale 再量化。省了什么显存？引入了什么误差？
9. 思考题：你是量化方向，如果要提升 QLoRA 精度，你会从哪些角度入手？（提示：量化格式选择、误差补偿 LoRA、group size、LoRA rank 与量化误差的交互）

---

## 七、干中学实践任务（核心！）

> 在 `practice_qlora.py` 里实现 mini QLoRA，**实测量化误差传播**。
> 依赖：仅标准库（`random`, `math`）。不需要装 vllm/torch。
> 设计哲学：你不只实现 QLoRA 前向，还要**测量量化误差如何影响最终输出**——这是量化研究的核心实验技能。

### 实践 1：量化 base + LoRA 叠加（热身）
实现：
- `quantize_w(W, bits=4) -> (W_quant, scale)`：简单均匀量化（W / scale 取整，scale = max/scale_max）
- `dequantize_w(W_quant, scale, bits) -> W_deq`：反量化
- `qlora_forward(x, W_true, W_quant, scale, A, B, scaling, bits)`：
  - base: `y_base = dequantize(W_quant, scale) @ x`（量化 base 前向）
  - lora: `y_lora = scaling * B @ A @ x`
  - total: `y = y_base + y_lora`

验证：bits=很高（如 16，近似全精度）时，量化 base 输出 ≈ 全精度 base 输出。

### 实践 2：量化误差传播测量（核心）
对比三种配置在相同输入下的输出误差：
- **全精度**：`y_full = W_true @ x + scaling*B@A@x`（base 不量化）
- **量化 base 无 LoRA**：`y_q_noLora = dequant(W_quant) @ x`
- **QLoRA**：`y_qlora = dequant(W_quant) @ x + scaling*B@A@x`

测量：
- `err_base = ‖y_q_noLora - W_true@x‖`（base 量化误差）
- `err_qlora = ‖y_qlora - y_full‖`（QLoRA 总误差）

验证：LoRA 叠加不引入额外误差（`err_qlora ≈ err_base`，因为 LoRA 在两配置里相同）；但若训练时 LoRA 是在量化 base 上学的，推理时 base 量化方式变了，误差会增大。

### 实践 3：量化粒度 vs QLoRA 精度（进阶）
对比不同量化粒度（per-tensor vs per-group，group_size=8 vs 4）下，QLoRA 的输出误差。
实现 per-group 量化：把 W 按 group_size 切，每组独立 scale。

验证：per-group（小 group）误差显著小于 per-tensor。**这就是 NF4 用 group=64 的原因**。

> 💡 实践 2 是灵魂。要点：① 全精度的 y_full 是 ground truth ② 量化 base 的误差 ε@x 是根源 ③ LoRA 叠加是线性的，不放大 base 误差 ④ 但训练时的"误差补偿"效应本实践不模拟（需训练循环），你可以在感悟里讨论。实践 3 让你亲手看到 group_size 对精度的影响——这是你量化方向的日常实验。

---

## 八、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_qlora.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_qlora.py 运行结果，重点贴误差对比表：全精度 vs 量化base无LoRA vs QLoRA）

---

## 九、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实践 2 里 err_qlora ≈ err_base，你对"LoRA 不放大 base 量化误差"的理解？② 实践 3 里 per-group vs per-tensor 误差差异多大？这解释了 NF4 group=64 的设计？③ 训练时的"LoRA 补偿量化误差"效应，你怎么理解（LoRA 学到的 A/B 部分抵消了 ε@x）？

（完成实践后填写）

---

## 十、个人复盘感悟（留给你写）

> 你是量化方向研究生，这是你的主场深水区，建议角度：① QLoRA 的"量化对 LoRA 透明"——LoRA 只看 fp16 输出，这个抽象让你怎么设计"量化感知的 LoRA"（让 LoRA 知道 base 量化误差，主动补偿）？② NF4 vs INT4 vs FP8 作为 QLoRA base，精度/速度权衡你怎么评估？③ double quantization 对 scale 再量化，引入二阶误差，你怎么评估这个 trade-off？④ 你如果要发 QLoRA 相关论文，会从哪个角度切入（误差补偿 LoRA？自适应量化粒度？量化感知 LoRA 初始化）？

（在此写下你的感悟）


---
---

> ✅ **量化主线（1~4 + 14）完整收口**。QLoRA 是量化与 LoRA 的交汇，也是你研究方向的高价值交叉。完成后告诉我下一步：
> - **① KV offload / 多模态**（剩余未覆盖领域）
> - **② 阶段性收尾**：14 讲覆盖八大领域，可做知识图谱总结
> - **③ 继续 QLoRA 深挖**：NF4 的 normal float 分布、double quant 实现
> - 或你指定的
