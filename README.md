# vLLM 源码学习笔记

> 本仓库用于系统学习 **vLLM** 的高性能推理引擎实现。围绕源码逐特性拆解，每个主题包含一份**学习笔记**（`.md`）与配套的**实践脚本**（`practice_*.py` / `_verify_ref.py`）。
>
> 目标：吃透 vLLM 的关键机制，建立可面试、可复现的知识体系。

当前覆盖：**第 01 ~ 16 讲**（八大领域 + 量化深水区 + KV 存储管理三件套 + NF4 数学）。

---

## 📑 目录

| 讲次 | 主题 | 一句话核心 |
|------|------|-----------|
| 01 | 量化方法注册与分发机制 | 字符串名 → Config 工厂 → 按 layer 分发 Method → CPA 三步管理权重 |
| 02 | AWQ 权重打包与 Marlin 分发 | INT4 按非标准位序打包，转成标准位序后选 Marlin/Machete kernel |
| 03 | compressed-tensors 统一量化格式 | config_groups 描述任意混合精度，三级 target 匹配分发 |
| 04 | scalar_type 类型系统 | 四字段描述所有数值类型，size_bits/mask/pack_factor 全派生 |
| 05 | PagedAttention 分页 KV Cache | KV 按 block 分页，block_table 做逻辑→物理映射，显存 20%→96% |
| 06 | Continuous Batching 统一调度 | 无 prefill/decode 阶段，token_budget 统一调度，显存不足抢占 |
| 07 | Chunked Prefill 长 prompt 切块 | 长 prompt 切 chunk，靠 FlashAttention varlen 零 padding 同 batch |
| 08 | Speculative Decoding 投机解码 | draft 猜 K 个 token，大模型一次 forward 并行验证，永不吃亏 |
| 09 | Prefix Caching 深入 | 链式 hash 识别相同前缀，ref_cnt 共享保护，LRU 驱逐 |
| 10 | MoE 专家路由 | gate 打分选 top-k 专家，只算 k 个专家加权融合 |
| 11 | LoRA 热加载 | 低秩 B@A 叠加，punica 按 adapter index 分组，热加载复用 index |
| 12 | 结构化输出 | grammar 编译成 FSM，bitmask 事前屏蔽非法 token |
| 13 | Expert Parallelism | 专家分布多卡，dispatch-compute-combine 三阶段 |
| 14 | QLoRA 深水区 | 4bit 量化 base + fp16 LoRA，量化对 LoRA 透明但误差 ε@x 传播 |
| 15 | KV Cache Offload | 显存满时 KV 换出 CPU/SSD，类似 OS 页面换入换出 |
| 16 | NF4 量化 | 4bit 量化点按正态分布分位数摆放，匹配 LLM 权重分布 |

---

## 🧭 特性知识图谱

### 一、八大领域总览

vLLM 的特性可归为**性能优化（省资源/用资源/加速）+ 架构（稀疏/多路复用）+ 正确性**五个维度，共八大领域：

```
                        vLLM 特性全景
        ┌──────────────────┼──────────────────┐
   性能优化（省/用/快）      架构（稀疏复用）      正确性
    ┌────┴────┐           ┌──┴──┐             │
  量化      调度          MoE   多adapter    结构化输出
 (省权重)  (省KV+用算力+加速) (稀疏激活) (共享base)   (保证格式)
```

| 领域 | 讲次 | 解决的问题 | 一句话核心 |
|------|------|-----------|-----------|
| **量化·基础** | 01~04 | 权重显存 | 注册表工厂分发 + 打包格式 + 统一元语言 + 子字节类型 |
| **调度·显存** | 05, 09, 15 | KV cache 碎片/重复/显存墙 | 分页分配 + 链式hash共享 + 跨存储换页 |
| **调度·算力** | 06, 07 | GPU 利用率 | 统一 token_budget 调度 + 长 prompt 切块 |
| **加速** | 08 | 解码吞吐 | draft 小模型猜 + 大模型并行验证 |
| **MoE** | 10, 13 | 大模型算力/显存 | topk 稀疏路由 + 专家分布多卡 |
| **多 adapter** | 11 | 微调显存 | 低秩 B@A 叠加 + 混合 batch 共享 base |
| **正确性** | 12 | 输出格式 | grammar bitmask 从源头屏蔽非法 token |
| **量化·深水区** | 14, 16 | 量化×LoRA 交叉 + NF4数学 | 量化 base + fp16 LoRA，误差透明传播；NF4 按正态分布摆量化点 |

### 二、特性关系图（依赖与正交）

```
                    ┌──────────── 量化主线 ────────────┐
                    │                                  │
   01 注册分发 ──→ 02 AWQ打包 ──→ 03 ct统一格式 ──→ 04 scalar_type
                    │                                  │
                    └──────────────────┬───────────────┘
                                       │ (量化是地基，被各处复用)
                                       ↓
   ┌──────────── 调度主线 ────────────┐    ┌──────── MoE 主线 ────────┐
   │                                  │    │                          │
   05 PagedAttention ──┬── 09 Prefix  │    10 单卡路由 ──→ 13 多卡EP   │
       (分页分配)       │   (hash共享) │       (topk gating)  (expert_map│
                       │   ↓          │                       +all2all) │
                       │   15 Offload │                          │
                       │   (换页)     │                          │
       ↓ 使能           │              │                          │
   06 Continuous ──────┴── 07 Chunked │                          │
      Batching              Prefill    │                          │
   (统一token_budget)   (长prompt切块) │                          │
                       │              │                          │
                       ↓              │                          │
                  08 Speculative      │                          │
                   Decoding           │                          │
                  (draft+verify)      │                          │
                                       │                          │
   ┌──── 多 adapter ────┐              │                          │
   │ 11 LoRA 热加载     │←─────────────┘ (量化 base + LoRA)       │
   │  (B@A叠加+混合)    │                                         │
   └────────────────────┘              │                         │
                  │                    │                          │
                  └────→ 14 QLoRA ←────┘ (量化×LoRA 交汇收口) ───┘
                         (误差传播)

   12 结构化输出（正交，采样层事前约束，与前所有性能优化叠加）
```

**关键关系**：
- **量化是地基**（01~04）：被 MoE 专家权重、LoRA base、spec decode draft 复用
- **PagedAttention 使能一切调度**（05→06/07/08/09/15）：分页让 KV 灵活分配，才可能有 continuous batching/chunked/spec/prefix cache/offload
- **KV 存储管理三件套**（05 分页 + 09 共享 + 15 换页）：GPU 内分配 → 内容复用 → 跨存储扩展
- **MoE 单卡→多卡**（10→13）：13 是 10 的分布式扩展
- **QLoRA 是两个主线的交汇**（14 = 量化01~04 × LoRA 11）
- **结构化输出正交**（12）：和所有性能优化叠加，不冲突

### 三、面试问答索引（按问题查讲次）

| 面试问题 | 对应讲次 |
|---------|---------|
| vLLM 量化怎么实现的？ | 01~04, 14 |
| AWQ 和 GPTQ 有什么区别？ | 02, 04 |
| 为什么 vLLM 比 HF generate 快？ | 05, 06, 07 |
| PagedAttention 原理？ | 05 |
| Continuous Batching 是什么？ | 06 |
| 长 prompt 怎么处理？ | 07 |
| 怎么加速 LLM 解码？ | 08 |
| 重复 prompt 怎么优化？ | 09 |
| MoE 怎么部署？ | 10, 13 |
| 怎么服务多个微调模型？ | 11, 14 |
| 怎么保证 LLM 输出 JSON？ | 12 |
| DeepSeek 的 EP 怎么做？ | 13 |
| 量化模型怎么做 LoRA？ | 14 |
| 显存不够怎么办？ | 05, 09, 14, 15 |
| 长 context（128K+）显存不够怎么办？ | 15, 07 |

### 四、量化方向特别地图

量化在 vLLM 里形成完整闭环：

```
基础机制层                    格式层              类型层
01 注册分发 ──────→ 03 compressed-tensors ──→ 04 scalar_type
    │                                            │
    ↓ 算法实现                                   ↓ 类型派生
02 AWQ打包+Marlin                              (uint4b8/int4/fp8)
    │
    ↓ 交叉应用
14 QLoRA（量化×LoRA）── 量化 base + LoRA，误差传播
    │
    ↓ 分布式量化
13 EP 的 FP8 dispatch ── 量化省通信
10 MoE 专家量化 ── 256 专家显存压力，量化收益倍增
```

**量化在 vLLM 的五个应用点**（面试加分）：
1. 权重量化（01~04 核心）
2. QLoRA（14，量化 base + LoRA 微调/推理）
3. MoE 专家量化（10，大 MoE 显存刚需）
4. EP FP8 dispatch（13，量化省 all2all 通信）
5. spec decode draft 量化（08，便宜的 4bit 小模型 draft）

---

## 📁 仓库结构

```
study/
├── 01_quantization_overview/      # 量化基础（注册分发）
├── 02_awq_marlin/                 # AWQ 打包 + Marlin kernel
├── 03_compressed_tensors/         # compressed-tensors 统一格式
├── 04_scalar_type/                # scalar_type 类型系统
├── 05_paged_attention/            # PagedAttention 分页 KV
├── 06_continuous_batching/        # Continuous Batching
├── 07_chunked_prefill/            # Chunked Prefill
├── 08_speculative_decoding/       # Speculative Decoding
├── 09_prefix_caching/             # Prefix Caching
├── 10_moe_routing/                # MoE 专家路由
├── 11_lora_hotswap/               # LoRA 热加载
├── 12_structured_output/          # 结构化输出
├── 13_expert_parallelism/         # Expert Parallelism
├── 14_qlora/                      # QLoRA 深水区
├── 15_kv_offload/                 # KV Cache Offload
└── 16_nf4/                        # NF4 量化
```

每个主题目录下：
- `XX_*.md` — 学习笔记（核心源码、关键概念、面试要点）
- `practice_*.py` — 可运行的实践脚本
- `_verify_ref.py` — 参考验证脚本（部分主题）

---

## 🗺️ 未覆盖特性（待学习）

- [ ] 多模态（prefix cache extra_keys 的多模态展开）
- [ ] Disaggregated Prefill/Decode（PD 分离）
- [ ] Sleep mode / CuMem allocator
- [ ] Pipeline Parallelism

> 每学一个新特性，在此打勾并更新上方知识图谱。

---

## 🔧 使用

实践脚本基于纯 Python / NumPy 实现（不依赖 vLLM 源码即可运行），用于验证概念：

```bash
python 01_quantization_overview/practice_register.py
```
