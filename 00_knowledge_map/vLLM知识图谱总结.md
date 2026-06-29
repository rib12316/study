# vLLM 知识图谱总结（跨讲全景）

> 📌 **维护规则**：每交付一个新特性（讲），同步在本文件追加对应条目，并更新"八大领域总览"和"特性关系图"。本文件是面试复习的总纲。
>
> 📅 当前覆盖：**第 1 ~ 16 讲**（八大领域全部建立 + 量化深水区收口 + KV 存储管理三件套完整 + NF4 数学）
> 🆕 最近新增：第 16 讲 NF4 量化（量化主线数学收口）

---

## 一、八大领域总览

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

---

## 二、特性关系图（依赖与正交）

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

---

## 三、逐讲速查卡（面试复习用）

### 📘 第 01 讲：量化方法注册与分发机制
- **一句话**：字符串名 → Config 工厂 → 按 layer 分发 Method → CPA 三步管理权重
- **核心源码**：`quantization/__init__.py:108`（get_quantization_config）、`base_config.py`（QuantizationConfig/QuantizeMethodBase）、`config/model.py:970`（override 责任链）
- **关键概念**：注册表工厂、override 责任链（GPTQ→auto_gptq 接管）、CPA 生命周期（Create/Process/Apply）、Lazy import
- **面试要点**：vLLM 量化是"策略模式+工厂模式"；override 用 classmethod（看 checkpoint）、get_quant_method 用实例方法（看 layer）；UnquantizedLinearMethod 也实现接口（Null Object 模式）
- **实践产物**：`practice_register.py`（注册表+工厂+装饰器三件套）

### 📘 第 02 讲：AWQ 权重打包 + Marlin 分发
- **一句话**：INT4 按非标准位序 `[0,4,1,5,2,6,3,7]` 打包，转换成标准位序后选 Marlin/Machete kernel
- **核心源码**：`auto_awq.py:72`（_REVERSE_AWQ_PACK_ORDER）、`auto_awq.py:92`（_convert_awq_to_standard_format）、`kernels/linear/__init__.py:648`（choose_mp_linear_kernel）
- **关键概念**：AWQ pack order（槽位j→标准序reverse[j]，逆置换用于pack）、pack 在输出维→输入维转换、kernel 责任链（按算力+can_implement 探测）
- **面试要点**：pack order 是 AWQ/GPTQ 不兼容的根源；Marlin 要求 pack 在输入维（kernel 友好）；choose_mp_linear_kernel 是 capability probe + fallback chain
- **实践产物**：`practice_awq_pack.py`（手写 pack/unpack + AWQ↔标准转换 + 矩阵级向量化）

### 📘 第 03 讲：compressed-tensors 统一量化格式
- **一句话**：用 config_groups（targets/weights/input_activations）描述任意混合精度，三级 target 匹配分发到 W4A16/W8A8 等
- **核心源码**：`compressed_tensors.py:297`（_quantization_scheme_map_from_config）、`utils.py:113`（find_matched_target）、`compressed_tensors.py:684`（_get_scheme_from_parts 决策树）
- **关键概念**：target_scheme_map（扁平化）、三级 target 匹配（精确/正则re:/类名包含）、不靠 override 抢权而是做"产出标准"
- **面试要点**：compressed-tensors 是"量化即配置"理想；`re:` 用 re.match（从头匹配不锚定结尾，易错点）；它没重写 override，靠 quant_method 精确匹配进工厂
- **实践产物**：`practice_compressed_tensors.py`（config 解析 + target 匹配 + 端到端）

### 📘 第 04 讲：scalar_type 类型系统
- **一句话**：四字段 (exponent/mantissa/signed/bias) 描述所有数值类型，size_bits/mask/pack_factor 全派生
- **核心源码**：`scalar_type.py:22`（ScalarType）、`scalar_type.py:327`（scalar_types 常量）、`quant_utils.py:461`（pack_values_into_int32）
- **关键概念**：int4 mantissa=3（含符号位）、uint4b8（bias=8，GPTQ 存无符号表带符号）、id 编码跨 Python/C++ 边界
- **面试要点**：torch.dtype 不支持 sub-byte 和 bias，scalar_type 补这两个缺口；pack_factor/mask 从类型派生（类型即配置）；GPTQ 用 uint4b8 解决"存无符号、表带符号"
- **实践产物**：`practice_scalar_type.py`（mini ScalarType + 通用 pack/unpack + GPTQ bias 验证）

### 📘 第 05 讲：PagedAttention 分页 KV Cache
- **一句话**：KV 按 block_size=16 分页，block_table 做逻辑→物理映射，双向链表空闲队列管理，显存利用率 20%→96%
- **核心源码**：`kv_cache_utils.py:179`（FreeKVCacheBlockQueue 哨兵双向链表）、`block_pool.py:144`（BlockPool）、`kv_cache_manager.py:244`（allocate_slots）
- **关键概念**：分页（OS 虚拟内存类比）、block_table、双向链表（为 C++ 移植不用 deque）、cdiv 元运算、优雅降级（不够返回 None 不 OOM）
- **面试要点**：消除内外碎片；哨兵节点省判空；显存不足返回 None 让调度器不调度（绝不 OOM）；借鉴 OS 分页
- **实践产物**：`practice_paged_attention.py`（BlockPool + 双向链表 + block_table + 抢占）

### 📘 第 06 讲：Continuous Batching 统一调度
- **一句话**：没有 prefill/decode 阶段，每 step 用 token_budget 统一调度（先 running decode 后 waiting prefill），显存不足抢占
- **核心源码**：`scheduler.py:388`（schedule，核心循环）、`scheduler.py:390`（"no phase"注释）、`scheduler.py:538`（抢占）、`scheduler.py:1106`（_preempt_request）
- **关键概念**：num_tokens_with_spec 统一抽象、先 running 后 waiting（保 decode 延迟）、抢占放回 waiting 头部
- **面试要点**：PagedAttention 是 continuous batching 的使能者（灵活显存让请求进出零成本）；统一抽象能覆盖 chunked/spec/prefix；队头阻塞的解法
- **实践产物**：`practice_continuous_batching.py`（两队列 + 混合调度 + 抢占）

### 📘 第 07 讲：Chunked Prefill 长 prompt 切块
- **一句话**：长 prompt 切成 chunk，每 step 算一个 chunk + decode 混合，靠 FlashAttention varlen 零 padding 同 batch
- **核心源码**：`scheduler.py:790-810`（threshold 主动切 + budget 被动切）、`config/scheduler.py:84`（默认开启）、`flash_attn.py:248`（query_start_loc/seq_lens varlen）
- **关键概念**：主动切（long_prefill_token_threshold≈4% max_len）vs 被动切（budget）、cu_seqlens 零 padding、4% 的权衡
- **面试要点**：chunked prefill 是 continuous batching 的特例；prefill/decode 能混在同一个 attention 靠 varlen；不切模式在长 prompt 会饿死（活教材）
- **实践产物**：`practice_chunked_prefill.py`（threshold chunking + 策略对比模拟器 + varlen cu_seqlens）

### 📘 第 08 讲：Speculative Decoding 投机解码
- **一句话**：draft 猜 K 个 token，大模型一次 forward 并行验证（memory-bound 下多算 K+1 位置几乎不增耗时），接受前缀+bonus，永不吃亏
- **核心源码**：`spec_decode/ngram_proposer.py:207`（N-gram KMP）、`spec_decode/metrics.py:114`（mean_acceptance_length=1+accepted/drafts）、`sampler.py:358`（spec token 组合）
- **关键概念**：五种 proposer（draft model/EAGLE/Medusa/N-gram/suffix）、accept 连续匹配前缀、bonus 保证下界=1、加速比=(1-α^(K+1))/(1-α)
- **面试要点**：memory-bound 是 spec 成立的物理基础；"永不吃亏"来自 bonus；概率拒绝采样保证输出分布不变；N-gram 对重复文本（代码）接受率高
- **实践产物**：`practice_speculative_decoding.py`（draft-verify 循环 + 加速比实测 + N-gram）

### 📘 第 09 讲：Prefix Caching 深入
- **一句话**：链式 hash（block_hash=hash(parent_hash,tokens)）识别相同前缀，ref_cnt 共享保护，LRU 驱逐 ref_cnt=0 的
- **核心源码**：`kv_cache_utils.py:577`（hash_block_tokens 链式）、`block_pool.py:199`（get_cached_block）、`block_pool.py:574`（_maybe_evict_cached_block）、`block_pool.py:597`（touch ref_cnt 保护）
- **关键概念**：链式 hash（区块链式累积，前N block同=前N×bs token同）、ref_cnt>0 绝不驱逐、extra_keys（多模态/LoRA 区分）
- **面试要点**：PagedAttention（物理分配）+ Prefix Caching（内容共享）= 完整 KV 管理；ref_cnt 类似 GC 引用计数；RAG/agent 重复 prefix 场景第二个请求吃满 cache
- **实践产物**：`practice_prefix_caching.py`（链式hash + ref_cnt共享 + LRU驱逐 + 端到端命中率）

### 📘 第 10 讲：MoE 专家路由
- **一句话**：gate 打分选 top-k 专家（softmax/sigmoid routing），只算 k 个专家 FFN 加权融合，fused kernel 按专家排序批量算
- **核心源码**：`router/fused_topk_router.py:69`（fused_topk softmax/sigmoid）、`fused_moe.py:1460`（_prepare_expert_assignment 按专家排序）、`fused_moe.py:161`（Triton kernel off_experts）
- **关键概念**：topk gating、softmax（互斥）vs sigmoid（独立）、renormalize、token 按专家排序批量（1次大GEMM vs k次小GEMM）
- **面试要点**：MoE 是"架构稀疏"（每次少算专家），量化是"数值稀疏"（每值少占位），两者正交可叠加；负载均衡靠 aux loss + round-robin
- **实践产物**：`practice_moe.py`（gate+topk + MoE前向 + 朴素vs排序对比）

### 📘 第 11 讲：LoRA 热加载
- **一句话**：低秩 B@A 叠加（参数省数量级），所有 adapter 的 A/B 预 stack，punica 按 token 的 adapter index 分组应用，热加载复用 index
- **核心源码**：`lora/layers/base_linear.py:129`（lora_a/b_stacked 预分配）、`base_linear.py:158`（set_lora）、`base_linear.py:215`（_apply_lora_to_output + punica）
- **关键概念**：ΔW=B@A 低秩分解、scaling=alpha/r、stacked 预分配（批量GEMM）、混合 batch（每token不同adapter）、index 槽位复用
- **面试要点**：MoE 是模型内部多路复用（专家），LoRA 是模型外部多路复用（adapter共享base）；punica 分组计算和 MoE 排序批量同一思想；B初始化0保证训练开始ΔW=0
- **实践产物**：`practice_lora.py`（单LoRA叠加 + 多adapter混合batch + 热加载生命周期）

### 📘 第 12 讲：结构化输出
- **一句话**：grammar 编译成 FSM，每步算 bitmask 标记合法 token，采样前非法 logit 置 -inf，事前约束非事后过滤
- **核心源码**：`structured_output/__init__.py`（_fill_bitmasks + grammar_bitmask）、`backend_xgrammar.py:136`（XgrammarGrammar fill_bitmask/accept_tokens/rollback）、`sampler.py`（masked_fill -inf）
- **关键概念**：bitmask（白名单）、grammar FSM 状态机、accept 推进状态、token-level vs char-level mismatch（xgrammar tokenize 感知）、四种后端
- **面试要点**：事前屏蔽 vs 事后过滤；spec decode 的 draft 要 grammar 验证（accept+rollback）；FSM 状态命名要精确（EXPECT_COLON 陷阱）
- **实践产物**：`practice_structured_output.py`（数字FSM + bitmask约束采样 + JSON键值生成）

### 📘 第 13 讲：Expert Parallelism
- **一句话**：专家分布多卡（expert_map 全局→本地映射），token dispatch 到目标卡 all2all、本地算、combine 回原卡，round_robin 负载更均
- **核心源码**：`expert_map_manager.py:22`（determine_expert_map 分片）、`all2all_utils.py`（dispatch/compute/combine）、`fused_moe.py:161`（off_experts==-1 跳过）
- **关键概念**：EP vs TP（按专家切 vs 按权重切）、expert_map（-1=不在本卡）、dispatch-compute-combine 三阶段、contiguous vs round_robin 放置、FP8 dispatch（量化省通信）
- **面试要点**：MoE 稀疏性让"按专家切"通信少；round_robin 负载均衡优于 contiguous；use_fp8_dispatch 是量化省通信的应用；EP×TP 两级切分
- **实践产物**：`practice_expert_parallelism.py`（expert_map + dispatch/compute/combine + 负载对比）

### 📘 第 14 讲：QLoRA 深水区
- **一句话**：4bit 量化 base + fp16 LoRA，前向 base 走量化 kernel 出 fp16、LoRA 在 fp16 域叠加，量化对 LoRA 透明但误差 ε@x 传播
- **核心源码**：`lora/layers/base_linear.py:186`（_get_quant_method 透明）、`base_linear.py:207`（quant_method.apply 量化base前向）、`base_linear.py:227`（add_lora_linear fp16叠加）
- **关键概念**：量化对 LoRA 透明（LoRA 只看 fp16 输出）、误差传播（err_qlora≈err_base，LoRA不放大）、per-group 误差 < per-tensor（NF4 g=64 依据）、double quantization、训练用NF4≠推理用AWQ
- **面试要点**：量化主线收口；LoRA 能部分补偿量化误差（训练时）；量化感知 LoRA 是可发论文方向；QLoRA 是大模型单卡微调事实标准
- **实践产物**：`practice_qlora.py`（量化base+LoRA + 误差传播实测 + 量化粒度对比）

### 📘 第 16 讲：NF4 量化
- **一句话**：NF4 把 16 个 4bit 量化点按正态分布分位数摆放（0附近密、两端稀），匹配 LLM 权重的正态分布，比均匀INT4精度高
- **核心源码**：`bitsandbytes_loader.py:433-436`（vLLM调quantize_4bit nf4）+ bitsandbytes NF4常量 + QLoRA论文
- **关键概念**：数据感知量化（按权重分布摆点）、NF4的16个正态分位数常量、归一化适配[-1,1]、group_size=64局部归一化、double quantization（对scale再量化省显存）、NF4无校准 vs AWQ校准感知
- **面试要点**：NF4精度优势前提是权重正态分布（实测正态下比均匀准1.26x，均匀分布权重下优势消失）；NF4是训练(QLoRA)最优4bit，推理常转AWQ（kernel更快）；数据感知 vs 校准感知是两条量化路线
- **实践产物**：`practice_nf4.py`（均匀INT4 + NF4实现 + 正态/均匀分布MSE对比）

### 📘 第 15 讲：KV Cache Offload
- **一句话**：GPU 显存满时把不常用 KV block 换出到 CPU/SSD，需要时换回，类似 OS 页面换入换出，突破 GPU 显存墙服务长 context
- **核心源码**：`kv_offload/base.py:168`（OffloadingManager 策略层 lookup/touch/prepare_load/store）、`base.py:450`（OffloadingWorker 机制层 submit_store/load/get_finished）、`factory.py`（GPU→CPU→SSD 多级 tier）
- **关键概念**：两级抽象（Manager 策略 + Worker 机制）、多级 tier（GPU→CPU→SSD）、异步换入换出（submit+poll，CUDA stream 通信计算 overlap）、LRU 驱逐、offload_prompt_only、预取、thrashing
- **面试要点**：KV 存储管理三件套（05 分页 + 09 共享 + 15 换页）完整；offload 是"跨存储层级的 prefix cache"；异步是可行的关键（同步会阻塞 GPU）；局部性差时 thrashing（频繁换入换出）——实测局部性好 GPU 命中率 90% vs 差 0%
- **实践产物**：`practice_kv_offload.py`（两级存储+LRU + 异步换入换出 + 命中率/局部性测量）

---

## 四、面试问答索引（按问题查讲次）

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

---

## 五、量化方向特别地图（你的主场）

你研究生方向是大模型量化，vLLM 里量化相关的知识形成完整闭环：

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

## 六、未覆盖特性（待学习，按需追加）

> 每学一个，在此打勾并补充到第三/四节。

- [x] **第 15 讲：KV Cache Offload**（GPU↔CPU↔SSD 分层换页，长 context）—— ✅ 已交付
- [ ] 多模态（prefix cache extra_keys 的多模态展开）
- [x] **第 16 讲：QLoRA 深挖 NF4**（NormalFloat 正态分位数量化点 + double quant）—— ✅ 已交付
- [ ] Disaggregated Prefill/Decode（PD 分离）
- [ ] Sleep mode / CuMem allocator
- [ ] Pipeline Parallelism

---

## 七、更新日志

| 日期 | 更新内容 |
|------|---------|
| 初版 | 建立知识图谱，覆盖 01~14 讲（八大领域 + 量化深水区） |
| +第15讲 | 追加 KV Cache Offload；调度·显存领域扩展为 05+09+15（KV 存储管理三件套完整）；关系图加 15 节点；面试索引加"长 context 显存"问题 |
| +第16讲 | 追加 NF4 量化；量化·深水区扩展为 14+16（NF4 是 QLoRA 量化数学核心）；量化主线完整收口 |

<!-- 维护规则：每交付新讲，在第三节追加"逐讲速查卡"条目，更新第一节总览表、第二节关系图、第四节面试索引、第六节打勾、本日志。 -->
