# 特性 #13：Expert Parallelism（EP）—— MoE 的专家怎么分布到多卡

> 学习阶段：AI Infra 基础储备 / 分布式 MoE（第10讲 MoE 的多卡延伸）
> 对应源码：`vllm/model_executor/layers/fused_moe/expert_map_manager.py`（专家分片 + expert_map）+ `fused_moe/all2all_utils.py`（dispatch/combine 通信）+ `fused_moe/fused_moe.py:1385`（expert_map 参数）
> 本讲定位：第10讲讲了单卡 MoE 的路由与 fused 计算。但 DeepSeek-V3 有 256 个专家、671B 参数——**单卡放不下**。这一讲回答：**专家怎么分到多卡？token 路由到别的卡的专家怎么办？怎么通信？** 这是大 MoE 部署的核心，面试高频（"DeepSeek 的 EP 怎么做"）。
> 干中学原则：本讲你要**亲手实现一个 mini EP**——专家分片（expert_map）+ 跨卡 token 路由（dispatch/compute/combine）+ 放置策略（round_robin vs contiguous）。

---

## 一、为什么需要 EP？（背景）

### 1.1 单卡的显存墙

DeepSeek-V3：256 个专家，每个专家是 ~2.6B 参数的 FFN。总 MoE 参数 ~671B。**单张 H100（80GB）根本放不下全部专家权重**（即使 4bit 量化也要 ~170GB）。

必须把专家分布到多卡。这就是 **EP（Expert Parallelism，专家并行）**。

### 1.2 EP vs TP：两种切法

| 切法 | 怎么切 | 通信 | 适用 |
|------|--------|------|------|
| **TP（Tensor Parallel）** | 把每个专家的权重矩阵按行/列切到多卡 | 每层 allreduce | 稠密模型常用 |
| **EP（Expert Parallel）** | 把不同**整个专家**分到不同卡 | all2all（按需） | MoE 专用 |

EP 的核心洞察：**MoE 的稀疏性让"按专家切"比"按权重切"更高效**。因为每个 token 只用 k 个专家，EP 下大部分计算在本地完成，只有路由到远程专家的 token 需要通信。

> 💡 面试一句话答：**EP 把 MoE 的不同专家分布到多卡（每卡持有部分专家），token 按路由结果 dispatch 到持有目标专家的卡、本地计算、combine 回原卡——利用 MoE 稀疏性（每 token 只用 k 个专家）减少通信，是大 MoE 部署的核心。**

---

## 二、核心机制①：专家分片与 expert_map（expert_map_manager.py:22）

### 2.1 均匀分片

`determine_expert_map`（22行）把 `global_num_experts` 个专家均匀分到 `ep_size` 张卡：

```python
base = global_num_experts // ep_size       # 每卡至少 base 个
remainder = global_num_experts % ep_size   # 余数
local_num_experts = base + 1 if ep_rank < remainder else base  # 前几个 rank 多分一个
```

例：256 专家、EP=8：每卡 32 个，均匀。13 专家、EP=4：base=3, remainder=1 → 卡0得4个，卡1/2/3各得3个。

### 2.2 expert_map：全局→本地映射

```python
expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
# 本卡的专家位置填本地 index，其他填 -1
expert_map[start_idx : start_idx+local_num_experts] = arange(0, local_num_experts)
```

`expert_map[global_id]`：
- `= local_id`：这个全局专家在本卡，本地编号 local_id
- `= -1`：不在本卡

这是 EP 的"路由表"。前向时（fused_moe.py:161）：
```python
off_experts = tl.load(expert_ids_ptr + pid_m)
if off_experts == -1:
    # 不在本卡，写零跳过
```

### 2.3 两种放置策略

`_determine_placement_strategy`（417行）+ `_init_round_robin_expert_routing_tables`（477行）：

- **Contiguous（连续）**：专家 0,1,2,3 在卡0；4,5,6,7 在卡1…… 通信局部性好但负载可能不均。
- **Round-robin（轮转）**：专家 0,4,8,12 在卡0；1,5,9,13 在卡1…… 负载更均衡（每个 token 的 top-k 更可能均匀分散到各卡）。

DeepSeek 倾向 round-robin（配合负载均衡训练）。

---

## 三、核心机制②：Dispatch-Compute-Combine（all2all_utils.py）

EP 下，token 路由到的专家可能不在本卡。三阶段处理：

### 3.1 Dispatch（分发）

每个 token 看自己的 topk_ids（第10讲），把 token 的 hidden state **发往**持有目标专家的卡。这用 **all2all** 通信（每卡发给所有其他卡）。

`all2all_utils.py` 的 dispatch 参数（如 `num_dispatchers=world_size`）配置通信。

### 3.2 Compute（本地计算）

每卡收到发给自己的 token（都是路由到本地专家的），用第10讲的 fused MoE kernel 算。**只算本地专家**（expert_map 过滤掉 -1）。

### 3.3 Combine（汇总）

计算结果 all2all 发回原卡，原卡把同一 token 的多个专家输出按 gate 权重加权融合（第10讲的 reduce）。

```
原卡0: token_A 路由到 expert2(卡0)、expert5(卡1)
  Dispatch: token_A 发往卡0（本地expert2）和卡1（expert5）
  卡0算expert2, 卡1算expert5
  Combine: 两卡结果发回卡0，加权融合
```

### 3.4 FP8 Dispatch（你量化方向的直接应用！）

`all2all_utils.py:194 use_fp8_dispatch`：**dispatch 阶段可以把 token 的 hidden state 用 FP8 量化传输**。all2all 是通信瓶颈（token 数据大），FP8 让通信量减半。这是"量化省通信"的典型场景——你研究方向的高价值应用。

---

## 四、EP 的工程挑战

1. **负载不均**：如果所有 token 都路由到卡0的专家，卡0成为瓶颈，其他卡闲置。训练加 aux loss 平衡；推理靠 round-robin 放置 + 调度。
2. **通信开销**：dispatch/combine 的 all2all 是 EP 的主要通信成本。FP8 dispatch、通信计算 overlap（用 CUDA stream）是优化点。
3. **EP × TP 组合**：可以同时 EP（专家切）+ TP（每个专家内部切）。DeepSeek 用 EP+TP 混合。
4. **EEP（Expert EPipeline）**：`eep_reconfigure.py`——动态调整专家分布（运行时根据负载迁移专家），vLLM 的新特性。

---

## 五、把第十三讲和前十二讲连起来

| 讲次 | 关系 |
|------|------|
| 第10讲（MoE） | 第13讲是它的多卡版——单卡路由→多卡分布 |
| 第1~4讲（量化） | EP 的 dispatch 可用 FP8（use_fp8_dispatch），专家权重可量化（显存省→更多专家放得下） |
| 第5讲（PagedAttention） | EP 的 token 在卡间流动，KV cache 仍在原卡（只 expert 计算跨卡） |
| **第13讲（EP）** | **MoE 的分布式扩展，通信是核心** |

**量化 + EP 是大 MoE 部署的黄金组合**：量化让单卡放更多专家（减少 EP 卡数），EP 让多卡协同服务超大 MoE。你量化方向在这两个维度都能贡献。

---

## 六、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：专家分片（基础）
1. 读 `expert_map_manager.py:22 determine_expert_map`。13 个专家、EP=4，每卡分几个？expert_map 在卡0是什么样的（哪些位置非 -1）？
2. `expert_map[global_id] = -1` 表示什么？前向 kernel（fused_moe.py:161）遇到 -1 怎么处理？
3. round_robin 和 contiguous 两种放置策略的区别？为什么 round-robin 对负载均衡更好？

### 任务 B：通信（核心）
4. 读 `all2all_utils.py`。dispatch/compute/combine 三阶段分别做什么？dispatch 的 all2all 传的是什么数据？
5. `use_fp8_dispatch`（194行）的作用？为什么 dispatch 阶段量化能省通信？（结合你量化方向）
6. 思考题：EP 下，一个 token 路由到的 k 个专家如果全在别的卡，它的 hidden state 要发几次？（提示：去重——同一目标卡只发一次）

### 任务 C：负载与组合（机制）
7. EP 负载不均的后果是什么？训练时怎么缓解（aux loss）？推理时靠什么（round-robin 放置）？
8. EP × TP 组合：EP 把专家切到卡，TP 把每个专家的权重再切。这种"两级切分"相比纯 EP 或纯 TP，省什么、加什么？
9. 思考题：你量化方向，如果把 DeepSeek 的专家全量化到 4bit，EP 的卡数能减多少？（算：671B 4bit ≈ 170GB，单卡 80GB → 还是要多卡，但卡数减半）

---

## 七、干中学实践任务（核心！）

> 在 `practice_expert_parallelism.py` 里实现一个 mini EP。
> 依赖：仅标准库。不需要装 vllm/torch。
> 设计哲学：你用纯 Python 模拟多卡（每"卡"是一个对象），实现专家分片 + 跨"卡"token 路由（dispatch/compute/combine）。这是 EP 算法的亲手实现。

### 实践 1：expert_map 生成（热身）
实现 `make_expert_map(global_num_experts, ep_size, ep_rank, strategy) -> list[int]`：
- 返回 `[global_num_experts]` 的列表，本卡专家填 local_id，其他填 -1
- 支持 round_robin 和 contiguous 两种策略

验证：13 专家、EP=4、rank=0，contiguous 策略应 `[0,1,2,3,-1,-1,...,-1]`（前4个）；round_robin 应 `[0,-1,-1,-1,1,-1,-1,-1,2,...]`（0,4,8 非-1）。

### 实践 2：Dispatch-Compute-Combine（核心）
模拟 EP=4、每卡持部分专家，一批 token 路由：
- `EPWorker` 类：持有自己的 expert_map + 本地专家权重
- `dispatch(tokens, topk_ids, all_workers) -> dict[rank, tokens]`：按 topk_ids 把 token 分发到目标卡（去重：同卡只发一次）
- `compute_local(worker, tokens, topk_ids)`：本地算路由到本地专家的部分
- `combine(local_results, topk_weights) -> final`：汇总加权融合

验证：构造 EP=2、4专家、几个 token，检查每个 token 的 hidden state 被正确 dispatch 到持有目标专家的卡、本地算、combine 回来加权融合正确。

### 实践 3：负载均衡观察（进阶）
实现 `measure_load_balance(tokens, topk_ids, ep_size, strategy) -> dict`：
- 统计每卡收到的 token 数（dispatch 后）
- 对比 round_robin vs contiguous 放置策略下的负载分布

验证：构造偏斜的路由（某些专家被高频选），观察 contiguous 下某卡过载、round_robin 下更均衡。

> 💡 实践 2 是灵魂。要点：① dispatch 按 topk_ids 决定目标卡（查 expert_map 的反查：global_id→哪个 rank）② 同一 token 发往多卡（k 个专家可能在不同卡）但每卡只发一次 ③ compute 只算本地专家 ④ combine 按原 topk_weights 加权。这模拟了真实 EP 的 all2all 数据流。

---

## 八、常见疑问（Q&A）

> 这一节记录学习过程中的真实疑问与解答，帮助澄清 EP 和 MoE 的常见误解。

### Q1：expert_map 存在哪里？是每个卡都有吗？

**是的，每个 EP 卡都有一份自己的 expert_map，但内容不同。**

`expert_map_manager.py:183` 的 `ExpertMapManager.__init__` 在**每个 GPU 进程**里实例化一次，每个进程调 `determine_expert_map` 传入**自己的 ep_rank**，生成**自己那份** expert_map。

- 每份 expert_map 的**形状都是 `[global_num_experts]`**（全局专家总数），不是 `[local_num_experts]`
- `expert_map[global_id] = local_id`：这个全局专家**在我这卡**，本地编号 local_id
- `expert_map[global_id] = -1`：这个全局专家**不在我这卡**

**为什么每卡存全局长度的 map？** 前向时（`fused_moe.py:161`）Triton kernel 拿到 token 路由到的 `global_expert_id`，要 O(1) 查"在不在本卡、本地编号多少"。全局长度的 map 直接索引即可，不用搜索。

**举例**：8 个全局专家，EP=2：

| | expert_map 内容（长度 8） |
|---|---|
| 卡0（持专家 0,1,2,3） | `[0, 1, 2, 3, -1, -1, -1, -1]` |
| 卡1（持专家 4,5,6,7） | `[-1, -1, -1, -1, 0, 1, 2, 3]` |

> 注意 `expert_map_manager.py:63`：`ep_size==1`（单卡）时返回 `None`——没有分片概念，不需要映射。

---

### Q2：两种放置策略的区别究竟是什么？为什么会有这种特性？

源码（`expert_map_manager.py:75-87`）：`linear`（=contiguous 连续）和 `round_robin`（轮转）。用 **8 专家、EP=2** 举例：

**linear（连续）**：全局专家 ID 连续聚在一起
```
卡0: 专家 0,1,2,3   卡1: 专家 4,5,6,7
```

**round_robin（轮转）**：全局专家 ID 交错散开
```
卡0: 专家 0,2,4,6   卡1: 专家 1,3,5,7
```

**为什么有两种？关键在于负载均衡。**

MoE 里每个 token 选 top-k 个专家。"负载均衡"指这些选择是否均匀落在各卡。

**场景**：训练时模型学到的路由偏好是"专家 0、1 被高频选中"。
- **linear**：专家 0、1 都在卡0 → 高频选择全压卡0 → 卡0 过载、卡1 闲置
- **round_robin**：专家 0 在卡0、专家 1 在卡1 → 高频选择分散到两卡 → 负载均衡

**直觉**：round_robin 把"相邻专家"分散到不同卡，即使路由有局部偏好（连续几个专家被高频选），也不会全压一卡。linear 把相邻专家聚一起，路由偏好易导致单卡过载。

**为什么会有这个特性**：
1. 训练的 aux loss 试图让路由均匀，但不可能完美——总有热门专家。round_robin 是**推理侧兜底**：即使路由不完美，物理放置也把热门专家打散。
2. linear 理论上通信更局部（连续专家在一卡），但实测路由通常 top-k 不连续，round_robin 的均衡收益 > linear 的局部性收益。
3. DeepSeek 等大 MoE 默认 round_robin。

> 第13讲实践 3 实测过：偏斜路由下 linear `{0:10,1:0}` 全压卡0，round_robin `{0:10,1:10}` 均衡。

---

### Q3：是一个卡专门负责分发吗？

**不是。每张卡都参与分发，是"全互联"（all-to-all），没有专门的"分发卡"。**

**误解**：卡0 是 master，收集所有 token，按路由分发给其他 worker 卡。**不是这样。**

**真相**：每张卡同时既是发送方又是接收方（`all2all_utils.py` 的 dispatch/compute/combine）：
1. 每卡都有自己 token 的一份 hidden state
2. 每卡看自己 token 的路由结果，判断该发往哪些卡
3. **所有卡同时 all-to-all**：每卡把自己该发的 token 发往目标卡，同时接收别的卡发给自己的
4. 每卡对收到的 token（路由到本地专家的）做本地 expert 计算
5. 反向 all-to-all（combine）：结果发回原卡加权融合

**为什么不用主从？**
1. **单点瓶颈**：master 卡承担 N 倍流量，成为瓶颈
2. **延迟**：主从串行化严重，all-to-all 并行度高
3. **硬件原语**：NCCL/IB 直接提供 all-to-all，硬件级优化

> EP 的通信是**对等的 all-to-all**（每卡既发又收，无中心），不是主从分发。

---

### Q4：为什么不提前确定好路由，让每卡直接收到它需要的，不用再分发？

**因为路由结果在"运行时"才能算出来，无法提前确定。**

router（gate）是线性层，路由分数 = `gate(hidden_state)`。你**无法提前知道**一个 token 会被路由到哪些专家，因为：
1. 路由取决于 hidden state 的具体数值
2. hidden state 是**前一层刚算出来的**——不可能在前一层之前就知道
3. 所以某一层的路由结果，**只能等那一层开始时才算**

**"提前确定"为什么不行**：设想在模型开头就规划好 token 全程的专家路径。做不到——第2层用哪些专家取决于第1层算出的 hidden state，第1层又取决于第1层选的专家……这是**动态依赖链**，每步依赖上一步的实际计算结果。

**所以分发是路由计算之后的必然结果**：每层开始 → 算路由 → 发现"我的 token 要去别卡的专家" → 这时才分发。无法提前。

> 唯一例外：多轮对话/重复 prompt 时，前几层路由可能和历史相同（hidden state 相同 → 路由相同），可缓存历史路由"预测"分发。但这是优化，且只对重复输入有效。

---

### Q5：8张卡，token 会复制8份到每卡吗？

**不会。token 只发给路由到的那些卡（top-k 个专家所在的卡），其他卡根本收不到。**

**错误想象**：每个 token 复制 8 份，每卡一份，各卡挑自己要的算。
**实际**：每个 token 只发给 top-k 个专家所在的卡。

**举例**：8卡、top-k=2，token A 路由到专家5（卡1）、专家12（卡3）：
```
token A 只发往: 卡1 和 卡3
卡0,2,4,5,6,7: 完全收不到 token A，不参与计算
```
token A 只产生 **2 份拷贝**，不是 8 份。如果 top-k 都在本卡，**完全不产生跨卡通信**。

**all-to-all 的"全互联"不等于"全复制"**：all-to-all 是通信原语的 capability（每卡都能给每卡发），不代表每个 token 真发给所有卡。实际 dispatch 只对路由到的目标卡发数据。

> 如果开了数据并行（DP），不同卡处理不同请求（batch 切分），每卡本来就有自己的 token（切分非复制）。DP+EP 叠加时，token "属于某卡（DP），但计算时可能临时去别卡（EP）"。

---

### Q6：MoE 是端到端选一次专家吗？每层能选不同专家吗？

**这是最常见的误解。MoE 不是整个模型选一次专家，而是每一层独立的 FFN 都可以选不同的专家。**

**MoE 替换的是 Transformer 里的 FFN 层。** 标准 Transformer 有 L 层，每层 = `[Attention] → [FFN]`。MoE 把**某些层的 FFN 换成"MoE 层"**（一个 router + N 个专家 FFN）。

**关键：每一层的 MoE 独立路由。** 同一个 token：
- 第1层可能选专家 {3, 7}
- 第2层可能选专家 {1, 5}
- 第3层可能选专家 {2, 9}
- ……每层重新路由，选的专家可以完全不同

**"前一层算完的 hidden states"** = 上一层 MoE/Attention 算完后输出的 token 向量 `h_{i-1}`，作为下一层 MoE 的输入。Transformer 逐层串行：
```
输入 embedding
  ↓ 第1层 Attention + MoE（选专家、算）→ hidden state h₁
  ↓ 第2层 Attention + MoE（选专家、算）→ hidden state h₂
  ↓ ... 每层独立路由 ...
```

**举例**：8层 MoE、4专家、top-k=2、EP=2（专家0,1在卡0，2,3在卡1）：
```
token "Hello" 逐层流过：
第1层: router 选 {专家0, 专家2}  → hidden state 到 卡0(专家0) + 卡1(专家2) 算
第2层: router 选 {专家1, 专家3}  → 和上一层完全不同！到 卡0(专家1) + 卡1(专家3)
第3层: router 选 {专家0, 专家1}  → 两个都在卡0，不用发给卡1
... 每层独立路由 ...
```

**每层 token 流动方向都不同**，取决于那一层 router 选了哪些专家。这就是为什么不能"提前一次性确定 token 该去哪卡"——每一层都在变。

### 修正后的 MoE 心智模型

> **MoE 是把 Transformer 的某些 FFN 层换成"专家层"。token 的 hidden state 逐层流动，每到一个 MoE 层，这一层的 router 现场决定它去哪 top-k 个专家，算完融合，得到新的 hidden state 传给下一层。每一层的选择独立、动态，无法提前规划。EP 下，每层的 dispatch/combine 按那一层的路由临时跨卡流动。**

---

### 六问串联

```
每卡都有自己的 expert_map（Q1）
        ↓ 决定"哪些专家在我这"
        
放置策略（Q2）决定"全局ID怎么映射到卡"
  round_robin 把热门专家打散 → 负载均衡
        
每层独立路由（Q6）→ 路由运行时才算（Q4）→ 按需发给目标卡（Q5，不全复制）
        ↓
所有卡对等 all-to-all（Q3，无 master）
```

---

## 九、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_expert_parallelism.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_expert_parallelism.py 运行结果，重点贴 dispatch/compute/combine 流程和负载对比）

---

## 十、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现 expert_map 后，你对"全局专家→本地专家"映射的理解？② dispatch/compute/combine 三阶段，你对"稀疏性让通信减少"的体会（每 token 只发 k 个目标卡）？③ round_robin vs contiguous 负载对比，差异明显吗？

（完成实践后填写）

---

## 十一、个人复盘感悟（留给你写）

> 你是量化方向研究生、AI Infra 求职者，建议角度：① EP 的 dispatch 用 FP8 量化（use_fp8_dispatch）——量化不只是省显存，还省通信，你怎么评估这个方向的价值？② 量化 + EP 组合：量化让单卡放更多专家（减 EP 卡数），EP 让多卡协同——你量化方向在"减少 EP 通信"和"减少 EP 卡数"两个维度都能贡献，你怎么看？③ DeepSeek 的 round-robin 放置 + 负载均衡训练，这种"训练-推理协同"的设计哲学你怎么看？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**MoE 主线（第10讲单卡 + 第13讲多卡）完整**。完成后告诉我下一步：
> - **① KV offload / 多模态**
> - **② 回量化**：QLoRA 深水区 / FP8 dispatch 的量化细节
> - **③ 阶段性收尾**：13 讲已覆盖 7 大领域，可做知识图谱总结
> - 或你指定的
