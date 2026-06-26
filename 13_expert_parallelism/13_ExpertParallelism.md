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

## 八、任务答卷区

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

## 九、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实现 expert_map 后，你对"全局专家→本地专家"映射的理解？② dispatch/compute/combine 三阶段，你对"稀疏性让通信减少"的体会（每 token 只发 k 个目标卡）？③ round_robin vs contiguous 负载对比，差异明显吗？

（完成实践后填写）

---

## 十、个人复盘感悟（留给你写）

> 你是量化方向研究生、AI Infra 求职者，建议角度：① EP 的 dispatch 用 FP8 量化（use_fp8_dispatch）——量化不只是省显存，还省通信，你怎么评估这个方向的价值？② 量化 + EP 组合：量化让单卡放更多专家（减 EP 卡数），EP 让多卡协同——你量化方向在"减少 EP 通信"和"减少 EP 卡数"两个维度都能贡献，你怎么看？③ DeepSeek 的 round-robin 放置 + 负载均衡训练，这种"训练-推理协同"的设计哲学你怎么看？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**MoE 主线（第10讲单卡 + 第13讲多卡）完整**。完成后告诉我下一步：
> - **① KV offload / 多模态**
> - **② 回量化**：QLoRA 深水区 / FP8 dispatch 的量化细节
> - **③ 阶段性收尾**：13 讲已覆盖 7 大领域，可做知识图谱总结
> - 或你指定的
