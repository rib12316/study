# 特性 #15：KV Cache Offload —— 显存不够时把 KV 换到 CPU/SSD

> 学习阶段：AI Infra 基础储备 / 长 context 服务（调度显存的延伸）
> 对应源码：`vllm/v1/kv_offload/base.py`（OffloadingManager/OffloadingWorker 抽象 + 多级 tier）+ `vllm/v1/kv_offload/factory.py`（GPU→CPU→SSD 组装）+ `vllm/v1/simple_kv_offload/`
> 本讲定位：第5讲 PagedAttention 解决"KV 在 GPU 显存里的碎片"，第9讲 Prefix Caching 解决"KV 的复用"。但当**长 context 把 GPU 显存彻底塞满**（比如 128K context 的请求），还能怎么办？答案：把不常用的 KV block 换出到 CPU 内存/SSD，需要时换回。这是 OS 虚拟内存"换页"思想在 LLM KV cache 上的应用，也是长 context 服务的关键。
> 干中学原则：本讲你要**亲手实现一个 mini KV offload**——GPU/CPU 两级存储、LRU 驱逐、异步换入换出模拟、命中率测量。

---

## 一、为什么需要 KV Cache Offload？（背景）

### 1.1 长 context 的显存墙

考虑一个 128K context 的请求。它的 KV cache（Llama-70B 级别）可能要 10~40GB。一个 GPU 80GB 显存，塞几个长 context 请求就满了。但 GPU 算力还远没用完——**显存成了吞吐的瓶颈，而不是算力**。

朴素解法：
- **更少并发**：只服务 1-2 个长请求 → GPU 算力闲置
- **拒绝长请求**：用户体验差

### 1.2 Offload 的思路：借 CPU/SSD 当"扩展显存"

服务器除了 GPU 显存，还有大量 **CPU 内存**（几百 GB）甚至 **SSD**（几 TB）。这些比 GPU 慢，但容量大、便宜。

**KV Cache Offload**：把暂时不用的 KV block 从 GPU 换出到 CPU/SSD，GPU 显存腾出来服务更多请求；当某个请求需要那些 KV 时，再换回 GPU。

这是经典 **OS 页面换入换出（swap）** 思想：
- GPU 显存 = 物理内存（快、小）
- CPU 内存 = swap 空间（慢、大）
- KV block = 内存页

> 💡 面试一句话答：**KV Cache Offload 把不常用的 KV block 从 GPU 换出到 CPU/SSD（借大容量慢存储扩展显存），需要时换回，类似 OS 页面换入换出；vLLM 用多级 tier（GPU→CPU→SSD）+ 异步换入换出（不阻塞 GPU 计算）+ LRU 驱逐，让长 context 服务突破 GPU 显存墙。**

---

## 二、核心抽象：OffloadingManager + OffloadingWorker（base.py）

vLLM 把 offload 拆成**调度器侧管理**和**worker 侧执行**两层：

### 2.1 OffloadingManager（base.py:168，scheduler 侧）

```python
class OffloadingManager(ABC):
    def lookup(self, key, req_context) -> LookupResult:
        # 查 block 是否已 offload 且 ready（HIT/MISS）
    def prepare_load(self, keys, req_context) -> LoadStoreSpec:
        # 准备从 offload 媒介换入 GPU（保护这些 block 不被驱逐）
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput:
        # 准备换出（返回被驱逐的 keys）
    def touch(self, keys, req_context):
        # 标记访问（更新 LRU，防驱逐）
    def complete_load(self, keys, req_context):
        # 换入完成，解除保护
```

Manager 在 scheduler 进程里运行，**只跟踪元数据**（哪些 block 在哪级、何时访问），不碰实际数据。

### 2.2 OffloadingWorker（base.py:450，worker 侧）

```python
class OffloadingWorker(ABC):
    def submit_store(self, ...) -> int:
        # 异步 GPU → offload 媒介（CPU/SSD）
    def submit_load(self, ...) -> int:
        # 异步 offload 媒介 → GPU
    def get_finished(self) -> list[TransferResult]:
        # 查哪些异步任务完成
    def wait(self, job_ids):
        # 阻塞等特定任务
```

Worker 在 GPU 进程里，**实际搬数据**（用 CUDA stream 异步拷贝）。

**两层分离的意义**：Manager 决定"换谁、何时换"（策略），Worker 执行"怎么搬"（机制）。策略和机制解耦，可以换不同的存储后端（CPU/SSD/远端）而不改调度逻辑。

### 2.3 多级 Tier（factory.py）

```python
# GPU → CPU → SSD 三级，每级一个 Manager+Worker
tier_gpu  ←→ tier_cpu ←→ tier_ssd
  (快/小)    (中/中)    (慢/大)
```

一个 block 可能先从 GPU 换到 CPU（快），CPU 满了再换到 SSD（慢）。`OffloadingManager` 的注释（base.py:294）提到"Managers that cascade to lower tiers should delay those tiers' calls"——级联时要协调各级时序。

---

## 三、关键机制①：异步换入换出（不阻塞 GPU）

这是 offload 可行的核心。如果换入换出是同步的，GPU 要等 CPU↔GPU 拷贝完成才能继续算——白等几十微秒，吞吐暴跌。

vLLM 的做法（`OffloadingWorker.submit_store/submit_load`）：
1. **submit**：提交一个异步拷贝任务（用独立 CUDA stream），立即返回 job_id，不等完成
2. GPU 继续做别的计算（attention 前向等）
3. **get_finished**：后续某个 step 检查"哪些异步拷贝完了"
4. 完成的 block 标记为 ready，可被使用

**通信计算 overlap**：拷贝和计算并行，GPU 不空等。这和第13讲 EP 的 all2all overlap 是同一个优化思想。

> 💡 异步的代价：换入需要"提前"。如果一个 block 本 step就要用，但还没换回 GPU，只能同步等（cache miss penalty）。所以 offload 要**预取**（predictive loading）——预测哪些 block 即将被用，提前异步换入。

---

## 四、关键机制②：LRU 驱逐与 touch 保护

GPU 显存满时，要驱逐一些 block 腾位。驱逐策略 = **LRU**（最久未访问的先走）。

`OffloadingManager.touch`（base.py:207）：每次访问一个 block，更新它的"最近访问时间"。驱逐时选最旧的。

**和第9讲 Prefix Caching 的 ref_cnt 的区别**：
- Prefix Cache 的 ref_cnt：**引用计数**，>0 绝不驱逐（有人在用）
- Offload 的 LRU：**时间戳**，即使没人用，只要最近访问过就不急着驱逐

两者可以叠加：ref_cnt>0 的绝不驱逐（强保护），ref_cnt=0 的按 LRU 排序（弱保护）。

---

## 五、关键机制③：何时换出？何时换入？

这是 offload 策略的核心问题：

### 5.1 换出（GPU→CPU）时机
- **显存压力大**：GPU 块池快满，主动换出 LRU 的 block 腾位
- **请求完成**：请求结束后，它的 KV 可能还被别的请求（prefix cache）复用，换到 CPU 保留
- **offload_prompt_only**（base.py:512）：只 offload prefill 阶段的 block（decode 阶段的 block 频繁更新，offload 不划算）

### 5.2 换入（CPU→GPU）时机
- **请求需要**：某请求要访问一个在 CPU 的 block，触发换入
- **预取**：预测即将访问的 block，提前异步换入（避免同步等待）

### 5.3 命中率 = 节省的 prefill

换入命中 = block 还在 CPU（没被 SSD 驱逐也没丢），直接换回 GPU，不用重算 prefill。命中率越高，省的计算越多。这和第9讲 prefix cache 的命中率概念一致——**offload 本质是"更大、更慢的 prefix cache"**。

---

## 六、Offload 的代价与边界

1. **延迟**：CPU↔GPU 拷贝有延迟（PCIe 带宽 ~30GB/s，比 GPU 内部带宽低一个量级）。频繁换入换出会拖慢。
2. **SSD 更慢**：SSD 随机读写延迟毫秒级，只适合"很久不用"的 block。
3. **预取难度**：预测哪些 block 即将被用很难（取决于请求模式）。预测错了要么白搬（浪费带宽），要么 miss（同步等）。
4. **和 prefix cache 的关系**：offload 是"跨存储层级的 prefix cache"。CPU 里的 block 也能被 hash 查到（第9讲），命中就换回。

---

## 七、把第十五讲和前十四讲连起来

| 讲次 | 关系 |
|------|------|
| 第5讲（PagedAttention） | 第15讲是它的"跨存储延伸"——第5讲管 GPU 内 block 分配，第15讲管 GPU↔CPU 的 block 流动 |
| 第9讲（Prefix Cache） | 第15讲是"更大更慢的 prefix cache"——CPU/SSD 层也能 hash 命中 |
| 第6讲（Continuous Batching） | offload 让更多请求能同时驻留（显存墙突破），提升并发 |
| 第7讲（Chunked Prefill） | 长 prompt 切块 + offload 配合，处理超长 context |
| **第15讲（KV Offload）** | **长 context 服务的显存扩展机制** |

**第5讲（分页）+ 第9讲（共享）+ 第15讲（换页）= 完整的 KV cache 存储管理**，从 GPU 内到跨存储层级。面试被问"长 context 显存不够怎么办"，答案就是这三讲串联。

---

## 八、代码阅读任务（必做）

> 在 `D:\code\vllm` 源码完成，答案写进"任务答卷区"。

### 任务 A：两层抽象（基础）
1. 读 `kv_offload/base.py:168 OffloadingManager` 和 `base.py:450 OffloadingWorker`。为什么把 offload 拆成 Manager（scheduler 侧）和 Worker（worker 侧）两层？各自负责什么？
2. `OffloadingManager` 的 `lookup` 返回 `LookupResult`（base.py:56）。HIT/MISS 分别表示什么？MISS 时怎么办？
3. `touch`（base.py:207）和第9讲的 `ref_cnt` 保护机制有什么区别？为什么 offload 用 LRU（时间戳）而 prefix cache 用 ref_cnt（计数）？

### 任务 B：异步与多级（核心）
4. 读 `OffloadingWorker.submit_store/submit_load`（base.py:456/462）。为什么是异步的（submit + get_finished）？同步会怎样？
5. `factory.py` 组装多级 tier（GPU→CPU→SSD）。一个 block 从 GPU 换出，最终可能在哪？换回时怎么知道它在哪级？
6. `offload_prompt_only`（base.py:512）默认 True。为什么只 offload prefill block，不 offload decode block？（提示：decode block 频繁更新）

### 任务 C：策略与交互（机制）
7. 什么时候换出（GPU→CPU）？什么时候换入（CPU→GPU）？预取为什么重要？
8. offload 和 prefix cache（第9讲）怎么协作？CPU 里的 block 能被 hash 查到吗？
9. 思考题：如果 CPU 内存也满了，block 要换到 SSD。SSD 随机读写延迟毫秒级，这对延迟敏感的服务意味着什么？怎么缓解？

---

## 九、干中学实践任务（核心！）

> 在 `practice_kv_offload.py` 里实现一个 mini KV offload。
> 依赖：仅标准库（`collections`, `time`）。不需要装 vllm/torch。
> 设计哲学：你用纯 Python 模拟 GPU/CPU 两级存储（两个 dict）+ LRU 驱逐 + 异步换入换出（用"待完成队列"模拟）。

### 实践 1：两级存储 + LRU 驱逐（热身）
实现 `KVOffloader`：
- `gpu_cache: dict[block_id, data]`（小容量，如 4 个 block）
- `cpu_cache: dict[block_id, data]`（大容量，如 16 个 block）
- `gpu_lru: list[block_id]`（GPU 的 LRU 顺序）
- `access(block_id) -> data`：访问 block
  - 在 GPU：直接返回，更新 LRU
  - 在 CPU：换入 GPU（若 GPU 满则 LRU 驱逐一个到 CPU），返回
  - 都没有：MISS（返回 None 或模拟重算）

验证：GPU 容量 4，访问 5 个不同 block，第 5 个应触发驱逐（最旧的换出）。

### 实践 2：异步换入换出（核心）
扩展 `KVOffloader`，支持异步：
- `submit_load(block_id) -> job_id`：提交异步换入（加入 pending 队列）
- `submit_store(block_id) -> job_id`：提交异步换出
- `poll() -> list[job_id]`：模拟"过了一段时间"，返回已完成的 job
- `is_ready(block_id)`：block 是否换入完成可用
- 访问未 ready 的 block 要等待（同步 fallback）

验证：submit 几个 load，poll 前访问应等待/失败，poll 后可访问。

### 实践 3：命中率测量 + 局部性观察（进阶）
实现 `simulate_access_pattern(accesses, gpu_cap, cpu_cap) -> dict`：
- 跑一个访问序列，统计：GPU 命中率、CPU 命中率（换入）、MISS 率
- 对比**局部性好**（重复访问少数 block）vs **局部性差**（随机访问）的命中率

验证：局部性好的序列 GPU 命中率高（少换入换出）；局部性差的频繁换入换出（thrashing）。

> 💡 实践 2 是灵魂。要点：① 异步 = submit 立即返回 + 后续 poll 查完成 ② 换入未完成时访问要等（cache miss penalty）③ 真实系统用 CUDA stream 异步拷贝，你用队列模拟。这让你体会"通信计算 overlap"的价值——同步会阻塞，异步让 GPU 不空等。

---

## 十、任务答卷区

> 代码阅读 A/B/C 答案写这里。实践代码放 `practice_kv_offload.py`，跑通后贴输出。

### 任务 A
（在此作答）

### 任务 B
（在此作答）

### 任务 C
（在此作答）

### 实践任务输出
（贴 practice_kv_offload.py 运行结果，重点贴命中率对比和异步等待演示）

---

## 十一、学习过程与实践总结（一起填）

> 完成实践后写 3-5 句。示例方向：① 实践异步换入换出后，你对"通信计算 overlap"的理解？② 局部性好 vs 差的命中率差异多大？这解释了为什么 offload 对"重复访问"的长对话友好？③ LRU 驱逐的 thrashing（频繁换入换出）你观察到了吗？怎么缓解？

（完成实践后填写）

---

## 十二、个人复盘感悟（留给你写）

> 你是 AI Infra 求职者，建议角度：① KV offload 借鉴 OS 页面换入换出，这种"跨领域迁移系统设计"你怎么看？还有哪些 OS 概念能借（预取、工作集、belady 最优）？② 异步换入换出 + 预取，这种"用预测换延迟"的思路你在哪见过（CPU cache prefetch？分支预测）？③ offload 对长 context / agent（超长对话）的价值你怎么评估？④ 量化方向：offload 的 KV 也能量化（FP8 KV cache），量化 + offload 组合能进一步扩容，你怎么评估？

（在此写下你的感悟）


---
---

> ✅ 本讲结束。**KV cache 存储管理三件套（第5讲分页 + 第9讲共享 + 第15讲换页）完整**。完成后告诉我下一步（或等命令）：
> - 多模态 / QLoRA 深挖（NF4）/ PD 分离 / sleep mode
> - 或其他
