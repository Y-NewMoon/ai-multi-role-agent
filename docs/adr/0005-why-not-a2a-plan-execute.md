# ADR-0005：为什么不上 A2A / plan-execute（架构边界）

- 状态：**已接受**（V2.5 收尾，基于 V2.4 `209371a` 证据）
- 时间：2026-10-23
- 关联：ADR-0002（LangGraph）、ADR-0004（群聊可见性）、ADR-0009（世界书）、ADR-0010（Skill/MCP）；V2.6 选项 A（任务型群聊 plan-execute）
- 影响范围：`src/app/agents/supervisor.py`、`src/app/agents/graph.py`、`src/app/services/group_service.py`；V2.6 规划

---

## 1. Context

系统要同时满足三件事：

1. **多角色同场**：用户一条消息，群里多个角色按顺序发言，每人保持独立人设、独立记忆、独立世界书。
2. **强隔离**：角色 A 的私聊秘密不能泄漏给角色 B；用户消息与公屏发言按 `visibility=group` 扇出到每个角色自己的 namespace（ADR-0001 / ADR-0004 的不变量）。
3. **单进程、可观测、可测试**：当前部署是一个 FastAPI 进程 + 一个 PostgreSQL；每轮每个发言人写一条 trace，全部有单测与评测覆盖（V2.4：490 tests，`209371a`）。

在这个背景下，"多角色"有三种常见做法：

- **A2A / 自由发消息**：agent 之间通过协议或共享消息总线互相调用；
- **plan-execute**：先由一个 planner 产出计划，再分派给 worker 执行、汇总；
- **supervisor-workers**：一个调度者按硬约束选择下一个发言人，每个 worker 是同一张 LangGraph 图的独立实例。

现状是第三种：`Supervisor`（[supervisor.py](file:///d:/Work/File/Ai-agent/src/app/agents/supervisor.py)）只做"下一位是谁"的决策；每个角色用 `build_graph()` 各起一张图（[graph.py](file:///d:/Work/File/Ai-agent/src/app/agents/graph.py#L88)）；群聊的扇出/可见性在 `GroupService`（[group_service.py](file:///d:/Work/File/Ai-agent/src/app/services/group_service.py)）。

## 2. Decision

**多角色采用 supervisor-workers；agent 之间不做 A2A、不自由发消息；日常角色扮演不走 plan-execute。**

### 2.1 为什么是 supervisor-workers

- **调度是"选人"，不是"协商"**。群聊的产品语义是"按顺序轮流说话"，不是"agent 之间谈判"。`Supervisor` 的产出是一个 `agent_id`，且受硬约束包裹：一轮内每人最多一次、连续同人强制换人、最多 `GROUP_MAX_TURNS` 次、解析失败退化为按成员顺序轮询（README「群聊」）。这些约束用一段确定性代码就能保证"永不卡死"，不需要协议层。
- **每个 worker 必须是独立图实例**。记忆召回、世界书检索、工具白名单、人设 prompt 都按 `agent_id` 走；`build_graph()` 是工厂而非单例（ADR-0002），才能让 N 个角色共享同一套节点实现、又各自持有依赖。
- **可见性由写端扇出 + 读端过滤保证**，不靠 agent 自觉。若允许 agent 之间自由发消息，等于在隔离墙上开一个洞：B 收到 A 的私聊内容后，无法证明它不会写进 B 的 namespace（ADR-0004 的冻结决策）。
- **一份状态、一份存储**。checkpointer 与 Store 同库同进程，角色间"传话"只是一次函数调用；A2A 的序列化、路由、重试、鉴权在这里没有收益。

### 2.2 为什么日常角色扮演不上 plan-execute

- 角色扮演的输入是**逐步反应**：用户说一句，角色回一句（可能连发 2–3 条）。没有稳定的"目标—子任务—依赖"结构，planner 只能把一句话拆成"理解—检索—回复"，这是图里已经固定的事情。
- plan-execute 会把**一轮拆成多次串行 LLM 调用**：规划 1 次 + 每个子任务 1 次 + 汇总 1 次。当前完整图在 fake 轻量栈下的 20×20 p95 首 token 已经是 1457ms（`r1b-clean`）；真实 bge 路径的早期对照是 6118ms（manifest `git_sha=462f193`、`dirty=true`，**不是 `209371a` 证据**，只作量级参考）；再加规划与汇总，延迟和成本都不可接受。
- 失败面会变大：计划解析失败、子任务超时、部分成功如何汇总，都要新增状态与重试语义；而现有 supervisor 已经有"解析失败退化为轮询"的兜底。
- 可观测性会变差：现在"每个发言人一条 trace"能直接解释一轮群聊；plan-execute 需要新增 plan/task 两级 trace，当前 trace schema（`trace_version=1.0`）没有这两级。

### 2.3 什么条件下才值得上 plan-execute（V2.6 选项 A）

只在**长任务、有明确交付物、可拆成并行子任务**时值得，例如"帮我策划一场茶会 → 查资料 / 写文案 / 排流程 → 汇总"。触发条件（同时满足才立项）：

1. 单轮任务需要 ≥ 3 个可独立完成的子任务，且子任务之间有依赖；
2. 用户期望的是**一份交付物**（方案、清单、文案），不是逐句对话；
3. 可以接受首字延迟从秒级升到十秒级，并为此单独设预算；
4. 作为**独立支线**（V2.6 demo）实现，用 LangGraph 子图 + `Send` 并行，不改日常角色扮演的主图。

不满足以上条件时，仍用 supervisor-workers + 工具循环（ADR-0010 的 `should_continue → tools → generate`，最多 3 轮）。

## 3. Alternatives considered

| 方案 | 优点 | 为什么不选 |
|---|---|---|
| **A2A / 自由发消息**（agent 间协议、共享消息总线） | 概念上"角色可以互相商量"；跨进程扩展直观 | 当前是单进程、同库、同 Store，协议只增加序列化/路由/重试/鉴权与失败面；自由发消息会绕过 ADR-0004 的读端可见性墙；多一跳网络就多一个不可观测的失败点。跨进程/多租户是**未来触发条件**，不是当前需求。 |
| **CrewAI / AutoGen** | 现成的 role/crew 抽象，上手快 | 需要 per-agent 持久化 checkpointer 与向量 Store，且要能按角色注入不同的世界书/工具/记忆；LangGraph 的 `BaseStore` 与 `checkpointer` 是一等公民（ADR-0002），换框架要重做隔离与恢复语义。 |
| **plan-execute** | 长任务结构清晰，可并行 | 日常对话没有稳定计划结构；每轮多 2 次 LLM 往返；新增计划/子任务两级状态与失败语义；当前延迟余量不支持（见 §2.2）。只作为 V2.6 独立 demo 的候选。 |
| **纯顺序链**（固定 A→B→C 轮询，无 supervisor） | 最简单、零决策成本 | 无法处理"这个话题只该某角色接""同一人不要连发""一轮该结束"；产品上会显得机械。supervisor 的价值就是**有约束的选人**，不是自由协商。 |
| **supervisor-workers（选中）** | 与现有图/Store/可见性墙同构；调度可测试、可兜底；每发言人一条 trace | 代价见 §4：串行延迟、N+1 次决策、trace 生命周期复杂。 |

## 4. Consequences

**正面**

- 隔离不变量不动：A2A 不存在，就没有"跨角色消息"这条绕过路径；`golden.isolation_leaks=0` 仍由现有评测守住。
- 调度可测：`parse_decision` / `choose_next` / 硬约束都有单测（`tests/test_supervisor.py`），群聊 trace 有 `tests/test_trace_group.py` 与 `tests/test_v24_rework.py`。
- 角色能力可独立演进：新增工具、世界书、记忆层只影响单角色图，不影响调度器。
- 零新增依赖、零新增迁移：多角色完全复用 V1/V2 的图与存储。

**代价（当前接受）**

| 维度 | 代价 | 现状 |
|---|---|---|
| 延迟 | 群聊串行发言，后一位要等前一位整轮结束；每个发言人都跑完整图（记忆 + 世界书 + 生成） | 接受：产品语义就是"轮流说"；压测只测单会话并发，不承诺群聊并行 |
| 失败面 | `Supervisor` 的 N 个发言人需要 N+1 次决策（含终止 END）；终止决策不属于任何发言人 | 已修：R5 把 END 决策 usage 追加到最后一位 trace；R13 在 `turn_done` 前冻结计时，断连/决策异常不丢 trace（`209371a`） |
| 可观测性 | 一次群聊多条 trace，需要 `parent_trace_id` 聚合；上一位 trace 的 `total_ms` 不能包含下一次决策 | 已修：`TraceRecorder.stop()` 冻结快照；`total_ms` 不再含下一次决策；仍有"END 决策耗时不计入 total_ms"的已知口径 |
| 测试复杂度 | 群聊要覆盖断连、决策异常、时间归属、调用归属 | 接受：`tests/test_v24_rework.py` 3 条专门回归；490 tests 全绿 |
| 能力上限 | 角色之间不能主动"商量"，复杂任务需人（用户）驱动 | 接受：V2.6 选项 A 单独做长任务支线 |

**未来触发条件（出现则重新评估，不在本 ADR 内扩范围）**

1. 需要跨进程/多实例部署，且角色分布在不同服务；
2. 需要接入外部 agent（第三方 A2A 端点）或对外暴露角色能力；
3. 长任务成为主场景（V2.6 选项 A 验证有效）；
4. 群聊并发/吞吐成为瓶颈，需要并行发言或流式交错。

以上任一触发时，**另立 ADR**，不修改本 ADR 的结论。

## 5. When to revisit

- V2.6 选项 A 立项时：评估 planner 的输入输出 schema、子图并行度、plan/task 两级 trace、失败汇总语义；
- 引入多实例部署前：先解决 B-02 的单进程锁与 checkpointer 并发（V2.4 §7#4），再谈 A2A；
- 真实 bge 路径的性能专项（E-01）完成后：若首 token 仍 >6s，plan-execute 的延迟代价需要重新量化；
- 若出现"角色之间必须互相调用工具结果"的真实需求：优先用共享 Store/服务层函数解决，仍不引入 A2A。

## 6. 证据与边界（与相邻 ADR 的分工）

| ADR | 管什么 | 本 ADR 不重复什么 |
|---|---|---|
| ADR-0002 | 编排层为什么用 LangGraph、`build_graph()` 为什么是工厂 | 不复述图节点与存储边界 |
| ADR-0004 | 群聊可见性墙：写端扇出、读端过滤、不建公共空间 | 不重新论证隔离；本 ADR 只说明"A2A 会破坏它" |
| ADR-0009 | 世界书为什么独立建表、三路召回、预算 | 不涉及检索算法 |
| ADR-0010 | Skill 协议、工具循环、结构化失败、MCP 暴露 | 不涉及工具实现；本 ADR 只说明"工具是模型面向，不是 agent 面向" |
| ADR-0011 | 提醒的两阶段 HITL | 不涉及提醒 |

> 一句话：**同构 agent + 共享 Store + 读端隔离，决定了"调度"比"通信"更便宜；日常对话没有计划结构，决定了 plan-execute 没有收益。**
