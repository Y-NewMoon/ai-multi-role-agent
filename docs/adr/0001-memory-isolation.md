# ADR-0001：记忆隔离采用逻辑隔离（namespace）

- 状态：**已接受**（补记：决策在 M1/M2 落地，本文于 M4 前追记，用于补齐方案 §6 的 ADR 清单）
- 关联：方案 §3.1（记忆隔离）、§3.11A#1/#2/#4（接缝）、§8 验证方式第 1–4 条、§9 风险 #5
- 影响范围：`src/app/memory/namespace.py`、`src/app/memory/store.py`、`src/app/services/chat_service.py`、`tests/test_memory_isolation.py`

---

## 1. Context

多角色并存是本项目核心约束：**各角色的长期记忆必须互不干扰**（"串味"是必须证伪的失败模式）。同时方案要求同一角色的记忆**跨会话共享**（角色成长卖点），且群聊（V2）加入后私聊/群聊的记忆不应割裂。

需要在两种物理形态间做选择：

- 每个角色一个独立存储（物理隔离）；
- 所有角色共用一个存储，按逻辑分区（逻辑隔离）。

## 2. Decision

**采用逻辑隔离：所有角色共用一个 `AsyncPostgresStore`，靠 namespace 强制分区。**

- namespace 只由 `agent_id` 决定：`(f"agent:{agent_id}", "memories")`（[namespace.py](file:///d:/Work/File/Ai-agent/src/app/memory/namespace.py#L10-L12)）。**不掺 session，不掺群聊上下文。**
- 会话状态用 `thread_id = f"{session_id}:{agent_id}"` 隔离（[chat_service.py](file:///d:/Work/File/Ai-agent/src/app/services/chat_service.py#L32-L35)），**禁止裸 `session_id`**。
- 记忆写入粒度 = `(agent_id, content)`，key = `sha1(f"{agent_id}:{content}")[:32]`，同内容重复写入自动覆盖（幂等）（[store.py](file:///d:/Work/File/Ai-agent/src/app/memory/store.py#L23-L26)）。
- `agent_id` 在**数据访问层强制注入**，不接受由 LLM 或前端直接决定。

## 3. 越权校验的位置

**接口层以 `session.agent_id` 为唯一依据，请求体中的 `agent_id` 一律忽略**（方案 §3.1）：`ChatService.load_context()` 只读业务库取 `session.agent_id` 与对应 `Agent`，不读取请求体 agent_id（[chat_service.py](file:///d:/Work/File/Ai-agent/src/app/services/chat_service.py#L58-L77)）。这样越权请求在进入图之前就被消解。

## 4. 备选方案与取舍

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **逻辑隔离（选中）** | 与 LangChain 官方 per-user/agent 模式一致；一套 Store、一份运维；隔离由数据结构保证 | 依赖 namespace 纯度，写错即串味 | ✅ 采用 |
| 物理隔离（每角色一库） | 隔离最硬 | 扩展性差、运维重（N 个库/N 条连接）；与"群聊要跨角色读同一批共享信息"冲突 | ❌ 弃用 |

## 5. Consequences

**正面**

- 隔离**由数据结构保证**，而非 prompt 约定：角色 B 的检索范围里根本不含 A 的 namespace 内容。
- 同一角色跨会话共享同一份记忆，天然满足"角色成长"卖点。
- 群聊（M4）可复用同一份记忆，接缝 2 零改动。

**代价**

- 存在"namespace 纯度"这一单点约束：任何一处把 session / 群聊上下文掺进 namespace，都会破坏隔离。**用测试强制断言**（见下）。

## 6. 验收（映射方案 §8 第 1–4 条）

`tests/test_memory_isolation.py`（真库 `roleplay_test`，6 例）已断言：

1. 向角色 A 写入记忆后，角色 B 检索返回空（`test_real_store_memories_isolated_between_agents`）；
2. 两个角色的 store namespace 不相等，且真 pgvector 索引下 top-k 能取回语义记忆；
3. 同一角色两 session 的 checkpoint `thread_id` 隔离，互不覆盖状态（`test_real_checkpoint_threads_isolated_across_sessions`）；
4. 越权请求以 `session.agent_id` 为准，且 A 的记忆不出现在 B 收到的 prompt 中（`test_real_service_uses_session_agent_and_keeps_memory_separate`）。

> 该测试文件**必须跑真库**，不用 `InMemoryStore` 兜底（测不出真实 schema 与向量索引行为，方案 §3.8）。
