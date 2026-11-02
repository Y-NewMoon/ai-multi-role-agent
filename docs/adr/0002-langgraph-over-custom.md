# ADR-0002：编排层采用 LangGraph，而非自研状态机

- 状态：**已接受**（补记：决策在 M1 落地，本文于 M4 前追记）
- 关联：方案 §3.2（记忆分层）、§3.3（图设计）、§3.8（存储边界）、§3.11A#3/#5（接缝）、§5（技术栈）
- 影响范围：`src/app/agents/`（`state.py` / `graph.py` / `nodes/`）、`src/app/main.py`、`migrations/`

---

## 1. Context

本项目需要一个能同时提供以下能力的编排层：

- **多轮会话状态**：每轮消息累积、按会话恢复；
- **持久化检查点**：对话历史在进程重启后不丢；
- **长期记忆存储 + 向量检索**：跨会话记忆召回；
- **可演进为群聊**：多个角色同场时，编排层不能被"单角色单例"焊死。

## 2. Decision

**编排层采用 LangGraph `StateGraph`；会话状态用 `AsyncPostgresSaver`（checkpointer），长期记忆用 `AsyncPostgresStore`。**

- 图为 5 节点线性链：`load_persona → retrieve_memory → compose_prompt → generate → update_memory`（[graph.py](file:///d:/Work/File/Ai-agent/src/app/agents/graph.py#L58-L71)）。
- 状态定义见 [state.py](file:///d:/Work/File/Ai-agent/src/app/agents/state.py#L13-L28)：`messages`（`add_messages` 累积）+ 服务层预取字段 + 图内中间产物。
- **`build_graph()` 是可实例化工厂，不是全局单例**（接缝 3）：依赖（`llm` / `memory_store` / `checkpointer` / `store` / 预算参数）全部由外部注入，每次调用返回独立已编译图（[graph.py](file:///d:/Work/File/Ai-agent/src/app/agents/graph.py#L23-L33)）。这是 M4 群聊"为每个角色各起一张图"的前提。

## 3. 存储边界（与 Alembic 的分工）

- **Alembic 只管业务表**：`agents / sessions / messages`（[0001_init.py](file:///d:/Work/File/Ai-agent/migrations/versions/0001_init.py)）。
- **LangGraph 自建表**（`checkpoints / checkpoint_writes / checkpoint_blobs / store / store_vectors`）由 `AsyncPostgresSaver.setup()` / `AsyncPostgresStore.setup()` 创建，**不纳入 Alembic 版本管理**，避免 `upgrade` 撞表（方案 §3.8）。
- 启动顺序固定：`alembic upgrade head` → 应用启动时 `setup()`（幂等）（[main.py](file:///d:/Work/File/Ai-agent/src/app/main.py#L33-L51)）。

## 4. 备选方案与取舍

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **LangGraph（选中）** | 官方维护的持久化 checkpoint + Store（含向量索引）；子图/多角色演进路径成熟；简历加分 | 引入框架耦合；版本行为需实测 | ✅ 采用 |
| 自研状态机 + 自建记忆表 | 完全可控 | 需自造 checkpoint 一致性、并发、向量索引；工作量大且易错；与"了解主流 Agent 框架"的定位相悖 | ❌ 弃用 |

## 5. Consequences

**正面**

- 与"记忆统一使用 LangGraph Store（不自研记忆表）"的已确认决策一致（方案 §2）。
- `build_graph()` 工厂形式使 M4 群聊可零重构地扩展到"N 个角色图实例"。

**代价**

- 存在一条**框架层约束**：向量索引在 store 建表时定死（`index={dims:1024, embed, fields:["text"]}`，[main.py](file:///d:/Work/File/Ai-agent/src/app/main.py#L45-L51)），**建表后不可更改**，更换 embedding 模型 = 重建向量表 + 全量重算。
- LangGraph 表不受 Alembic 管辖，意味着其 schema 由框架版本决定，需在 README / 本文档显式标注，避免误用 `alembic downgrade` 处理。
- Windows 下 psycopg 异步需 `SelectorEventLoop`，故服务必须经 `scripts/run_server.py` 启动（见方案 §5 与 README）。
