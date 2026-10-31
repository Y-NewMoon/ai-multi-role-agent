# 多会话 LLM 对话后端（AI 多角色扮演智能体）

基于 **LangGraph + FastAPI + PostgreSQL/pgvector** 的多会话对话后端：每个角色维护独立长期记忆
（语义 / 事实 / 档案 / 层级摘要四层），按会话隔离对话历史，SSE 逐 token 流式输出；带世界书
**三路召回 + RRF + 重排**、**Skill / MCP 工具层**、**评测闭环**与运行期 **trace**。
附一个 Vite + React 的最小前端（单人对话 / 群聊 / 角色管理）。

**关键指标**（每个数字都有 manifest：`git_sha` + 配置 + 输入输出哈希，见文末「效果与评测」）

| 维度 | 现状 | 口径 |
|---|---|---|
| 自动化测试 | **591 collected**（49 条需真库的集成用例默认 skip） | `uv run pytest -q` |
| 评测集 | **8 套 / 350 条**（golden 40 · scenario 80 · adversarial 40 · retrieval 60 · lore 40 · memory 40 · skill_use 30 · agentic_rag 20） | `eval/suites/` |
| 检索质量 | hit@1 **0.85 → 0.90 → 0.975 → 1.00**（纯向量 → +关键词 → +重排 → +改写） | 40 条标注、2 篇文档 20 chunks |
| 并发压测 | 20×20 / 400 请求，p95 首 token **1457ms**、**11.46 req/s**（fake LLM + 关世界书；真实 bge 路径 6118ms） | `eval/report/loadtest-20261022-r1b-clean` |

> 口径与保留项（`hit@3` 饱和、adversarial 只跑 sample、真实 bge 未优化）一并写在
> [效果与评测](#效果与评测数字--样本--口径--出处) 表后，不藏。

## 演示

https://github.com/user-attachments/assets/eadb4330-a932-44f9-ad46-8f01a37258f2

> 按功能顺序演示 8 个场景：私聊 + 思考链 → 私聊里的闲聊 + 思考链 → 私聊清空闲聊 → 私聊撤回
> → 群聊 + 思考链 → 群聊里的闲聊 + 思考链 → 群聊清空闲聊 → 群聊撤回。
>
> 时长 2:05 · 1280×720

## 前置要求

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Docker | 任意近期版本 | 用于拉起 PostgreSQL + pgvector |
| Python | **≥ 3.12** | 见 `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | 最新 | Python 包与虚拟环境管理，需自行安装 |
| Node.js | ≥ 18（含 npm） | 仅前端需要 |
| LLM API Key | — | 默认走 DeepSeek 的 OpenAI 兼容接口，需自备 |

> **体积与耗时提示**：`uv sync` 默认安装 `embeddings` 组（sentence-transformers + torch，约 **2.5GB**）；
> 首次启动还会下载本地嵌入模型 `bge-m3`。请预留磁盘空间与下载时间。

## 技术栈

- **编排**：LangGraph `StateGraph`（6 节点线性图：`load_persona → retrieve_memory → retrieve_lore → compose_prompt → generate → update_memory`；
  V2.2 角色绑定了工具时，`generate` 后插入 `should_continue → tools → generate` 有界循环，最多 3 轮）
- **存储**：PostgreSQL + pgvector（`pgvector/pgvector:pg16`）
  - 业务表（`agents` / `sessions` / `messages` 等）由 Alembic 管理
  - 对话状态用 `AsyncPostgresSaver`（checkpointer），记忆用 `AsyncPostgresStore`（向量索引）
- **LLM**：OpenAI 兼容接口（默认 DeepSeek）
- **Embedding**：本地 `bge-m3`，维度固定 **1024**（`EMBEDDING_DIMS`，建表后不可改）

## 目录结构

```
src/app/
  agents/        LangGraph 图与节点
  api/           FastAPI 路由（v1: chat / agents / sessions / groups / uploads / lore）与依赖
  core/          异常与日志
  db/            SQLAlchemy 模型与引擎
  llm/           LLM / Embedding Provider
  lore/          世界书知识库（解析 / 切分 / 关键词 / 索引 / 检索 / 服务 / DTO）
  memory/        记忆读写（Store 封装、namespace、召回、摘要）
  prompts/       人设与摘要 prompt 模板
  schemas/       Pydantic DTO
  services/      业务服务（chat / agent / session / group / reminder）
  skills/        V2.2 Skill 工具层（协议 / 注册表 / 执行器 / 内置工具 / MCP server）
migrations/      Alembic 迁移
scripts/         启动脚本与示例角色种子
tests/           pytest（含真库集成测试）
frontend/        Vite + React + TS 最小前端（单人对话 + 群聊 + 角色管理）
```

## 快速开始

```bash
# 0. 前置：已安装 Docker、Python ≥ 3.12、uv、Node.js

# 1. 起数据库（含 roleplay 与 roleplay_test 两个库）
docker compose up -d

# 2. 配置环境变量（Windows PowerShell 用：Copy-Item .env.example .env）
cp .env.example .env
#   然后编辑 .env，至少填好 LLM_API_KEY

# 3. 安装依赖（首次较慢，约 2.5GB）
uv sync

# 4. 迁移业务表
uv run alembic upgrade head

# 5. 灌入示例角色（推荐；跳过则角色列表为空）
uv run python scripts/seed_personas.py

# 6. 启动后端
uv run python scripts/run_server.py        # http://127.0.0.1:8000
```

启动前端（另开一个终端）：

```bash
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

> **Windows 提示**：psycopg 异步需 `SelectorEventLoop`，直接 `uvicorn` CLI 会选到 Proactor
> 事件循环导致报错，请统一用 `scripts/run_server.py` 启动。

> **换个 LLM**：把 `.env` 里的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 指向任意
> OpenAI 兼容服务即可（如自建 vLLM、其他厂商的兼容端点）。

## 配置（.env）

`.env.example` 只列出常用项；下表为**全部可配置项**（未写进 `.env` 时取代码内置默认值，需要时自行追加）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | — | 主库连接串（**必填**） |
| `TEST_DATABASE_URL` | 空 | 测试库；不填则真库集成测试自动跳过 |
| `LLM_PROVIDER` | `deepseek` | Provider 标识 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `LLM_API_KEY` | 空 | **必填**，不填则对话与摘要报错 |
| `LLM_MODEL` | `deepseek-chat` | |
| `EMBEDDING_PROVIDER` | `local` | `local`（bge-m3）/ `fake`（确定性哈希向量，**仅压测/测试**） |
| `EMBEDDING_MODEL` | `bge-m3` | 本地嵌入模型 |
| `EMBEDDING_DIMS` | `1024` | 向量维度，**建表后不可改** |
| `PROMPT_TOKEN_BUDGET` | `8000` | 拼 prompt 的 token 预算 |
| `PROMPT_RECENT_ROUNDS` | `10` | 拼 prompt 时保留的最近轮数 |
| `SUMMARY_TRIGGER_ROUNDS` | `10` | 触发滚动摘要的轮数阈值（V2.3 D9：与 `PROMPT_RECENT_ROUNDS` 对齐，配置层校验 `trigger ≤ recent`） |
| `SUMMARY_CHUNK_ROUNDS` | `10` | 每次摘要覆盖的轮数 |
| `REPLY_MAX_MESSAGES` | `3` | 一次回复最多连发几条 |
| `REPLY_THINK_RETRY_ENABLED` | `true` | 整轮缺内心独白时补写一次（额外一次完整 prompt 调用，可关闭） |
| `GROUP_MAX_MEMBERS` | `8` | 群内角色数上限（不含用户） |
| `GROUP_MAX_TURNS` | `10` | 一轮群聊内最多发言次数 |
| `IDLE_GROUP_MAX_MESSAGES` | `10` | 群聊一次闲聊累计消息上限 |
| `IDLE_GROUP_TURN_MAX_MESSAGES` | `2` | 群聊闲聊时每角色每次最多发言条数 |
| `AVATAR_DIR` | `data/avatars` | 头像本地存储目录（相对项目根） |
| `AVATAR_URL_PREFIX` | `/static/avatars` | 头像静态服务挂载前缀 |
| `AVATAR_MAX_BYTES` | `2097152` | 头像单文件上限（2MB） |
| `LORE_ENABLED` | `true` | 世界书检索总开关；关闭后 `retrieve_lore` 直接返回空、不访问数据库 |
| `LORE_MAX_RATIO` | `0.20` | 检索 lore 段占 prompt 预算的比例 |
| `LORE_ALWAYS_ON_RATIO` | `0.08` | always-on（核心设定）段占 prompt 预算的比例 |
| `LORE_QUERY_REWRITE` | `true` | 查询改写开关（含代词、过短，或上一轮有未消解指代时触发；CI 与单测用 false） |
| `LORE_TOP_K` | `20` | 单路召回与融合后进入 rerank 的条数上限 |
| `LORE_STATIC_MODE` | `full` | V2.2 静态注入模式：`full`（V2.1 行为，全量检索）/ `always_on`（只注入核心设定，不走向量/关键词/rerank）/ `off`（不注入）；供 agentic RAG 三策略对比 |
| `WEB_SEARCH_URL` | 空 | `web_search` 工具后端地址；与 KEY 任一为空时该工具返回 `unavailable`（不算失败） |
| `WEB_SEARCH_API_KEY` | 空 | 同上 |
| `SKILL_MAX_TOOL_ROUNDS` | `3` | 图内工具循环上限（达到上限强制文本收尾） |
| `MEMORY_EXTRACT_ENABLED` | `true` | V2.3 事实抽取总开关；`false` 时行为逐字退回 V2.2（不调 LLM、不写新条目、prompt 不变） |
| `MEMORY_EXTRACT_MIN_CHARS` | `8` | 本轮用户文本短于此值则跳过抽取 |
| `MEMORY_EXTRACT_TIMEOUT_S` | `8` | 抽取调用超时（秒）；超时按 fail-open 跳过本轮 |
| `MEMORY_FACT_MODE` | `overwrite` | 冲突处理：`overwrite`（旧条目标 `superseded_at`）/ `keep_both_with_note`（保留两条并在旧值后加「（你以前说 X）」） |
| `MEMORY_FACT_LIMIT` | `200` | 每角色事实条数硬上限（超限按 `importance` 升序 + 最旧淘汰） |
| `MEMORY_PROFILE_ENABLED` | `true` | 四槽位档案（称呼 / 禁忌 / 关系阶段 / 重要日期）常驻注入开关 |
| `MEMORY_PROFILE_RATIO` | `0.10` | profile 段占 prompt 预算的比例上限 |
| `MEMORY_WEIGHTED_RECALL` | `true` | 加权召回开关；`false` = 纯余弦（消融用） |
| `MEMORY_W_COSINE` / `MEMORY_W_RECENCY` / `MEMORY_W_IMPORTANCE` | `0.6` / `0.25` / `0.15` | 加权召回三项权重 |
| `MEMORY_RECENCY_HALFLIFE_DAYS` | `14` | recency 半衰期（天） |
| `MEMORY_RECALL_OVERSCAN` | `50` | 加权重排的 overscan 条数（应用层重排，不动 Store） |
| `MEMORY_RECALL_LIMIT` | `5` | 语义记忆召回条数（原写死在 `retrieve_memory.py`） |
| `MEMORY_L2_ENABLED` | `true` | L2 层级摘要开关 |
| `MEMORY_L2_RATIO` | `0.05` | L2 段占 prompt 预算的比例上限 |
| `MEMORY_L2_EVERY_MERGES` | `3` | 每 N 次 L1 滚动合并重合成一次 L2（≈ N × 10 轮） |
| `MEMORY_L2_MAX_CHARS` | `300` | L2 文本字数上限 |
| `TRACE_ENABLED` | `true` | trace 总开关；`false` 时主流程照常、不写文件 |
| `TRACE_DIR` | `data/traces` | trace JSONL 目录（按天分文件，相对项目根） |
| `TRACE_SAMPLE_RATE` | `1.0` | trace 采样率（0~1，越界按 1.0 并 warning） |
| `TRACE_MAX_FIELD_CHARS` | `200` | trace 内文本字段截断长度（隐私 + 体积） |
| `TRACE_MEMORY_STATS` | `false` | C-03 观察项；`true` 时每轮多一次 `asearch(limit=1000)` 统计 namespace 条目 |
| `PROMPT_VERSION` | `v3` | 写入 trace 的 prompt 版本标识；自 `v3` 起所有角色 prompt 均含【当前时间】，已知上一条用户消息且间隔 ≥10 分钟时另加“你们上次说话是…”一行（首轮/旧数据无此行；工具角色沿用 V2.6 的【可用工具】段） |
| `LLM_PRICE_INPUT_CNY_PER_1M` | `2.0` | 输入价（元/百万 token），**示例值**，部署须按账单覆盖 |
| `LLM_PRICE_OUTPUT_CNY_PER_1M` | `8.0` | 输出价（元/百万 token），**示例值** |
| `LLM_MAX_CONCURRENCY` | `8` | 全局 LLM 并发上限（`asyncio.Semaphore`，<1 按 1 并 warning） |
| `LLM_TIMEOUT_S` | `30` | 单次 LLM 调用超时兜底（秒，≤0 按 30 并 warning） |
| `LLM_TIMEOUT_FALLBACK` | `（我这边有点走神，稍等一下再说。）` | 超时/失败时的降级回复文本 |
| `DB_POOL_MIN_SIZE` | `4` | psycopg 连接池下限（checkpointer + store 共用；<1 按 1 并 warning） |
| `DB_POOL_MAX_SIZE` | `20` | 连接池上限；**应 ≥ 并发 SSE 连接数**，否则 checkpoint/记忆查询在取连接处排队（<min 时按 min） |
| `REMINDER_PENDING_TTL_HOURS` | `24` | pending 提醒的惰性过期小时数（≤0 按 24 并 warning） |
| `CHAT_IDEMPOTENCY_TTL_S` | `600` | 非流式 `/chat` 幂等缓存 TTL（单进程内存） |
| `CHAT_IDEMPOTENCY_MAX_ENTRIES` | `1024` | 幂等缓存容量上限（LRU 淘汰） |
| `MEMORY_API_AGENTS` | `ext` | 本机记忆接口的 agent 白名单（逗号分隔；纯数字一律拒绝） |
| `MEMORY_API_BUFFER_PATH` | `data/memory_api/buffer.json` | 未摘要消息滚动缓冲文件（gitignored，不得提交） |
| `MEMORY_API_BUFFER_MAX_MESSAGES` | `200` | 缓冲消息上限；超限返回 503 `buffer_overflow`（不静默删） |
| `MEMORY_API_IDEMPOTENCY_TTL_S` | `600` | 记忆接口 turn 幂等缓存 TTL（单进程内存，重启失效） |
| `MEMORY_API_IDEMPOTENCY_MAX_ENTRIES` | `512` | 记忆接口幂等缓存容量 |
| `MEMORY_API_MAX_TEXT_CHARS` | `8000` | 单条 `user_text` / `replies[]` / `contents[]` 上限 |
| `MEMORY_API_MAX_REPLIES` | `20` | 每轮最多角色回复条数 |
| `FAKE_LLM_DELAY_MS` | `10` | fake provider 每段延迟（仅 `LLM_PROVIDER=fake`） |
| `FAKE_LLM_REPLY` | `（压测假回复）` | fake provider 固定回复（仅压测） |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（`llm_provider` / `fake_llm_calls` / 库名 / 嵌入与 trace 配置；压测脚本据此拒绝非 fake） |
| POST | `/api/v1/chat` | 非流式对话（可选 `client_msg_id`，同值重放返回 `idempotent=true`） |
| POST | `/api/v1/chat/stream` | **SSE 流式对话** |
| POST | `/api/v1/chat/idle/stream` | **SSE 自主闲聊**（私聊，按钮触发一次） |
| GET / POST | `/api/v1/agents` | 角色列表 / 创建 |
| GET / PATCH / DELETE | `/api/v1/agents/{agent_id}` | 角色详情 / 更新 / 删除（级联） |
| POST | `/api/v1/agents/{agent_id}/memory/clear` | 清除该角色记忆与全部对话历史 |
| GET / POST | `/api/v1/sessions` | 会话列表（可按 `agent_id` 过滤）/ 创建 |
| GET / PATCH / DELETE | `/api/v1/sessions/{session_id}` | 会话详情 / 更新 / 删除 |
| GET | `/api/v1/sessions/{session_id}/messages` | 会话消息快照 |
| POST | `/api/v1/sessions/{session_id}/messages/{message_id}/recall` | 撤回该条消息 |
| POST | `/api/v1/sessions/{session_id}/idle/clear` | 清除本会话的自主闲聊消息 |
| GET / POST | `/api/v1/groups` | 群聊列表 / 创建（`{name, agent_ids}`） |
| GET / DELETE | `/api/v1/groups/{group_id}` | 群详情（含 `agent_ids`）/ 删除群 |
| DELETE | `/api/v1/groups/{group_id}/members/{agent_id}` | 将某角色移出群 |
| GET | `/api/v1/groups/{group_id}/messages` | 群消息快照 |
| POST | `/api/v1/groups/{group_id}/messages/{message_id}/recall` | 撤回群消息 |
| POST | `/api/v1/groups/{group_id}/chat/stream` | **SSE 群聊**（多角色轮流发言） |
| POST | `/api/v1/groups/{group_id}/idle/stream` | **SSE 群聊自主闲聊** |
| POST | `/api/v1/groups/{group_id}/idle/clear` | 清除该群的自主闲聊消息 |
| POST | `/api/v1/uploads/avatar` | 上传角色头像（`multipart/form-data`） |
| GET | `/static/avatars/{name}` | 头像静态资源 |
| POST | `/api/v1/lore/docs` | **导入 / 更新一篇世界书**（同 `slug` 即更新，`version+1`；并发冲突 409） |
| GET | `/api/v1/lore/docs?agent_id=` | 某角色的世界书列表（含 chunk 数与 token 总数） |
| GET | `/api/v1/lore/docs/{doc_id}` | 世界书详情 |
| DELETE | `/api/v1/lore/docs/{doc_id}` | 删除一篇（软删：doc 与 chunks 置 `deleted`） |
| POST | `/api/v1/lore/reindex` | 增量重建（补齐缺失的 embedding / tsv；`dry_run` 只统计） |
| GET | `/api/v1/lore/preview?agent_id=&q=` | 调试：三路命中 → RRF → rerank → 最终注入（rerank 降级时 `rerank=false`） |
| GET | `/api/v1/reminders?agent_id=&session_id=` | 列出该会话的提醒（默认不含 `cancelled`） |
| POST | `/api/v1/reminders/{reminder_id}/confirm` | 确认提醒：`pending` → `active`（幂等） |
| DELETE | `/api/v1/reminders/{reminder_id}` | 取消提醒：`pending`/`active` → `cancelled`（幂等） |
| POST | `/api/v1/memory/turn` | 本机记忆接口：追加一轮（用户 + 角色回复），触发记忆/抽取/摘要/L2 |
| POST | `/api/v1/memory/context` | 本机记忆接口：按 `query` 读取 profile/facts/L2/summary/memories |
| POST | `/api/v1/memory/forget` | 本机记忆接口：按内容删除语义记忆（不动 facts/profile/L2/summary） |
| GET | `/api/v1/memory/state` | 本机记忆接口：条目/事实/档案/缓冲规模（`session` 可选） |

### SSE 协议（`POST /api/v1/chat/stream`）

请求体：`{"session_id": 1, "message": "在吗"}`，响应 `text/event-stream`，事件类型：

- `token`：`data: {"text": "增量文本"}`（可多次）
- `done`：`data: {"session_id": 1, "agent_id": 2, "replies": [{"content": "...", "message_id": 9, "thinking": "..."}]}`
- `error`：`data: {"detail": "错误信息"}`（流开始后异常以事件表达，HTTP 仍为 200）

时序保证：逐 token 下发 → 图执行完成（记忆已落 Store）→ 写 `messages` 快照 → 才推 `done`。
V2.4 起 SSE **不做幂等**（不认 `client_msg_id`），幂等只用于非流式 `/chat`。

### 群聊（M4）

一轮群聊里，用户发一条消息，组内多个角色**串行轮流**发言。调度用 LLM supervisor +
硬约束（一轮内每人最多一次、连续同人强制换人、一轮最多 `GROUP_MAX_TURNS` 次、
解析失败退化为按成员顺序轮询），保证永不卡死。

**可见性墙在读端**：用户消息与各角色公屏回复不建公共空间，而是**扇出副本**写进每个成员
**自己的** namespace（`agent:{id}`，标 `visibility=group`）；每个角色只能读到自己的副本，
私聊秘密不会泄漏给他人。

**SSE 协议（`POST /api/v1/groups/{group_id}/chat/stream`）**，请求体 `{"message": "..."}`：

- `speaker`：`data: {"agent_id": 1, "name": "四季夏目"}`（轮到谁发言）
- `token`：`data: {"agent_id": 1, "text": "增量文本"}`（该角色逐 token，可多次）
- `turn_done`：`data: {"agent_id": 1, "reply": "该角色本轮完整回复"}`
- `done`：`data: {"group_session_id": 1, "order": [1,4,5], "replies": [...]}`（整轮结束）
- `error`：`data: {"detail": "错误信息"}`

时序保证：`speaker → token* → turn_done` 依次交替（每个发言人一组）；整轮所有角色结束、
`group_messages` 快照落库之后，才推最终 `done`。

> 群内角色数上限 `GROUP_MAX_MEMBERS`（默认 8）、一轮发言上限 `GROUP_MAX_TURNS`（默认 10）见 `.env`。

### 世界书知识库（V2.1）

为每个角色维护一份**世界书**（背景设定），在对话前按当前输入召回相关条目、注入 prompt，让回答贴合设定。
设计与选型见 `doc/decisions/0007-chinese-fulltext.md`、`0009-lore-retrieval.md`。

**数据层**：`lore_docs`（一篇设定，`slug` 稳定标识、`always_on` 核心设定）+ `lore_chunks`（切分后的条目，含
`embedding vector(1024)` / `tsv` / `keywords TEXT[]`），索引 `ix_lore_hnsw`（余弦）/ `ix_lore_tsv` / `ix_lore_kw`（GIN）。

**摄取**：`POST /api/v1/lore/docs` 传 Markdown（`## ` 切 section，> 500 token 的 section 按 400 token / 60 overlap 再切），
入库前 jieba 分词生成 `tsv`、bge-m3 生成向量、抽取关键词；同 `slug` 重复导入 = 内容变更则 `version+1` 并增量重建（幂等）。

**检索（`retrieve_lore` 节点，图里位于 `retrieve_memory` 之后）**：三路召回 →
- **always-on**：`always_on=true` 的核心设定，常驻注入；
- **keyword**：查询经 jieba 分词后 `to_tsquery('simple', '词1 | 词2')`（写入侧同样 jieba 分词，两侧一致才有效）；
- **vector**：bge-m3 余弦距离（HNSW）。

三路用 **RRF** 融合（`score(d)=Σ 1/(60+rank_r(d))`，并列按 `source_ref` 升序）→ **rerank**（`BAAI/bge-reranker-v2-m3` CrossEncoder，懒加载、失败自动降级为不重排且不报错）→ 按预算截断。含代词（这/那/她/他/它/其）、`len<8`，或上一轮有未消解指代时先做 query 改写（1.5s 超时，超时回退原查询）。

**预算（`compose_prompt`）**：`LORE_ALWAYS_ON_RATIO`（默认 8%）给 always-on、`LORE_MAX_RATIO`（默认 20%）给检索 lore；
超预算按序裁剪：always-on（`source_ref` 倒序）→ 摘要整段 → 记忆整段 → lore rerank 分升序 → 最旧轮次（摘要/记忆无独立段上限）。

**调试**：`GET /api/v1/lore/preview?agent_id=1&q=...` 返回三路命中、RRF、rerank 与最终注入，可解释召回。

> 检索质量与四组消融（纯向量 / +关键词 / +rerank / +改写）见 `eval/README.md` 与 `doc/V2/V2.1-消融报告.md`。
> `retrieval_lore` 需要真库 + 真嵌入，**不进 PR CI**。

### 工具调用（Skill 工具层，V2.2）

角色可绑定一组**工具**（`agents.allowed_skills`，`TEXT[]`，默认 `'{}'` = 不绑工具）。绑了工具的角色，
`compose_prompt` 追加 `【可用工具】` 段（只放「工具名 + 何时使用」，完整 JSON Schema 只在 `generate` 绑定时下发）；
`generate` 后进入有界工具循环（`should_continue → tools → generate`，最多 `SKILL_MAX_TOOL_ROUNDS` 轮，达到上限强制
`tool_choice="none"` 收尾）。工具消息不写入 `state.messages`，不污染 prompt/记忆/快照。
设计决策见 `doc/decisions/0010-skills-mcp.md`、`0011-reminder-hitl.md`。

**五个内置工具**（契约 §2.5）

| 工具 | 参数 | 返回 `data` | 失败 `error_type` | side_effect |
| --- | --- | --- | --- | --- |
| `get_current_time` | `timezone="Asia/Shanghai"` | `{iso, human, timezone, offset}` | `invalid_timezone` | 否 |
| `roll_dice` | `dice="1d6"`、`times=1`（1~10） | `{rolls, total, dice, times}` | `invalid_dice` | 否 |
| `web_search` | `query`（1~200 字）、`top_k=3`（1~5） | `{results:[{title,url,snippet}], provider}` | `unavailable` / `timeout` / `empty` | 否 |
| `set_reminder` | `when`、`content`（1~200 字） | `{reminder_id, status, remind_at, idempotent}` | `invalid_time` / `invalid_content` | **是** |
| `read_lore` | `query`、`top_k=5`（1~10） | `{chunks:[{source_ref,header_path,content,score}]}` | `unavailable` / `empty` | 否 |

`read_lore` 复用 V2.1 的 `Retriever`（`rewrite_enabled=False`，不注入 prompt，只回灌模型）；查询世界书的角色可用它核实设定事实。

**执行器**（`src/app/skills/executor.py`）统一兜底：pydantic 参数校验、超时（默认 5s；`read_lore` 因 rerank 为 15s）+
1 次重试（0.5s 退避）、同请求同参数缓存（`sha256(name + canonical_json(args))`）、上游 5xx/429 熔断该工具 60s、
单条结果 4000 token / 单请求累计 8000 token 截断；**任何未捕获异常都收敛为结构化 `ToolResult`（`error_type=internal`），不抛 500**，
失败结果照常回灌模型，最终回复仍保持角色口吻。

**HITL 提醒**：`set_reminder` 只落 `pending` 不激活（模型应提示用户确认，不得声称已设置成功）；
经 `POST /api/v1/reminders/{id}/confirm` 后才 `active`。同 `(agent_id, session_id, content, remind_at)`
重复调用返回既有记录（DB 唯一键，`idempotent=true`）。**V2.2 不做定时投递**，`fired` 状态保留给 V2.4。

**MCP**：`python -m app.skills.mcp_server`（需 `PYTHONPATH=src`）起 stdio server，暴露同样 5 个工具；
有状态工具（`read_lore` / `set_reminder`）经 `MCP_AGENT_ID` / `MCP_SESSION_ID` 环境变量注入上下文，缺失时返回 `context_missing`。
冒烟：`uv run python scripts/mcp_smoke.py`（未装 `mcp` 时打印 skip 并退出 0）。

**评测**：`skill_use`（30 条：工具触发 / 参数 / no-tool / 失败矩阵）与 `agentic_rag`（20 条：static / agentic / hybrid 三策略）
走独立脚本，**不进** `run_eval.SUITE_NAMES`、不进 PR CI：

```powershell
uv run python scripts/run_skill_eval.py --agent 1    # → eval/report/skill-use-*.{json,md} + manifest.json
uv run python scripts/run_agentic_rag.py --agent 1   # → eval/report/agentic-rag-*.{json,md} + manifest.json
```

两脚本都写 `manifest.json`（`git_sha` / `git_dirty` / 时间戳 / 套件与报告 `sha256`），并落盘每 case 的
`tool_calls` / `tool_results` / `ranked` / `final_reply` 供独立重算；指标口径见 `eval/README.md`。

### 记忆系统（V2.3）

V2.2 的记忆只有「语义记忆（向量召回）+ 会话滚动摘要」；V2.3 在此基础上补了四层，全部仍存 LangGraph `BaseStore`
（`agent:{agent_id}/memories`），**不新增任何业务表**（理由见 `doc/decisions/0006-memory-facts.md`）。

| 层 | key | 写入时机 | 注入方式 |
| --- | --- | --- | --- |
| 语义记忆 | `sha1(agent_id:content)[:32]`（旧规则不变） | 每轮消息 | 加权召回 top-`MEMORY_RECALL_LIMIT` |
| **事实** | `fact:{sha1(agent\|subject\|predicate\|object)[:24]}` | 每轮回复后由 `update_memory` 抽取 | 同语义记忆一起参与加权召回 |
| 当前事实指针 | `fact_cur:{sha1(agent\|subject\|predicate)[:24]}` | 冲突判定 | 不注入，仅用于 O(1) 判冲突 |
| **档案 profile** | `profile:{称呼\|禁忌\|关系阶段\|重要日期}` | 抽取时顺带产出 | 常驻注入（≤ `MEMORY_PROFILE_RATIO`），**不走向量召回** |
| **L2 关系档案** | `l2:portrait` | 每 `MEMORY_L2_EVERY_MERGES` 次 L1 合并重合成 | 常驻注入（≤ `MEMORY_L2_RATIO`），≤300 字 |
| L2 计数 | `l2:counter` | 每次 L1 滚动合并 +1 | 不注入 |

- **事实抽取**：`update_memory` 在回复落库后**一次性**调用 LLM，同时产出 `facts[]` 与 `profile{}`（`D3`，每轮 1 次调用）。
  抽取发生在回复之后（`D2`），不增加首字延迟；失败 / 超时 / JSON 解析失败一律 fail-open 跳过本轮（不阻断回复）。
- **冲突失效**：同 `(subject, predicate)` 出现新 `object` 时，`overwrite` 给旧条目标 `superseded_at`（检索默认过滤），
  `keep_both_with_note` 保留两条并对旧值加「（你以前说 X）」。切换开关：`MEMORY_FACT_MODE`。
- **加权召回**：`0.6*cos + 0.25*recency + 0.15*importance`，先 overscan=50 取候选再应用层重排；
  `MEMORY_WEIGHTED_RECALL=false` 退回纯余弦。召回文本带角色标签：`用户说过：…` / `你说过：…`。
- **层级摘要 L2**：按**对话进度**合成（不是「攒够 N 个会话」）——每攒够 N 次 L1 滚动合并（N≈3，即 ≈30 轮）就把
  「现有 L2 + 当前 L1 + 高 importance facts」重合成一段 ≤300 字的「我和这位客人之间发生过什么」。
  计数器与 L2 都按 agent 存、**不掺 session_id**，因此私聊 / 群聊共用同一份关系记忆。
- **摘要阈值对齐（D9）**：`SUMMARY_TRIGGER_ROUNDS` 默认 20 → 10，与 `PROMPT_RECENT_ROUNDS=10` 对齐，
  消除「第 11–20 轮掉出近轮窗口却还没进摘要」的上下文空窗；配置层对 `trigger > recent` 打 warning。
- **回填**：`uv run python scripts/backfill_memory.py --agent 1 [--dry-run]` 给存量条目补 `kind` / `importance` / `created_at`（幂等）。
- **手动重合成**：`uv run python scripts/compact_memory.py --agent 1 [--force]`（`--force` 忽略计数阈值）。
- **评测**：`eval/suites/memory.jsonl`（40 条）+ `scripts/run_memory_eval.py`（冲突 / 档案 / 召回五指标）与
  `scripts/run_long_session.py`（100 轮合成历史、**不调 LLM**，只比 token 与分段占比）；口径见 `eval/README.md`。

### 本机记忆接口（V3 M2）

给**同机外部进程**读写一份隔离记忆（`agent` / `session` 都是字符串，不建角色行、不建表）。
只绑 `127.0.0.1`、不鉴权；`agent` 必须在 `MEMORY_API_AGENTS` 白名单里，纯数字一律拒绝。

| 端点 | 作用 |
| --- | --- |
| `POST /api/v1/memory/turn` | 追加一轮（用户 + 每条角色回复），直接调 `update_memory`：写语义记忆 / 抽取 facts+profile / 滚动摘要 / L2 |
| `POST /api/v1/memory/context` | 按 `query` 返回 `profile` / `facts` / `boundaries=[]` / `l2` / `memories` / `summary` / `debug` |
| `POST /api/v1/memory/forget` | 按内容删除语义记忆，返回 `deleted` / `missing`（**不删** facts/profile/L2/summary） |
| `GET /api/v1/memory/state` | 条目数（分页统计，不受 1000 封顶）/ 未失效事实 / profile 槽位 / L2 计数 / 缓冲规模 |

- **滚动缓冲**：未摘要消息落在 `MEMORY_API_BUFFER_PATH`（单文件 JSON，原子写）；每条消息（含 AI 回复）都有 `id`，
  摘要后按 `RemoveMessage.id` 真删；进程重启从文件恢复；同 `(agent, session)` 串行、单进程。
- **错误信封**：仅 `/api/v1/memory/*` 使用 `{ok:false, contract:"1", error:{code,message}}`；既有 API 仍 `{detail}`。
  错误码闭集：`invalid_request` / `agent_not_allowed` / `duplicate_message_id` / `buffer_overflow` /
  `buffer_persist_failed` / `memory_write_failed` / `retrieval_failed` / `forget_failed` / `state_failed` / `internal_error`。
- **不做什么**：不生成回复、不调 `compose_prompt`、不碰图 / checkpointer / persona / lore / tools；
  `/memory/boundary` 随 M4；不做多实例（单文件缓冲会被互相覆盖）。

```powershell
# turn
curl.exe -X POST http://127.0.0.1:8000/api/v1/memory/turn `
  -H "Content-Type: application/json" `
  -d '{"agent":"ext","session":"main","user_text":"今天有点累","replies":["那就早点休息。"],"at":"2026-10-25T10:35:00+08:00"}'

# context
curl.exe -X POST http://127.0.0.1:8000/api/v1/memory/context `
  -H "Content-Type: application/json" `
  -d '{"agent":"ext","session":"main","query":"用户喜欢什么茶"}'

# forget
curl.exe -X POST http://127.0.0.1:8000/api/v1/memory/forget `
  -H "Content-Type: application/json" `
  -d '{"agent":"ext","contents":["今天有点累"]}'

# state
curl.exe "http://127.0.0.1:8000/api/v1/memory/state?agent=ext&session=main"
```

### 可观测与稳定性（V2.4）

**trace（文件优先，不新增表）**：每次对话/每个群聊发言人写一条 JSONL 到 `data/traces/YYYY-MM-DD.jsonl`
（`TRACE_DIR` 可改）。`contextvars` 隔离并发请求；采样 `TRACE_SAMPLE_RATE`；**写失败 fail-open**
（只 warning，不影响回复）；`TRACE_ENABLED=false` 时完全不写。文本字段按 `TRACE_MAX_FIELD_CHARS` 截断。

字段（`trace_version=1.0`，完整样例见交接文档 §8.1）：`trace_id` / `parent_trace_id` / `mode`
（`private` / `group` / `idle`）/ `session_id` / `group_session_id` / `agent_id` / `turn` / `speaker_index` /
`started_at` / `finished_at` / `first_token_ms` / `total_ms` / `nodes[]` / `retrieval` / `tools[]` /
`llm`（`model` / `calls` / `prompt_tokens` / `completion_tokens` / `cost_cny` / `usage_source`）/
`budget` / `error` / `degraded`。

- **时间戳口径**：`started_at` / `finished_at` 固定按 `Asia/Shanghai` 写（`+08:00`），
  与宿主时区无关——CI（UTC）与部署机（`+08:00`）输出同一种偏移，按天分文件名也用同一时区。

- **`retrieval.lore_ids` 口径**：元素是 chunk 的 `source_ref`，即 `"{slug}:{seq:02d}"`
  （如 `yunqi_tea:10`、`qingxi_town:05`），**不是**数字 `doc_id`；`slug` 为世界书文件名/文档 slug。
  库、API 与评测（`eval/lore_eval.py`）三处同源，负控见 `tests/test_v24_rework.py`。
- **`budget` 与 `llm.prompt_tokens` 不是同一口径**：`budget.estimated_prompt_tokens` /
  `budget.prompt_token_budget` 由 `compose_prompt` 对**最终 prompt 内容**估算（判断是否该裁剪）；
  `llm.prompt_tokens` 是一轮内**所有 LLM 调用**（生成 + 事实抽取等）的估算之和，通常更大。
  两者不可直接相减或互换。`retrieval.rewrite.ms` 是整轮 `retrieve_lore` 节点的**上界**（含三路召回/融合），
  不是纯改写耗时。
- **成本**：token 用 `cjk1_other4` 估算（中文 1 字 ≈ 1 token，其余 4 字 ≈ 1 token），
  `cost_cny = round(prompt/1e6*LLM_PRICE_INPUT_CNY_PER_1M + completion/1e6*LLM_PRICE_OUTPUT_CNY_PER_1M, 6)`。
  **默认价格是示例值，部署必须按实际账单覆盖 `.env`**；估算是相对值，不是账单。
  `LLM_PROVIDER=fake` 时 trace 写 `model="fake"`、`usage_source="fake"`、`cost_cny=0`，不计费。
- **聚合**：`uv run python scripts/trace_report.py --date 2026-10-21 --out eval/report/trace-smoke.json`
  （另支持 `--since/--until`、`--sample`、`--limit`、`--trace-dir`）。输出 JSON + Markdown + manifest，
  含错误率 / 降级率 / p50、p95（total 与首 token）/ 节点耗时与失败率 / 工具失败率 / 按 agent、session 的成本；
  **fake trace 单独计 `fake_traces/fake_calls`，不进入成本与延迟聚合**；空输入 `traces=0` 且指标为 `null`，
  不伪造。脚本本身**不调用 LLM**（`usage.llm_calls=0`）。
- **隐私**：trace 含会话/角色 id 与截断后的文本片段，按内部日志管理（勿公开上传；`data/` 已在 `.gitignore`）。

**稳定性**

- **并发信号量**：全局 `asyncio.Semaphore(LLM_MAX_CONCURRENCY)` 包住所有 LLM 调用；**单次超时兜底**
  `asyncio.wait_for(LLM_TIMEOUT_S)`，超时返回 `LLM_TIMEOUT_FALLBACK` 文本并把该轮标记 `degraded=true`，
  该轮 `update_memory` 跳过所有写入（不污染记忆），接口不 500。
- **锁**：同 `thread_id` 串行、不同键并行；`KeyedLockPool` 引用计数，用完即释放条目（1000 会话后锁表归零）。
  **单进程语义**：多实例部署仍可能并发写 checkpointer，需换分布式锁（§7）。
- **连接池**：checkpointer 与 store 共用 `DB_POOL_MAX_SIZE`（默认 20）个 psycopg 连接；上限应覆盖并发 SSE 数，
  否则请求会在取连接处排队（20 并发下 4 连接会把 p95 首 token 从 1.26s 抬到 1.99s）。
- **checkpoint 并发**：`langgraph-checkpoint-postgres` 3.1.2 的 `AsyncPostgresSaver` 对连接池也使用实例级
  全局 `asyncio.Lock`，会把同进程所有会话的 checkpoint I/O 串行化（吞吐封顶 ~1.5 req/s）；`src/app/db/checkpointer.py`
  的 `PooledAsyncPostgresSaver` 在池化连接下关掉该锁（每次 `_cursor` 独占一条池内连接），同 `thread_id` 的
  先后仍由上面的锁池保证。上游修复后可删掉该类（§7）。
- **非流式幂等（D8）**：`POST /api/v1/chat` 带 `client_msg_id`（≤64 字）时，同 `(session_id, client_msg_id)`
  在 `CHAT_IDEMPOTENCY_TTL_S`（默认 600s）内重放只执行一次，返回首次结果且 `idempotent=true`；
  缓存为单进程 `OrderedDict`（容量 `CHAT_IDEMPOTENCY_MAX_ENTRIES`，LRU 淘汰）。**SSE 不做幂等**，
  空 `client_msg_id` 行为与 V2.3 一致。

**B-01 工具可用性**：每个 Skill 声明 `available`（如 `web_search` 在 `WEB_SEARCH_URL` / `WEB_SEARCH_API_KEY`
任一为空时不可用）。不可用工具**不下发**给模型（`to_openai_tools` / `【可用工具】` / MCP 列表均过滤），
避免模型调用后只拿到 `unavailable`。

**B-02 锁池**：见上「锁」——`src/app/core/locks.py`，聊天与群聊共用。

**B-03 提醒 TTL**：`pending` 超过 `REMINDER_PENDING_TTL_HOURS`（默认 24h）在 `list` / `confirm` 时被
置为 `cancelled`（惰性清理，无后台任务），不会出现在列表、也不会被确认成 `active`；
`set_reminder` 的描述与 `when_to_use` 明确要求「创建后必须请用户确认，不得说已记下/已设置」。

**压测（fake LLM + fake embedding，0 真实 token）**

```powershell
$env:DATABASE_URL="postgresql://roleplay:roleplay@localhost:5432/roleplay_test"
$env:LLM_PROVIDER="fake"
# R1：压测只衡量服务链路；bge-m3 在 20 并发下的 CPU 争用会淹没首 token 指标
$env:EMBEDDING_PROVIDER="fake"
$env:LORE_ENABLED="false"
# 每次压测使用独立 trace 目录：eval/report/trace-<run_id>，run_id 与 --out 文件名一致
$env:TRACE_DIR="eval/report/trace-20261021"
$env:TRACE_SAMPLE_RATE="0.05"
uv run python scripts/run_server.py --port 8010

# 另开终端：先建 20 个不同会话（输出 session_ids），再压测
uv run python scripts/seed_loadtest.py --count 20 --prefix loadtest-20261022 `
  --out eval/report/loadtest-sessions.json
uv run python scripts/load_test.py --base-url http://127.0.0.1:8010 `
  --concurrency 20 --rounds 20 --session-ids <上面的 session_ids> `
  --out eval/report/loadtest-20261021.json
# 清理（同一 prefix）
uv run python scripts/seed_loadtest.py --cleanup --prefix loadtest-20261022
```

脚本先 `GET /health` 校验 `llm_provider=fake`（否则退出 2，**不允许用真实模型压测**），并在前后各读一次
`fake_llm_calls` 计数器，差值写 `fake_llm_calls`（不再用“成功请求数”冒充）。然后并发打 `/chat/stream`，
统计错误率、首 token / 总耗时 p50、p95 与吞吐；输出 JSON + `.cases.jsonl`（逐请求原始数据，可独立重算）
+ Markdown + manifest。manifest 记录 server 的真实库名、嵌入/世界书/trace 配置与 trace 输入哈希。
`usage.llm_calls=0`。fake LLM/嵌入只测服务链路，**不代表真实模型/检索质量延迟**；该配置与生产差异见交接文档 §5。
无 `httpx` 时写 `skipped` 报告并退出 0。

### 删除语义

- **删除角色**：级联删除该角色的全部会话与消息，并清理其记忆（含各会话摘要）与 checkpoint；
  同时自动退出其所在的所有群，并移除其在群内的发言（DB 外键级联）。
- **删除会话**：仅删该会话的消息、checkpoint 与滚动摘要；**保留角色及其跨会话的语义记忆**。
- **删除群**：删除该群全部消息与成员关系，并清理每个成员在该群的 checkpoint。
- **移出群成员**：移除该成员并清理其在该群的 checkpoint；若群内成员被清空，该群及其消息一并删除。
- **删除世界书**：软删——`lore_docs` 与该 doc 的全部 `lore_chunks` 置 `deleted` 并清空 `embedding` / `tsv`
  （保留行以备审计与增量重建）；删除角色时其世界书由外键 `ON DELETE CASCADE` 一并删除。

## 效果与评测（数字 / 样本 / 口径 / 出处）

> 规则：每个数字都能追到 `eval/report/` 的 JSON/manifest、`doc/` 下的报告，或 `pytest -q` 输出；带样本量、基线、配置口径与保留项。V2.1 只有 `.md` 报告、1470ms 只有审查方复跑记录、490 只有 pytest 输出——这些行的「出处」列即为证据类型。
> 汇总归档见 `doc/eval-report-v2.4.md`；决策理由见 `doc/decisions/README.md`；真实失败与回归见 `doc/failure-log.md`。
>
> **说明**：`doc/` 与 `eval/report/` 属于本地技术文档与评测产物，不随仓库发布；上文与表格中出现的这些路径仅作记录，仓库内没有对应文件，可对照 `eval/suites/`、`tests/` 与提交历史复核。

### 世界书检索（V2.1，40 条标注查询）

| 指标 | 数字 | 样本 / 配置口径 | 基线 | 出处 |
|---|---:|---|---|---|
| hit@1 | **1.0000** | 40 条 `retrieval_lore`（专名 12 / 指代 12 / 多跳 8 / 别名 8）；real bge-m3 + bge-reranker-v2-m3；`roleplay_test`；agent 1；2 篇文档 20 chunks | 纯向量 A 组 0.8500 | `doc/V2/V2.1-消融报告.md`（报告，无 manifest）；`eval/report/ablation-20261018-220739.md` |
| MRR | **1.0000** | 同上 | 0.9208 | 同上 |
| nDCG@10 | **0.9873** | 同上 | 0.9229 | 同上 |
| hit@3 | 1.0000 | 同上；四组都已饱和（A 组也是 1.0000），**不用于区分增益** | 1.0000 | 同上 |
| 平均注入 token | 1550.7 | 同上；A 1552.5 → D 1550.7（C 组峰值 1560.7），rerank 只改排序不改条数 | 1552.5（A） | 同上 |
| rerank 延迟代价 | +4.5s/查询 | 同上；B 145.3ms → C 4633.2ms（CPU CrossEncoder） | — | 同上 §3.2 |

### 工具调用（V2.2，30 条 skill_use）

| 指标 | 数字 | 样本 / 配置口径 | 基线 | 出处 |
|---|---:|---|---|---|
| trigger_accuracy | **1.0** | 30 条（5 工具 × 5 + no-tool 5）；`web_search=fake`、`read_lore` 真 Retriever（rewrite 关）、`set_reminder` 真库；`deepseek-chat` | `eval/baseline/v2.2-tools.json` 同为 1.0（delta 0） | `eval/report/skill-use-20261020-223131.json` + `.md` |
| argument_accuracy | **1.0** | 同上 | 1.0 | 同上 |
| no_tool_accuracy | **1.0** | 同上 | 1.0 | 同上 |
| unexpected_error_rate | **0.0** | 同上 | 0.0 | 同上 |
| 失败矩阵 | 执行器 10 类闭集 + 工具级 4 类 | 执行器：`invalid_arguments / unknown_tool / timeout / upstream_error / empty / truncated / loop_detected / unavailable / context_missing / internal`；工具级：`invalid_timezone / invalid_dice / invalid_time / invalid_content` | — | ADR-0010 §2.1、§2.6；`tests/test_skills_executor.py` |
| MCP | `list_tools=5` | `scripts/mcp_smoke.py`；审查环境沙箱未跑 stdio（进程内等价验证） | — | ADR-0010 §2.5 |

> 工具基线不涨分（基线已是 1.0），这不是"提升"，是"没有回归"；失败矩阵的价值在 `error_type` 闭集与"不 500"，不在准确率。

### 评测 / 稳定性 / 成本（V2.4，HEAD `209371a`；6118 行为早期对照）

| 指标 | 数字 | 样本 / 配置口径 | 基线 | 出处 |
|---|---:|---|---|---|
| 单测 | **490 passed, 0 skip** | `uv run pytest -q`；`209371a`（V2.5 在 `09607ae` 复核） | 473（V2.4 首轮） | `uv run pytest -q` 输出（无 manifest）；审查报告 §14.1 |
| golden `--ci` | isolation_leaks=0 / memory_hit=1.0 / forbid=0 | 全量 golden；0 LLM；`EMBEDDING_PROVIDER=fake`；`git_dirty=false` | `eval/baseline/v2.3.json` 同值 | `eval-20261022-211920.json` + `.manifest.json` |
| memory | conflict/profile/recall@5/fact@5 全 **1.0** | 40 条；`bge-m3`；0 LLM；分母 memory 15 / fact 5 | `eval/baseline/v2.3-memory.json` 同值 | `memory-20261022-211927.json` + `.manifest.json` |
| 20×20 p95 首 token（完整图） | **1457ms** | 400 请求（20 并发 × 20 轮）；`LLM_PROVIDER=fake` + `EMBEDDING_PROVIDER=fake` + `LORE_ENABLED=false` + extraction ON；门槛 1500ms；余量 43ms | extraction OFF 752ms（`r1a-clean`） | `loadtest-20261022-r1b-clean.json` + `.manifest.json` |
| 20×20 p95 首 token（审查复跑） | 1470ms | 同上，审查方独立 server；余量 30ms | — | 审查报告 §14.3（复跑记录，无 manifest） |
| 真实 bge 路径（对照） | **6118ms** | 400 请求；`LORE_ENABLED=true`、真实 bge-m3；manifest `git_sha=462f193`、`dirty=true`、`database=null`；**早期对照，不是 `209371a` 证据，也不是本版门槛** | — | `loadtest-20261022` |
| cost/turn | 0.00685 | 1 轮真实冒烟；prompt 3005 / completion 105；`cjk1_other4` **estimator，不是账单** | — | `trace-smoke.json`（`9e912e7`） |
| adversarial（sample 4） | refusal 1.0；persona 1.0 / 0.75 | 8 次判定（4 refusal + 4 persona）；`deepseek-chat` judge r2；**只抽样，不是全量** | v2.3 全量 1.0 | `eval-20261022-211500` / `211247` |
| trace 覆盖 | 1/1，23 字段齐全 | 1 轮真实 LLM；`trace_version=1.0` | — | `trace-smoke`；审查报告 §3.2 |

**口径与保留（必读）**

- **1457/1470ms 是轻量栈**：fake LLM + fake embedding + 关世界书，只证明服务链路；真实 bge 路径早期对照是 6118ms（manifest `462f193`、dirty=true，**非 209371a 证据**）、真实冒烟首 token 11981ms。不得当作生产 SLA。
- **余量小**：距 1500ms 只有 30–43ms，宿主噪声约 ±10%，早前同配置有 1485–1617ms 波动（E-01）。
- **sample 4 不是全量零回归**：全量 adversarial（套件 40 条）与 `--suite all` 都未跑，结果只作冒烟；**E-02 按 sample-only 结项，本版不作全量零回归声明，跨版本比较仅限 sample 口径**；`eval/baseline/v2.4.json` 因无全量分母不冻结，对照仍用 `v2.3.json` / `v2.3-memory.json`（决策见 `doc/V2/Ai-agent-V2.5-收尾与对外表达-交接文档.md` §7.1）。
- **cost 是估算**：`cjk1_other4` + 示例单价，部署必须按账单覆盖 `.env`；fake trace 不计成本。
- **hit@3 饱和**：世界书 hit@3 四组都是 1.0，真正的增益看 hit@1 / MRR / nDCG@10。
- **旧版本**：V2.0–V2.3 的数字与文件指针见 `doc/eval-report-v2.4.md` §7；没有跑过的指标不补写。

## 前端（最小聊天页）

Vite + React + TS，顶部切换三个入口：

- **单人对话**：选角色 → 发消息 → SSE 流式显示回复；支持「自主闲聊」与消息撤回。
- **群聊**：新建群（勾选角色，≤ `GROUP_MAX_MEMBERS`）→ 发消息 → 各角色按 `speaker/turn_done`
  依次流式发言；支持移出成员与删除群。
- **角色管理**：新建 / 编辑 / 删除角色，上传头像，清除角色记忆。

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
```

- 开发服务器把 `/api` 与 `/static` 代理到 `http://127.0.0.1:8000`（见 `vite.config.ts`），**需先起后端**。
- `npm run build` = `tsc --noEmit` 类型检查 + 打包到 `frontend/dist`。
- 消息只存在前端内存：单聊按角色各自独立会话（切换保留），群聊切换群即清空；
  刷新均不加载历史（后端 `messages` / `group_messages` 保留快照，前端暂未拉取）。

## 测试

```bash
uv run pytest -q
```

- 不依赖真库的单测（图编排、prompt、service、SSE 路由）直接运行。
- 真库集成测试需 `TEST_DATABASE_URL`（默认指向 `roleplay_test`）可用，否则自动跳过。

## CI

`.github/workflows/ci.yml`：拉起 pgvector 服务 → 建测试库 → 迁移主库与测试库 → `uv run pytest -q`。

CI 通过 `--no-group embeddings` 跳过本地嵌入的重依赖（sentence-transformers/torch，约 2.5GB）；
测试一律使用 FakeEmbeddings，不需要真实 embedding 模型。本地开发照常用 `uv sync` 即可（默认含 embeddings 组）。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 前端打开后没有任何角色 | 未执行第 5 步种子脚本。运行 `uv run python scripts/seed_personas.py`，或到「角色管理」手动新建 |
| 发消息报错 / 长时间无回复 | `.env` 中 `LLM_API_KEY` 为空或无效；填好后重启后端 |
| Windows 启动报事件循环错误 | 必须用 `uv run python scripts/run_server.py`，不要直接用 `uvicorn` CLI |
| 首次启动很慢 / 卡住 | 正在下载 `bge-m3` 嵌入模型或安装 torch，属正常现象 |
| `pytest` 报 `ModuleNotFoundError` | 用了系统 python；请统一用 `uv run pytest` |
| 头像 404 | 确认后端已在运行；开发期头像经前端 `/static` 代理访问 |
