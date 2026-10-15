# AI 多角色扮演智能体

基于 **LangGraph** 的多角色扮演聊天后端：为每个角色维护长期记忆（语义 + 滚动摘要），
按会话隔离对话历史，支持 **SSE 流式** 逐 token 输出。附一个 Vite + React 的最小前端
（单人对话 / 群聊 / 角色管理）。

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

- **编排**：LangGraph `StateGraph`（5 节点线性图：`load_persona → retrieve_memory → compose_prompt → generate → update_memory`）
- **存储**：PostgreSQL + pgvector（`pgvector/pgvector:pg16`）
  - 业务表（`agents` / `sessions` / `messages` 等）由 Alembic 管理
  - 对话状态用 `AsyncPostgresSaver`（checkpointer），记忆用 `AsyncPostgresStore`（向量索引）
- **LLM**：OpenAI 兼容接口（默认 DeepSeek）
- **Embedding**：本地 `bge-m3`，维度固定 **1024**（`EMBEDDING_DIMS`，建表后不可改）

## 目录结构

```
src/app/
  agents/        LangGraph 图与节点
  api/           FastAPI 路由（v1: chat / agents / sessions / groups / uploads）与依赖
  core/          异常与日志
  db/            SQLAlchemy 模型与引擎
  llm/           LLM / Embedding Provider
  memory/        记忆读写（Store 封装、namespace、召回、摘要）
  prompts/       人设与摘要 prompt 模板
  schemas/       Pydantic DTO
  services/      业务服务（chat / agent / session / group）
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
| `EMBEDDING_PROVIDER` | `local` | |
| `EMBEDDING_MODEL` | `bge-m3` | 本地嵌入模型 |
| `EMBEDDING_DIMS` | `1024` | 向量维度，**建表后不可改** |
| `PROMPT_TOKEN_BUDGET` | `8000` | 拼 prompt 的 token 预算 |
| `PROMPT_RECENT_ROUNDS` | `10` | 拼 prompt 时保留的最近轮数 |
| `SUMMARY_TRIGGER_ROUNDS` | `20` | 触发滚动摘要的轮数阈值 |
| `SUMMARY_CHUNK_ROUNDS` | `10` | 每次摘要覆盖的轮数 |
| `REPLY_MAX_MESSAGES` | `3` | 一次回复最多连发几条 |
| `GROUP_MAX_MEMBERS` | `8` | 群内角色数上限（不含用户） |
| `GROUP_MAX_TURNS` | `10` | 一轮群聊内最多发言次数 |
| `IDLE_GROUP_MAX_MESSAGES` | `10` | 群聊一次闲聊累计消息上限 |
| `IDLE_GROUP_TURN_MAX_MESSAGES` | `2` | 群聊闲聊时每角色每次最多发言条数 |
| `AVATAR_DIR` | `data/avatars` | 头像本地存储目录（相对项目根） |
| `AVATAR_URL_PREFIX` | `/static/avatars` | 头像静态服务挂载前缀 |
| `AVATAR_MAX_BYTES` | `2097152` | 头像单文件上限（2MB） |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/chat` | 非流式对话 |
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

### SSE 协议（`POST /api/v1/chat/stream`）

请求体：`{"session_id": 1, "message": "在吗"}`，响应 `text/event-stream`，事件类型：

- `token`：`data: {"text": "增量文本"}`（可多次）
- `done`：`data: {"session_id": 1, "agent_id": 2, "replies": [{"content": "...", "message_id": 9, "thinking": "..."}]}`
- `error`：`data: {"detail": "错误信息"}`（流开始后异常以事件表达，HTTP 仍为 200）

时序保证：逐 token 下发 → 图执行完成（记忆已落 Store）→ 写 `messages` 快照 → 才推 `done`。

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

### 删除语义

- **删除角色**：级联删除该角色的全部会话与消息，并清理其记忆（含各会话摘要）与 checkpoint；
  同时自动退出其所在的所有群，并移除其在群内的发言（DB 外键级联）。
- **删除会话**：仅删该会话的消息、checkpoint 与滚动摘要；**保留角色及其跨会话的语义记忆**。
- **删除群**：删除该群全部消息与成员关系，并清理每个成员在该群的 checkpoint。
- **移出群成员**：移除该成员并清理其在该群的 checkpoint；若群内成员被清空，该群及其消息一并删除。

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
