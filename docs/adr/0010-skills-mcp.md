# ADR-0010：Skill 协议、渐进披露的图内工具循环，以及 MCP 暴露

- 状态：**已接受**（V2.2 · Skill 工具层）
- 时点：2026-10-20
- 关联：V2.2 契约 §2.1~§2.7、§2.9、§2.10；D1（工具轮不流式）、D3（SearchProvider 可注入）、
  D5（`allowed_skills` TEXT[]）、D6（MCP 上下文用环境变量）、D7（read_lore 关改写）、
  D8（同请求缓存 / 最多 3 轮 / 末轮 `tool_choice="none"`）、D9（新 suite 不进 `run_eval`）；
  ADR-0002（LangGraph）、ADR-0009（世界书三路召回）
- 影响范围：`src/app/skills/**`（base/registry/executor/builtin/mcp_server）、
  `src/app/agents/{state,graph}.py`、`src/app/agents/nodes/{generate,tools,compose_prompt}.py`、
  `src/app/llm/{base,openai_compat}.py`、`src/app/config.py`、`src/app/main.py`、
  `migrations/versions/0006_skills_reminders.py`、`eval/suites/{skill_use,agentic_rag}.jsonl`、
  `scripts/{run_skill_eval,run_agentic_rag,mcp_smoke}.py`

---

## 1. Context

V2.0/V2.1 的角色回复只有一条"静态"信息通路：`retrieve_memory`（记忆）+ `retrieve_lore`
（世界书）。世界书三路召回虽然有预算配额，但**注入量受 `LORE_STATIC_MODE=full` 的 20% 预算硬约束**，
长尾设定（某个招牌名、某个道具的由来）经常挤不进 prompt。模型拿不到设定时只能"凭印象编造"，
这正是人设一致性的主要失分点（V2.0 评测里 persona 维的常见失败原因）。

需要在"静态注入"之外，给模型一条**按需取用**的能力：模型自己判断"这个问题我要查设定"，
主动调用检索；同时这条能力不能把"不使用工具"的角色体验改坏。

约束（来自 V2.2 §1.3 零回归不变量）：

- `allowed_skills=[]` 时，`compose_prompt` 输出必须与 V2.1 **逐字一致**（不得出现 `【可用工具】` 段）；
- `skill_registry=None` 时，图拓扑必须与 V2.1 一致（不得出现 `tools` / `should_continue` 节点）；
- 无工具时 `generate` 仍走原 `astream` 流式路径，SSE `text`/`done` 时序不变；
- 工具失败不得让请求 500，最终回复必须保持角色口吻。

## 2. Decision

### 2.1 工具是"协议 + 统一结构化失败"，执行器收敛一切异常

`Skill` 用 `Protocol` 描述（`name/description/when_to_use/parameters/handler/timeout_s/max_retries/side_effect/examples`），
`BaseSkill` 提供默认值与共享 `to_openai_tool()`（`additionalProperties=false`）。工具的返回值统一为
`ToolResult(ok, data, error_type, message, latency_ms, cached)`——**失败也走返回值，不抛异常**。

`ToolExecutor` 实现契约 §2.2 的七条硬要求（逐条有单测）：

1. 参数校验 `model_validate` → 失败 `invalid_arguments` + `data={"fields":{...}}`；
2. 超时 `asyncio.wait_for(handler, skill.timeout_s)`，按 `max_retries`（默认 1）指数退避 0.5s；
3. 上游错误（`UpstreamError`/5xx/429）重试 1 次，仍失败 → `upstream_error` 并**对该工具熔断 60s**；
4. 结果超 4000 token → 截断（优先裁最大列表字段，再截长字符串）+ `summary`，`ok=True`、`truncated=True`；
5. **同请求缓存**：`sha256(name + canonical_json(arguments))`，命中直接 `cached=True`；
6. 循环保护：`tool_rounds >= max_tool_rounds` 时不再进 `tools`，末轮 `tool_choice="none"`；
7. 异常兜底：任何未捕获异常 → `internal` + 日志，**绝不让请求 500**。

10 个 `error_type` 为闭集（§2.6）：`invalid_arguments / unknown_tool / timeout / upstream_error /
empty / truncated / loop_detected / unavailable / context_missing / internal`。

### 2.2 渐进披露：空白名单 = 零变化

`allowed_skills` 解析优先级为 `state.get("allowed_skills") or 图构造时的 allowed_skills or []`
（compose_prompt / generate / tools 三处一致）：

- **空**：`compose_prompt` 不追加 `【可用工具】` 段；`generate` 走 V2.1 原 `astream` 路径；
  `build_graph(skill_registry=None)` 不挂 `tools`/`should_continue` 节点。旧角色行为**零变化**。
- **非空**：`compose_prompt` 只追加 `name：when_to_use`（不放 JSON Schema）；完整 schema 在
  `generate` 绑定时通过 `registry.to_openai_tools(allowed)` 下发。

"渐进披露"体现在两层：prompt 里只给**摘要**（省 token），schema 只在真正请求工具时下发；白名单
为空时模型根本不知道有工具存在。

### 2.3 图内工具循环有界，且工具消息不污染状态

`compose_prompt → generate → should_continue`，`should_continue` 命中 `tool_calls` 且未达上限时
进 `tools → generate`：

- `generate` 工具模式下若只回 `tool_calls`，返回 `reply=""`、**不写空 AIMessage**；
- `tools` 节点逐个执行并把结构化结果累计进 `state.tool_results`（`operator.add`），`tool_calls` 清空、
  `tool_rounds + 1`；单轮多工具按顺序执行，一个失败不阻断其他；
- **assistant(tool_calls) / tool 消息只存在于 `generate` 的本地 LLM 消息列表**，不写入 `state.messages`，
  避免污染 prompt 拼装、记忆写入与快照；
- D1：工具轮**不流式**；最终文本以一次 `text` 事件下发，`done` 时序不变（避免在 `astream` 里解析
  OpenAI tool_call 增量，把 token 级流式留到后续版本）。

### 2.4 `read_lore` 复用 `Retriever`，并把"静态注入"做成三策略开关

`read_lore` 复用 V2.1 的 `Retriever`（含 bge-m3 向量 + 中文 FTS + bge-reranker），**不改检索算法**；
按 D7 关闭 query 改写（`rewrite_enabled=False`）——工具 query 已由模型给出，工具内部再调 LLM 会造成
递归与额外延迟。

为此新增 `LORE_STATIC_MODE: str = "full"`（默认保持 V2.1 行为），在 `graph._retrieve_lore` 分派：

| 模式 | 行为 |
|---|---|
| `full` | V2.1 原路径（always-on + keyword + vector + RRF + rerank + 预算） |
| `always_on` | 只召回 `always_on` 文档的 chunk（`source_ref` 升序、8% 预算），不走向量/关键词/rerank |
| `off` | 直接返回 `{"recalled_lore": []}`，**不访问数据库** |

`always_on` 复用 `Retriever` 既有的 `_route_always_on / _load_rows / _fit_budget`，不新增检索实现。
三种模式支撑 §2.10 的 static / agentic / hybrid 对比实验。

### 2.5 MCP 只做"能被列出、能被调用"，上下文用环境变量显式注入

`app.skills.mcp_server` 用 `mcp` SDK 起 stdio server，`list_tools` 返回 5 个工具的 OpenAI 兼容
schema，`call_tool` 返回 `ToolResult` 的 JSON（失败不抛异常）。D6：stdio 无请求上下文，用
`MCP_AGENT_ID` / `MCP_SESSION_ID` 注入；缺失时**有状态工具**（`read_lore` / `set_reminder`）返回
`context_missing`，其余照常可用。

`web_search` 走可注入的 `SearchProvider`（D3）：无 key 时 `HttpSearchProvider` 返回 `unavailable`，
**不把外部搜索服务的可用性变成验收阻塞**；评测用 fake provider。

### 2.6 评测独立于 `run_eval`（D9）

`skill_use`（30 条）/ `agentic_rag`（20 条）**不进 `run_eval.SUITE_NAMES`**，走独立脚本
`scripts/run_skill_eval.py` / `scripts/run_agentic_rag.py` + 独立基线文件，避免改变 V2.0/V2.1 报告的
`dev:test` 统计；PR CI 仍只跑 `golden --ci`。

## 3. 明确不做

- 不做工具 token 级流式（D1）；不做前端工具确认 UI；不做外部 MCP server 的硬依赖；
- 不改 V2.1 检索算法（RRF/rerank/预算一律不动）；
- 不改 `eval/suites` 里的 5 个旧 suite，不改 `eval/metrics.py` / `eval/split.py` 的既有语义；
- 不让 `skill_use` / `agentic_rag` 进 PR CI。

## 4. Consequences

**正面**

- 旧角色零回归：空白名单逐字一致、`skill_registry=None` 拓扑不变、无工具仍走 `astream`（三条不变量有单测）。
- 失败是数据不是异常：10 个 `error_type` 闭集 + 执行器七条硬要求，工具故障不会变成 500。
- 5 个工具彼此独立、可注入依赖（`SearchProvider` / `session_factory` / `Retriever`），测试无需真网络/真 key。

**代价 / 风险**

- 工具轮增加延迟：`read_lore` 含 CPU 交叉编码，实测单次 8s 量级（`timeout_s` 从默认 5s 放宽到 15s，
  见 §5 偏差记录）；`agentic` 策略首字延迟明显高于 `static`（见 agentic-rag 报告）。
- `always_on` 策略的实际收益取决于 `lore_docs.always_on` 语料：当前夹具两篇 doc 均为 `False`，
  故 hybrid 的静态注入为空（记录为发现，见 agentic-rag 报告）。
- 工具循环引入 3 轮上限与缓存，语义边界（同请求 vs 跨请求）需在实现与测试里明确。

## 5. 验收

- `tests/test_skills_{contract,registry,executor}.py`：协议/注册表/执行器七条硬要求；
- `tests/test_tools_{basic,lore_reminder}.py`：5 工具正常/参数错/超时/空结果；
- `tests/test_graph_tools.py`：循环保护与空白名单不变量；`tests/test_agentic_rag.py`：实验结构；
- `scripts/mcp_smoke.py`：`list_tools=5`、`get_current_time` 调用成功、退出 0；
- `scripts/run_skill_eval.py` 报告：`trigger_accuracy / argument_accuracy / no_tool_accuracy /
  unexpected_error_rate`（门禁见交接文档 §4.2）。
