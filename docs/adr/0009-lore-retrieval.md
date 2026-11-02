# ADR-0009：世界书三路召回、预算配额，以及为何不复用 LangGraph Store

- 状态：**已接受**（V2.1 · 世界书知识库）
- 时点：2026-10-18
- 关联：V2.1 契约 §4.3~§4.9、§5 T1~T8；ADR-0001（记忆隔离）、ADR-0004（群聊可见性）
- 影响范围：`src/app/lore/**`、`src/app/db/models.py`（新增 `LoreDoc` / `LoreChunk`）、
  `migrations/versions/`、`src/app/agents/{state,graph}.py`、`src/app/agents/nodes/{retrieve_lore,compose_prompt}.py`、
  `src/app/api/v1/lore.py`

---

## 1. Context

V2.0 的记忆（语义记忆 + 情景摘要）落在 **LangGraph `AsyncPostgresStore`**：namespace
`(f"agent:{agent_id}", "memories")`，向量索引建表时定死 `dims=1024`，检索走
`Store.search` 的向量 top-k（写死字段 `["text"]`）。

世界书（role 设定 / 世界观资料）与"记忆"在**语义与生命周期**上不同：

| 维度 | 记忆（V2.0） | 世界书（V2.1） |
|---|---|---|
| 作者 | 对话中自动抽取 | **运营者手写**（导入 Markdown） |
| 生命周期 | 只增、可覆盖 | 导入 / 更新（version+1）/ 软删 / 增量重建 |
| 结构 | 扁平的 `(agent_id, content)` | 文档 → section → chunk，带 `header_path` / `seq` |
| 引用稳定性 | 无外部引用 | **`source_ref` 稳定**，评测与 preview 都按它对齐 |
| 召回方式 | 向量 | always-on + 关键词（中文 FTS）+ 向量，三路互补 |
| 常驻 | 无 | 有（`always_on`，核心设定必须常驻） |

## 2. Decision

### 2.1 世界书落在**独立业务表**，不进 LangGraph Store

新增 `lore_docs` / `lore_chunks` 两表（迁移），`lore_chunks` 保存
`header_path / content / token_count / keywords / content_hash / status / embedding / tsv`。

**不复用 `AsyncPostgresStore` 的理由（逐条）**：

1. **没有更新/删除语义**：Store 是 KV + 向量索引，写即覆盖，**无 version、无软删、无增量 diff**。
   世界书要求"同 slug 更新 version+1""多余 `source_ref` 软删""幂等重导不变"，Store 表达不了。
2. **没有全文检索**：Store 只有向量字段，无法承载中文 FTS（`tsv @@ to_tsquery`，见 ADR-0007）。
   关键词路对专名/数字/引号别名是刚需。
3. **没有稳定外部 id**：Store 的 key = `sha1(f"{agent_id}:{content}")`，**随内容变而变**；
   评测用例与 preview 需要跨环境可复现的 `source_ref = f"{slug}:{seq:02d}"`，Store 给不了。
4. **向量索引维度/字段建表后不可改**：Store 索引与记忆共用 `dims=1024 / fields=["text"]`。
   把世界书塞进同一 namespace 会让"记忆召回"与"设定召回"**互相污染**，直接威胁
   ADR-0001 / ADR-0004 的隔离不变量。
5. **没有 always-on 与 header_path**：常驻设定与分层标题路径是结构信息，不是向量能表达的。

> 一句话：Store 是"记忆的向量抽屉"，不是"可版本化的设定库"。

### 2.2 三路召回 → RRF → rerank → 预算

- **always-on**：`lore_docs.always_on = true` 的 chunk，按 `source_ref` 升序，**先占预算**（≤8%）。
- **keyword**：jieba 分词 → `tsv @@ to_tsquery('simple', '词1 | 词2')`，`ts_rank_cd` 降序，上限 20；
  无可用词（全是单字/停用词）时**跳过该路、不报错**。
- **vector**：`ORDER BY embedding <=> :q` 余弦距离升序，上限 20。
- **RRF**：`score(d) = Σ_r 1/(60 + rank_r(d))`，并列按 `source_ref` 升序（确定性）。
  取 top-20 进入 rerank。
- **rerank**：`BgeReranker` 懒加载 `BAAI/bge-reranker-v2-m3`；**加载失败自动降级 `NullReranker`**
  （保持 RRF 顺序，`rerank=false`，**不抛异常打断对话**）。
- **预算**（见 2.3）后得到最终注入列表，每行 `（header_path）content`。

**为什么不只做向量路**：向量对专名/数字/引号别名不敏感（"三不卖""何掌柜"）。
三路互补是契约 §4.4 的硬要求，消融（§8.3）用于量化每路的边际贡献。

### 2.3 预算配额与裁剪顺序（契约 §4.7）

总预算 = `PROMPT_TOKEN_BUDGET`（默认 8000）。persona 必留，其余段按上限分配；
超预算时按固定顺序裁剪：

| 段落 | 上限 | 裁剪顺序 |
|---|---|---|
| persona（system 基座） | 必留 | 不裁 |
| always-on 设定 | 8% | ① 按 `source_ref` **倒序**丢 |
| 滚动摘要 | 无独立段上限 | ② 整段丢 |
| 语义记忆 | 无独立段上限 | ③ 整段丢 |
| 检索 lore | 20% | ④ 按 **rerank 分升序**丢 |
| 近 N 轮 | 剩余 | ⑤ 从**最旧**丢 |

> 只有 always-on 8% 与 lore 20% 是生效的段上限；摘要 / 语义记忆沿用 V2.0，无独立段上限、整段丢（M2 口径，与交接文档 §4.7 一致）。

**关键不变量**：`recalled_lore` 为空时，`compose_prompt` 的返回值必须与 V2.0 **逐字一致**
（固定 state 快照断言）。这是 V1 零回归的核心证据。

## 3. 明确不做

- 不把 Store 的记忆搬进新表、不改 `MemoryStore.search` 语义（隔离不变量不动）。
- 不支持 PDF / DOCX 等非 Markdown 导入（见 ADR-0007）。
- 不做跨 agent 共享设定、不做设定之间的引用解析（多跳靠召回，不靠图）。

## 4. Consequences

**正面**

- 设定与记忆**物理隔离**：两张新表 + 独立向量列，ADR-0001/0004 的不变量不受影响。
  迁移为**纯新增**，可回滚，不触碰既有数据。
- `source_ref` 稳定 → 评测可跨环境复现、preview 可解释（三路/融合/重排/预算全可见）。
- 降级路径完备：rerank 缺失 → RRF 顺序；embedder 缺失 → 仅关键词路；无设定 → 与 V2.0 逐字一致。
- 无 lore 时图与 prompt零改动：`retrieve_lore` 在 `retriever is None` / `lore_enabled=False`
  时返回 `{"recalled_lore": []}`，**不访问数据库**。

**代价 / 风险**

- 多一条检索链路与一张迁移表，测试面变大（parser/chunker/keywords/indexer/retriever/service/api/graph）。
- 向量列在 `lore_chunks` 与记忆 Store 索引**各存一份**嵌入，写入时各有一次 embedding 调用
  （导入是低频操作，可接受）。
- rerank 模型 2GB+，必须懒加载；首次调用卡 10~30s，需超时兜底（契约 §10.5）。
- 预算裁剪顺序一旦确定即被快照测试锁定，后续调整需另立 ADR。

## 5. 验收

- `tests/test_lore_*.py`（≥4 文件）：解析/切分、索引幂等与增量、三路+RRF+预算、服务/API 结构。
- `tests/test_lore_prompt.py`：无 lore 时 `compose_prompt` 与 `0a3938c` 逐字一致（快照）。
- `eval/suites/retrieval_lore.jsonl`：40 条真实库检索用例；数字见交接文档 §8.3 消融表。
