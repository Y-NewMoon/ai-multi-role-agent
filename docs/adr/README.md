# 架构决策记录（ADR）

> 本目录只收录**影响架构走向**的 6 篇决策记录。完整 `doc/`（含交接文档、评测报告、待修清单）为本地技术文档，不随仓库发布——原因见下方「为什么只有这几篇」。

| # | 决策 |
|---|---|
| [0001](./0001-memory-isolation.md) | 记忆隔离只由 namespace 决定（不掺 session / 群聊） |
| [0002](./0002-langgraph-over-custom.md) | 编排层用 LangGraph StateGraph，而非自研状态机 |
| [0005](./0005-why-not-a2a-plan-execute.md) | 为什么不上 A2A / plan-execute（架构边界） |
| [0009](./0009-lore-retrieval.md) | 世界书三路召回、RRF 融合与预算配额 |
| [0010](./0010-skills-mcp.md) | Skill 协议、渐进披露的工具循环与 MCP 暴露 |
| [0012](./0012-v3-time-and-determinism.md) | 时间感全量化，以及「同 now 逐字稳定」的确定性口径 |

## 为什么只有这几篇

1. **可验证性**：这 6 篇对应代码里的硬约束（隔离、编排、工具、检索、确定性），读完之后能在 `src/` 与 `tests/` 里找到对应实现与断言；
2. **不含隐私**：不含真实对话内容、不含部署凭据；
3. **其余文档留在本地**：`doc/` 里的交接文档与 `eval/report/` 的评测产物涉及大量本地路径与中间数据，体积与噪声都大，且 README「效果与评测」一节已经给出数字、样本量、口径与保留项。

## 每个数字怎么复现

```powershell
# 单测（真库用例需 TEST_DATABASE_URL，否则自动 skip）
uv run pytest -q

# 确定性门禁（0 LLM 调用）
uv run python eval/run_eval.py --suite golden --ci
```

评测用例在 `eval/suites/`，指标实现在 `eval/metrics.py`；任何对外引用的数字都要求 manifest（`git_sha` + 配置 + 输入输出哈希）。
