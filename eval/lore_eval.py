"""世界书检索评测（契约 4.10 / T7）：真库 + 真嵌入跑**真实 Retriever**。

与 `lore_fixture.py`（V2.0 的确定性 fixture 检索）不同，本模块评测的是 V2.1 的三路召回
+ RRF + rerank + 预算链路，因此必须有真库与真向量，否则一律跳过。

- 语料：`eval/fixtures/lore_long/<slug>.md`（每篇 ≥2000 字、10 个 `##` 小节）；
- 导入：`Indexer.upsert_doc` 把语料写入评测角色（`DEFAULT_AGENT_ID`，幂等可重复跑）；
- 指标：对 `RetrievalResult.reranked` 的 `source_ref` 排序算 hit@1 / hit@3 / mrr / ndcg@10；
- 跳过（任一满足）：连不上真库 / 无 sentence-transformers / 目标角色 lore 表为空。
  跳过时四个指标一律 `null`（不是 `0.0`），`skipped=true` 并写 `skip_reason`。
"""

from __future__ import annotations

from pathlib import Path

from metrics import hit_at_k, mrr, ndcg_at_k, pass_rate

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "lore_long"

# 评测专用角色：两篇 fixture 都导入到此 agent_id（套件用例的 agent_id 亦为它）
DEFAULT_AGENT_ID = 1

METRIC_NAMES = ("hit@1", "hit@3", "mrr", "ndcg@10")


def load_fixtures() -> list[tuple[str, str, str]]:
    """读取 `lore_long` 下每篇 Markdown → (slug, title, content)。

    slug 取文件名 stem；title 取首个 `# ` 标题（无则退回 slug）。
    """
    fixtures: list[tuple[str, str, str]] = []
    for path in sorted(FIXTURE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        fixtures.append((path.stem, title, text))
    return fixtures


def skipped_metrics(reason: str) -> dict:
    """跳过形态：四个指标为 `null`（不是 `0.0`），`queries=0`、`rerank=false`。"""
    row: dict = {name: None for name in METRIC_NAMES}
    row.update({"queries": 0, "rerank": False, "skipped": True, "skip_reason": reason})
    return row


def measure(
    cases: list[dict], ranked_by_case: dict[str, list[str]], *, rerank: bool
) -> dict:
    """由每例的检索排序算四个指标；返回 {"metrics", "failures"}。"""
    hits1: list[float] = []
    hits3: list[float] = []
    ranks: list[float] = []
    ndcgs: list[float] = []
    failures: list[dict] = []
    for c in cases:
        ranked = ranked_by_case.get(c["id"], [])
        expected = set(c["expect_refs"])
        h1 = hit_at_k(expected, ranked, 1)
        h3 = hit_at_k(expected, ranked, 3)
        hits1.append(h1)
        hits3.append(h3)
        ranks.append(mrr(expected, ranked))
        relevances = [1.0 if ref in expected else 0.0 for ref in ranked]
        ndcgs.append(ndcg_at_k(relevances, 10))
        if h1 == 0.0:
            failures.append(
                {
                    "suite": "retrieval_lore",
                    "id": c["id"],
                    "reason": f"top-1 未命中（top-3 {'命中' if h3 else '未命中'}）",
                }
            )
    metrics = {
        "hit@1": pass_rate([h == 1.0 for h in hits1]) if hits1 else None,
        "hit@3": pass_rate([h == 1.0 for h in hits3]) if hits3 else None,
        "mrr": (sum(ranks) / len(ranks)) if ranks else None,
        "ndcg@10": (sum(ndcgs) / len(ndcgs)) if ndcgs else None,
        "queries": len(cases),
        "rerank": rerank,
        "skipped": False,
        "skip_reason": None,
    }
    return {"metrics": metrics, "failures": failures}


async def ensure_agent(session_factory, agent_id: int = DEFAULT_AGENT_ID) -> int:
    """确保评测角色存在（`lore_docs.agent_id` 有 FK 到 `agents.id`）。

    显式插入 id 后需把序列推到当前最大值，避免后续自增插入撞上同一 id。
    """
    from sqlalchemy import select, text

    from app.db.models import Agent

    async with session_factory() as db:
        existing = (
            await db.execute(select(Agent.id).where(Agent.id == agent_id))
        ).scalar_one_or_none()
        if existing is None:
            db.add(
                Agent(
                    id=agent_id,
                    name=f"lore-eval-agent-{agent_id}",
                    persona_prompt="评测专用角色（世界书检索，不参与对话）。",
                )
            )
            await db.execute(
                text(
                    "SELECT setval(pg_get_serial_sequence('agents', 'id'), "
                    "(SELECT COALESCE(MAX(id), 1) FROM agents), true)"
                )
            )
            await db.commit()
    return agent_id


async def import_fixtures(session_factory, embedder, agent_id: int = DEFAULT_AGENT_ID) -> int:
    """把 `lore_long` 语料导入指定角色，返回落库 chunk 总数。"""
    from app.lore.indexer import Indexer

    await ensure_agent(session_factory, agent_id)
    indexer = Indexer(session_factory=session_factory, embedder=embedder)
    total = 0
    for slug, title, content in load_fixtures():
        result = await indexer.upsert_doc(
            agent_id=agent_id, slug=slug, title=title, content=content
        )
        total += result.chunks
    return total


async def count_active_chunks(session_factory, agent_id: int = DEFAULT_AGENT_ID) -> int:
    """该角色 `status='active'` 的 chunk 数（0 → 触发 no_lore_docs 跳过）。"""
    from sqlalchemy import func, select

    from app.db.models import LoreChunk

    async with session_factory() as db:
        return int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(LoreChunk)
                    .where(
                        LoreChunk.agent_id == agent_id,
                        LoreChunk.status == "active",
                    )
                )
            ).scalar_one()
        )


async def run(cases: list[dict], retriever) -> dict:
    """逐例跑真实 Retriever，用 reranked 排序算指标；返回 {"metrics", "failures"}。"""
    ranked_by_case: dict[str, list[str]] = {}
    rerank_any = False
    for c in cases:
        result = await retriever.retrieve(agent_id=c["agent_id"], query=c["query"])
        ranked_by_case[c["id"]] = [ref for ref, _ in result.reranked]
        rerank_any = rerank_any or bool(result.rerank)
    return measure(cases, ranked_by_case, rerank=rerank_any)
