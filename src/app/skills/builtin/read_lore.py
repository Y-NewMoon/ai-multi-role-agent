"""`read_lore`（契约 §2.5-5）：复用 `Retriever`（关闭 query 改写），只回灌模型。"""

from pydantic import BaseModel, Field

from app.skills.base import BaseSkill, ToolContext, ToolResult


class ReadLoreArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)


def make_read_lore_skill(retriever=None) -> BaseSkill:  # noqa: ANN001
    async def handler(ctx: ToolContext, args: ReadLoreArgs) -> ToolResult:
        if retriever is None:
            return ToolResult(ok=False, error_type="unavailable", message="设定资料检索不可用")

        result = await retriever.retrieve(agent_id=ctx.agent_id, query=args.query)
        scores = {ref: score for ref, score in result.reranked}
        by_ref = {hit.source_ref: hit for hit in result.injected}
        # 候选顺序：rerank 分降序优先，其后补 always-on（injected）
        ordered: list[str] = [ref for ref, _ in result.reranked]
        for hit in result.injected:
            if hit.source_ref not in ordered:
                ordered.append(hit.source_ref)
                scores.setdefault(hit.source_ref, hit.score)
        if not ordered:
            return ToolResult(ok=False, error_type="empty", message="未命中任何设定资料")

        refs = ordered[: args.top_k]
        missing = {ref for ref in refs if ref not in by_ref}
        rows = {}
        if missing:
            # 复用同一检索器的取行逻辑（不改 RRF/rerank/预算）
            rows = await retriever._load_rows(agent_id=ctx.agent_id, refs=missing)

        chunks: list[dict] = []
        for ref in refs:
            hit = by_ref.get(ref)
            row = rows.get(ref)
            if hit is not None:
                chunks.append(
                    {
                        "source_ref": hit.source_ref,
                        "header_path": hit.header_path,
                        "content": hit.content,
                        "score": hit.score,
                    }
                )
            elif row is not None:
                chunks.append(
                    {
                        "source_ref": row.source_ref,
                        "header_path": row.header_path,
                        "content": row.content,
                        "score": float(scores.get(ref, 0.0)),
                    }
                )
        if not chunks:
            return ToolResult(ok=False, error_type="empty", message="未命中任何设定资料")
        return ToolResult(ok=True, data={"chunks": chunks})

    return BaseSkill(
        name="read_lore",
        description="检索角色设定资料（世界书）中与查询相关的片段，返回命中内容与相似度分数",
        when_to_use=(
            "当用户询问角色设定的具体事实（如店铺招牌、道具由来、人物身世、地点名称等）时使用；"
            "作答前先检索核实，不要凭印象编造"
        ),
        parameters=ReadLoreArgs,
        handler=handler,
        # 检索含 bge-reranker 交叉编码（离线 CPU 实测单次 8s 量级），
        # 沿用 BaseSkill 默认 5s 会被执行器判 timeout；此处按检索场景放宽。
        timeout_s=15.0,
    )
