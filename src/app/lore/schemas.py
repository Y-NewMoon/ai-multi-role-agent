"""世界书 DTO（契约 4.9）。"""

from datetime import datetime

from pydantic import BaseModel, Field

# slug 规则（契约 4.3）：小写字母 / 数字 / 下划线，1~40 位
SLUG_PATTERN = r"^[a-z0-9_]{1,40}$"


class LoreDocCreate(BaseModel):
    """导入 / 更新一篇设定（不接受请求带来的人设覆盖，契约 4.9）。"""

    agent_id: int
    slug: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    always_on: bool = False


class LoreDocImportOut(BaseModel):
    doc_id: int
    version: int
    chunks: int
    tokens: int
    changed: bool


class LoreDocRead(BaseModel):
    """列表 / 详情（含 chunk 数与 token 总数）。"""

    id: int
    agent_id: int
    slug: str
    title: str
    version: int
    always_on: bool
    status: str
    chunks: int
    tokens: int
    created_at: datetime
    updated_at: datetime


class ReindexIn(BaseModel):
    agent_id: int
    dry_run: bool = False


class ReindexOut(BaseModel):
    agent_id: int
    dry_run: bool
    documents: int
    chunks_rebuilt: int


class FusedItem(BaseModel):
    source_ref: str
    score: float


class InjectedItem(BaseModel):
    source_ref: str
    tokens: int


class BudgetInfo(BaseModel):
    lore_tokens: int
    lore_cap: int


class PreviewOut(BaseModel):
    """调试视图：三路命中 → RRF → rerank → 最终注入（契约 4.9）。"""

    query: str
    rewritten: str
    rewrite_used: bool
    routes: dict[str, list[str]]
    fused: list[FusedItem]
    reranked: list[FusedItem]
    rerank: bool
    injected: list[InjectedItem]
    budget: BudgetInfo
