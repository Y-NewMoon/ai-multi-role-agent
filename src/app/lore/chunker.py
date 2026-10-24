"""切分与 token 估算（契约 4.2）：sections → chunks。

规则：
- section 估算 ≤ 500 token：整段一个 chunk；> 500：400 token 窗口、60 token overlap；
- 窗口边界优先落在 `。！？\\n` 之后，找不到再硬切；
- < 50 token 的 section 与相邻 section 合并（合并后仍超 500 则按窗口切）；
- `header_path = f"{doc.title} / {section_title}"`；section 无标题时 = doc.title；
- 两次切分必须逐位一致（确定性）；纯空白 section 不落库。

token 估算与 `compose_prompt._estimate_tokens` 同公式（CJK 1 字 1 token，其余 4 字 1 token），
由 `tests/test_lore_parser.py` 用 20 组样本断言逐值相等。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.lore.parser import ParsedDoc, Section

# 预算（契约 4.2 定值）
MAX_TOKENS = 500
WINDOW_TOKENS = 400
OVERLAP_TOKENS = 60
MIN_TOKENS = 50

# 断句优先的边界字符
_SENTENCE_ENDS = "。！？\n"


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def estimate_tokens(text: str) -> int:
    """与 `compose_prompt._estimate_tokens` 同公式的保守估算。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + (len(text) - cjk + 3) // 4


@dataclass
class Chunk:
    """一个检索单元（对应 `lore_chunks` 一行；`source_ref` 由 indexer 依 slug 生成）。"""

    seq: int
    header_path: str
    content: str
    token_count: int


@dataclass
class _Piece:
    """合并后的片段：起点标题路径 + 正文 + 更深标题锚点。"""

    base_path: list[str]
    text: str
    anchors: list[tuple[int, tuple[str, ...]]]


def _base_path(doc_title: str, section: Section) -> list[str]:
    """section 的起点标题路径：doc.title [+ section.title]。"""
    path = [doc_title] if doc_title else []
    if section.title:
        path.append(section.title)
    return path


def _concat(left: _Piece, right: _Piece) -> _Piece:
    """把右片并入左片（保留左片起点路径，锚点偏移右移）。"""
    shift = len(left.text)
    anchors = list(left.anchors) + [
        (off + shift, path) for off, path in right.anchors
    ]
    return _Piece(base_path=list(left.base_path), text=left.text + right.text, anchors=anchors)


def _header_path(piece: _Piece, offset: int) -> str:
    """取 `offset` 处生效的完整 header_path。"""
    deep: tuple[str, ...] = ()
    for off, path in piece.anchors:
        if off <= offset:
            deep = path
        else:
            break
    return " / ".join([*piece.base_path, *deep])


def _offset_for_tokens(text: str, start: int, target: int) -> int:
    """返回最小 `end > start` 使 `estimate(text[start:end]) >= target`；不足则返回 len(text)。"""
    if estimate_tokens(text[start:]) < target:
        return len(text)
    lo, hi = start + 1, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[start:mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _snap_to_sentence(text: str, start: int, end: int) -> int:
    """把窗口右边界左移到最近的 `。！？\\n` 之后；找不到则返回原 `end`。"""
    for pos in range(end, start, -1):
        if text[pos - 1] in _SENTENCE_ENDS:
            return pos
    return end


def _content_offset(raw: str, start: int) -> int:
    """内容 strip 后的起点在原文中的偏移（header_path 按内容起点取锚点）。"""
    return start + (len(raw) - len(raw.lstrip()))


def _windows(piece: _Piece) -> list[tuple[int, str]]:
    """把超长片段切成 (起点偏移, 文本) 窗口序列。"""
    text = piece.text
    if estimate_tokens(text) <= MAX_TOKENS:
        return [(0, text)]

    out: list[tuple[int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        end = _offset_for_tokens(text, start, WINDOW_TOKENS)
        if end < length:
            end = _snap_to_sentence(text, start, end)
        if end <= start:
            end = min(start + 1, length)
        out.append((start, text[start:end]))
        if end >= length:
            break
        advance = _offset_for_tokens(text, start, WINDOW_TOKENS - OVERLAP_TOKENS)
        if advance <= start:
            advance = start + 1
        start = advance
    return out


def chunk_document(
    parsed: ParsedDoc,
    *,
    max_tokens: int = MAX_TOKENS,
    window_tokens: int = WINDOW_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[Chunk]:
    """把解析结果切成有序 chunks（seq 从 1 起，跨整篇文档连续）。"""
    # 1) 过滤纯空白 section
    pieces: list[_Piece] = []
    for section in parsed.sections:
        if not section.text.strip():
            continue
        pieces.append(
            _Piece(
                base_path=_base_path(parsed.title, section),
                text=section.text,
                anchors=list(section.anchors),
            )
        )

    # 2) 过短合并：< min_tokens 向后吸收相邻 section
    merged: list[_Piece] = []
    idx = 0
    while idx < len(pieces):
        current = pieces[idx]
        idx += 1
        while estimate_tokens(current.text) < min_tokens and idx < len(pieces):
            current = _concat(current, pieces[idx])
            idx += 1
        merged.append(current)
    # 末尾仍过短 → 并回前一片（避免孤立的短尾片）
    if len(merged) >= 2 and estimate_tokens(merged[-1].text) < min_tokens:
        merged[-2] = _concat(merged[-2], merged[-1])
        merged.pop()

    # 3) 逐片切窗口，逐 chunk 编号
    chunks: list[Chunk] = []
    seq = 1
    for piece in merged:
        if estimate_tokens(piece.text) <= max_tokens:
            content = piece.text.strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    seq=seq,
                    header_path=_header_path(piece, _content_offset(piece.text, 0)),
                    content=content,
                    token_count=estimate_tokens(content),
                )
            )
            seq += 1
            continue
        for start, raw in _windows(piece):
            content = raw.strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    seq=seq,
                    header_path=_header_path(piece, _content_offset(raw, start)),
                    content=content,
                    token_count=estimate_tokens(content),
                )
            )
            seq += 1
    return chunks
