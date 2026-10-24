"""Markdown 设定解析（契约 4.2）：Markdown 文本 → sections。

规则：
- 以 `^## ` 切 section；`# ` 一级标题只作 doc 级标题，不单独成 chunk；
- H3 及更深标题归入所在 section，作为 `header_path` 的更深层级
  （契约 4.1 的 `"茶馆 / 历史 / 祖父"` 与 4.7 的注入示例即该形态）；
- section 无标题（`##` 之前的正文，或 `##` 后无标题）时，`header_path = doc.title`。

纯空白 section 由 chunker 过滤，不落库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ATX 标题：`#{1,6}` + 空白 + 标题文本（契约要求 `## ` 带空格）
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")


@dataclass
class Section:
    """一个 `##` 级 section。

    - `title`：`##` 标题文本；空串表示「无标题 section」。
    - `text`：section 正文（不含 `##` 标题行；含更深的 `###` 标题行）。
    - `anchors`：`(相对 text 的字符偏移, 更深标题路径)` 升序列表；用于按窗口起点
      定位 `header_path`。无更深标题时为空列表。
    """

    title: str
    text: str
    anchors: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)


@dataclass
class ParsedDoc:
    """解析结果：doc 级标题 + 有序 sections。"""

    title: str
    sections: list[Section]


def parse_markdown(text: str, *, doc_title: str) -> ParsedDoc:
    """把 Markdown 文本解析为 sections。

    `doc_title` 由调用方给定（API 的 `title` 字段），不取自文档内的 `#` 行
    —— 契约 4.9 示例中 `title="一盏春茶馆"` 与正文 `# 一盏春` 并不相同。
    """
    sections: list[Section] = []
    title = ""
    lines: list[str] = []
    anchors: list[tuple[int, tuple[str, ...]]] = []
    stack: list[str] = []
    offset = 0
    started = False

    def _append(line: str) -> None:
        nonlocal offset
        lines.append(line)
        offset += len(line)

    def _flush() -> None:
        nonlocal title, lines, anchors, stack, offset, started
        if started:
            body = "".join(lines)
            # 纯空白 section 不产出（契约 4.2；chunker 处再兜底过滤）
            if body.strip():
                sections.append(Section(title=title, text=body, anchors=list(anchors)))
        title, lines, anchors, stack, offset, started = "", [], [], [], 0, False

    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        match = _HEADING.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1:
                # doc 级标题：只作标题，不落 chunk
                _flush()
                continue
            if level == 2:
                _flush()
                started = True
                title = heading
                continue
            # H3+：归入当前 section，只更新更深标题路径（标题行不进正文）
            if not started:
                started = True
                title = ""
            depth = level - 2  # H3 → 1
            stack = stack[: depth - 1]
            stack.append(heading)
            # 锚点落在其后正文的起点（当前正文偏移）
            anchors.append((offset, tuple(stack)))
            continue

        if not started:
            started = True
            title = ""
        _append(raw)

    _flush()
    return ParsedDoc(title=doc_title, sections=sections)
