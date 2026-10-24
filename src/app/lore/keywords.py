"""关键词抽取与中文分词（契约 4.4）：jieba posseg 的标题词 / 专名 / 引号内容。

- 写入与查询两侧都必须走 jieba：`to_tsvector('simple', ' '.join(jieba_cut(content)))`；
  直接对中文原文调 `to_tsvector('simple', ...)` 会整句变成一个词元，关键词路等于全废（§10.1）。
- 关键词上限 8 个；单字词丢弃（与检索侧「长度 ≥ 2 的词」一致）。
"""

from __future__ import annotations

import re

import jieba
import jieba.posseg as pseg

# 引号内容：「」 『』 “” 与直引号
_QUOTED = re.compile(r"[「『“\"]([^」』”\"]{1,20})[」』”\"]")
# 标题词可用的实词词性（名词类 + 动词 / 英文 / 简称 / 习语）
_CONTENT_POS = frozenset(
    {"n", "nr", "ns", "nt", "nz", "nrt", "vn", "an", "eng", "j", "l", "i", "v"}
)
# 正文专名（契约 4.4 的「专名」）
_PROPER_POS = frozenset({"nr", "ns", "nt", "nz", "nrt"})
# header_path 的分隔串（与 chunker 一致）
HEADER_SEP = " / "

MIN_LEN = 2
MAX_KEYWORDS = 8


def cut(text: str) -> list[str]:
    """jieba 精确分词，丢弃纯空白词。"""
    return [w for w in jieba.cut(text) if w.strip()]


def cut_for_tsv(text: str) -> str:
    """tsv 入库/查询用的分词串（空格连接，契约 4.4）。"""
    return " ".join(cut(text))


def extract_keywords(content: str, header_path: str = "") -> list[str]:
    """按「标题词 → 专名 → 引号内容」抽取，去重后截断到 8 个（契约 §5 T3）。"""
    out: list[str] = []

    def add(token: str) -> None:
        token = token.strip()
        if len(token) >= MIN_LEN and token not in out:
            out.append(token)

    # 1) 标题词（header_path 各段的实词）
    for segment in header_path.split(HEADER_SEP):
        for word, flag in pseg.cut(segment):
            if flag in _CONTENT_POS:
                add(word)

    # 2) 正文专名
    for word, flag in pseg.cut(content):
        if flag in _PROPER_POS:
            add(word)

    # 3) 引号内容
    for match in _QUOTED.finditer(content):
        add(match.group(1))

    return out[:MAX_KEYWORDS]
