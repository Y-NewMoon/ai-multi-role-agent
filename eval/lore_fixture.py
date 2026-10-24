"""检索 fixture：语料加载 + 固定切分 + 余弦检索（确定性，契约 4.1.4 / 交付物 6）。

- 语料：`eval/fixtures/lore_corpus/<corpus>.md`，3 篇设定文档（茶馆 / 校园 / 江湖）；
- 切分：按 `##` 二级标题顺序编号，chunk id 规则 `{corpus_stem}:{section_index:02d}`，
  `section_index` 从 1 开始；整篇无 `##` 则为 `:01`。切分规则固定，不引第三方切分器；
- 检索：字符 1/2-gram TF 余弦相似度（标题词加权 ×2）。纯函数、无模型、无 IO 副作用，
  跨机器结果逐位一致。V2.1 的真实向量/混合检索以本基线为对照。

注意：不用真实数据库 id（V2.0 阶段没有 lore 表），也不依赖 sentence-transformers，
保证 `--ci` 无 embeddings 依赖组时检索指标照常产出。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "lore_corpus"

_KEEP_RE = re.compile(r"[^0-9a-z一-鿿]+")


def normalize(text: str) -> str:
    """小写化并只保留中文/字母/数字，其余字符折叠为空格。"""
    return _KEEP_RE.sub("", text.lower())


def _terms(text: str) -> list[str]:
    """字符 1-gram + 2-gram（中文按字，英文数字串按字符）。"""
    seq = normalize(text)
    unigrams = list(seq)
    bigrams = [seq[i : i + 2] for i in range(len(seq) - 1)]
    return unigrams + bigrams


def vectorize(text: str, title: str = "") -> dict[str, float]:
    """词频向量（标题加权 ×2），返回 L2 归一化后的稀疏向量。"""
    counts: dict[str, int] = {}
    for term in _terms(text):
        counts[term] = counts.get(term, 0) + 1
    for term in _terms(title):
        counts[term] = counts.get(term, 0) + 2
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm == 0:
        return {}
    return {t: v / norm for t, v in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def chunk_document(text: str, corpus_stem: str) -> list[dict]:
    """固定切分：按 `##` 二级标题切；无 `##` 则整篇一个 chunk。

    返回 [{"chunk_id", "index", "title", "content"}]，index 从 1 开始。
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title, current_body = "", []
    has_h2 = any(line.startswith("## ") for line in lines)
    if not has_h2:
        return [
            {
                "chunk_id": f"{corpus_stem}:01",
                "index": 1,
                "title": "",
                "content": text.strip(),
            }
        ]
    for line in lines:
        if line.startswith("## "):
            if current_title or current_body:
                sections.append((current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_title, current_body))
    chunks: list[dict] = []
    for i, (title, body) in enumerate(sections, start=1):
        if not title and not "".join(body).strip():
            continue
        chunks.append(
            {
                "chunk_id": f"{corpus_stem}:{i:02d}",
                "index": i,
                "title": title,
                "content": "\n".join(body).strip(),
            }
        )
    return chunks


def load_corpus(corpus_stem: str) -> list[dict]:
    """加载 `eval/fixtures/lore_corpus/<corpus_stem>.md` 并按固定规则切分。"""
    path = FIXTURE_DIR / f"{corpus_stem}.md"
    if not path.exists():
        raise FileNotFoundError(f"fixture 语料不存在：{path}")
    return chunk_document(path.read_text(encoding="utf-8"), corpus_stem)


def available_corpora() -> list[str]:
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.md"))


def rank_chunks(query: str, chunks: list[dict], top_k: int = 10) -> list[str]:
    """确定性余弦检索：分数降序，同分按 chunk_id 升序，返回 chunk_id 列表。"""
    q_vec = vectorize(query)
    scored = []
    for chunk in chunks:
        c_vec = vectorize(chunk["content"], title=chunk.get("title", ""))
        scored.append((cosine(q_vec, c_vec), chunk["chunk_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cid for _, cid in scored[:top_k]]
