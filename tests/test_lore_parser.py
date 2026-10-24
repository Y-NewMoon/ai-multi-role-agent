"""V2.1 解析与切分测试（T2）：`##` 切分、滑窗、断句优先、过短合并、确定性、token 同式。

不依赖数据库与真嵌入，CI 可跑。
"""

from app.agents.nodes.compose_prompt import _estimate_tokens
from app.lore.chunker import Chunk, chunk_document, estimate_tokens
from app.lore.parser import parse_markdown


def _long_paragraph(n_sentences: int = 30) -> str:
    """n 句等长纯 CJK 语料（每句 20 个汉字、无标点），用于滑窗测试。

    全为 CJK 汉字 ⇒ token 数 == 字数，窗口断言可直接按字数写。
    标点（如 `。`）按既有公式计入「非 CJK、4 字 1 token」，故此处刻意不含标点。
    """
    return "".join("句乙" + "甲" * 18 for _ in range(n_sentences))


def test_split_by_h2_and_h1_not_chunked() -> None:
    # 两段各 ≥50 token，避免触发过短合并，从而能看出 `##` 各自成 chunk
    history = "历史正文" + "甲" * 56 + "。"
    signboard = "招牌正文" + "乙" * 56 + "。"
    md = f"# 一盏春\n\n## 历史\n{history}\n## 招牌\n{signboard}\n"
    parsed = parse_markdown(md, doc_title="一盏春茶馆")
    assert [s.title for s in parsed.sections] == ["历史", "招牌"]

    chunks = chunk_document(parsed)
    assert len(chunks) == 2
    # H1 只作 doc 级标题，不落 chunk
    assert "一盏春" not in chunks[0].content
    assert chunks[0].content == history
    assert chunks[0].header_path == "一盏春茶馆 / 历史"
    assert chunks[1].content == signboard
    assert [c.seq for c in chunks] == [1, 2]


def test_header_path_untitled_section_is_doc_title() -> None:
    md = "开篇无标题的引言。\n\n## 历史\n正文。\n"
    parsed = parse_markdown(md, doc_title="一盏春茶馆")
    chunks = chunk_document(parsed)
    assert chunks[0].header_path == "一盏春茶馆"
    # 无标题引言与「历史」过短被合并（<50 token），合并后沿用首片起点路径
    assert len(chunks) == 1
    assert chunks[0].header_path == "一盏春茶馆"


def test_header_path_deep_heading() -> None:
    md = "# 茶馆\n\n## 历史\n\n### 祖父\n祖父在 1979 年盘下这间铺面。\n"
    parsed = parse_markdown(md, doc_title="茶馆")
    chunks = chunk_document(parsed)
    assert len(chunks) == 1
    assert chunks[0].header_path == "茶馆 / 历史 / 祖父"


def test_long_section_slides_window_with_overlap() -> None:
    body = _long_paragraph(30)  # 600 字 = 600 token
    md = f"## 长节\n{body}\n"
    parsed = parse_markdown(md, doc_title="茶馆")
    chunks = chunk_document(parsed)

    assert len(chunks) == 2
    # 400 token 窗口：首窗恰好 400 字（边界落在句末）
    assert len(chunks[0].content) == 400
    assert chunks[0].content == body[:400]
    # 次窗从 340 字处开始（overlap 60）
    assert chunks[1].content == body[340:]
    # 重叠区逐字相同：首窗末 60 字 == 次窗首 60 字
    assert chunks[0].content[-60:] == chunks[1].content[:60]


def test_window_boundary_prefers_sentence_end() -> None:
    # 每句 30 字（以「。」收尾）：400 token 处不是句末，应回退到最近句末
    body = "".join("句" + "乙" * 28 + "。" for _ in range(30))
    md = f"## 长节\n{body}\n"
    chunks = chunk_document(parse_markdown(md, doc_title="茶馆"))
    assert len(chunks) >= 2
    # 边界落在句末：长度是 30 的整数倍，且以「。」结束
    assert chunks[0].content.endswith("。")
    assert len(chunks[0].content) % 30 == 0


def test_short_sections_merge() -> None:
    md = "## 甲\n短一。\n## 乙\n短二。\n## 丙\n短三。\n"
    chunks = chunk_document(parse_markdown(md, doc_title="T"))
    assert len(chunks) == 1
    assert "短一" in chunks[0].content and "短三" in chunks[0].content


def test_blank_section_not_persisted() -> None:
    md = "# T\n\n## 空\n\n## 有内容\n这里有正文。\n"
    chunks = chunk_document(parse_markdown(md, doc_title="T"))
    assert len(chunks) == 1
    assert chunks[0].content == "这里有正文。"
    assert chunks[0].header_path == "T / 有内容"


def test_deterministic_two_runs() -> None:
    md = "# T\n\n## 甲\n" + _long_paragraph(40) + "\n## 乙\n短。\n"
    first: list[Chunk] = chunk_document(parse_markdown(md, doc_title="T"))
    second: list[Chunk] = chunk_document(parse_markdown(md, doc_title="T"))
    assert [
        (c.seq, c.header_path, c.content, c.token_count) for c in first
    ] == [(c.seq, c.header_path, c.content, c.token_count) for c in second]


def test_token_estimate_matches_compose_prompt() -> None:
    samples = [
        "",
        "abc",
        "中文",
        "一盏春茶馆",
        "祖父在 1979 年盘下这间铺面。",
        "a中b文c",
        "   ",
        "12345678",
        "123456789",
        "混合 mixed 文本 123 中文",
        "。！？\n",
        "X" * 100,
        "中" * 100,
        "half中文halfenglish12345",
        "茶馆 / 历史 / 祖父",
        "祖父在1979年盘下这间铺面，取名一盏春。",
        "emoji🙂与文字",
        "\t新行\n无空格",
        "句子01甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲。",
        "最终样本 final sample 42",
    ]
    assert len(samples) == 20
    for text in samples:
        assert estimate_tokens(text) == _estimate_tokens(text)
