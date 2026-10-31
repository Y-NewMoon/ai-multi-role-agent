"""V2.3 事实抽取（契约 §2.3）。

每轮回复后，从「本轮用户消息 + 本轮角色回复」一次性抽出 facts + profile + importance。
D3：一次调用同时产出三样，避免每轮 3 次 LLM；D8：任何失败 fail-open（返回 None）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from app.llm.base import LLMProvider
from app.prompts.memory_extract import EXTRACT_PROMPT, EXTRACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 2000
MAX_FACTS = 5
MAX_PREDICATE_CHARS = 20
MAX_OBJECT_CHARS = 40
MAX_PROFILE_VALUE_CHARS = 40
MAX_ATTEMPTS = 2  # 首次 + 重试 1 次

PROFILE_WHITELIST = ("称呼", "禁忌", "关系阶段", "重要日期")
SUBJECT_WHITELIST = ("用户", "角色")

_PLACEHOLDER_OBJECTS = {
    "",
    "不知道",
    "没说",
    "没提",
    "无",
    "没有",
    "不详",
    "未知",
    "不确定",
    "未提及",
    "none",
    "n/a",
    "null",
}
_EDGE_PUNCT = "，。！？；：、,.!?;: \t\r\n　\"'“”‘’（）()【】[]《》<>「」『』"

# 超长画像值的收尾：优先断在标点，避免把引号/括号切成半个
_PROFILE_CUT_MARKS = "，。；、！？,;!?"
_PROFILE_PAIRS = {"“": "”", "「": "」", "『": "』", "（": "）", "(": ")", "《": "》"}
_PROFILE_CLEAR_WORDS = {"", "无", "没有", "暂无", "不适用", "none", "null", "n/a"}


def _balance_pairs(text: str) -> str:
    """补齐落单的左引号/左括号（只补右半边，不改内容）。"""
    for left, right in _PROFILE_PAIRS.items():
        if text.count(left) > text.count(right):
            text += right
    return text


def _truncate_profile(value: str, limit: int = MAX_PROFILE_VALUE_CHARS) -> str:
    """画像值超长时：能在标点处断就断在标点，否则截断补省略号。

    与旧实现（`[:40]` 硬切）的区别：不会出现 `不要称呼角色为“宝宝` 这种半个引号。
    """
    text = str(value or "").strip()
    if len(text) <= limit:
        return _balance_pairs(text)

    head = text[:limit]
    cut = max((head.rfind(ch) for ch in _PROFILE_CUT_MARKS), default=-1)
    if cut >= max(6, limit // 4):
        result = head[: cut + 1]
    else:
        result = head[: limit - 1].rstrip() + "…"

    result = result.rstrip(_EDGE_PUNCT)
    result = _balance_pairs(result)
    if len(result) > limit:
        result = head[: limit - 1].rstrip() + "…"
    return result


@dataclass(frozen=True)
class ExtractedFact:
    subject: str  # 只允许 "用户" / "角色"
    predicate: str  # ≤ 20 字
    object: str  # ≤ 40 字
    importance: int  # 1~5


@dataclass(frozen=True)
class Extraction:
    facts: list[ExtractedFact] = field(default_factory=list)
    profile: dict[str, str] = field(default_factory=dict)


def _clean(text: object) -> str:
    """去首尾空白与标点。"""
    return str(text or "").strip().strip(_EDGE_PUNCT).strip()


def _extract_json_block(text: str) -> str | None:
    """从模型输出里抠出单个 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return None


def _parse_facts(raw_facts: object) -> list[ExtractedFact]:
    if not isinstance(raw_facts, list):
        return []
    out: list[ExtractedFact] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        subject = _clean(item.get("subject"))
        if subject not in SUBJECT_WHITELIST:
            continue
        predicate = _clean(item.get("predicate"))[:MAX_PREDICATE_CHARS]
        obj = _clean(item.get("object"))[:MAX_OBJECT_CHARS]
        if not predicate or not obj or obj.lower() in _PLACEHOLDER_OBJECTS:
            continue
        try:
            importance = int(item.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3
        importance = min(5, max(1, importance))
        out.append(
            ExtractedFact(subject=subject, predicate=predicate, object=obj, importance=importance)
        )
    out.sort(key=lambda f: -f.importance)
    return out[:MAX_FACTS]


def _parse_profile(raw_profile: object) -> dict[str, str]:
    if not isinstance(raw_profile, dict):
        return {}
    out: dict[str, str] = {}
    for slot in PROFILE_WHITELIST:
        if slot not in raw_profile:
            continue
        value = _truncate_profile(_clean(raw_profile[slot]))
        # 空串 / 占位词 = 明确撤销该槽位（update_profile 写空值会删掉旧档案）
        out[slot] = "" if value.strip().lower() in _PROFILE_CLEAR_WORDS else value
    return out


def parse_extraction(text: str) -> Extraction | None:
    """解析抽取输出；非法 JSON / 结构错误返回 None（供重试）。"""
    block = _extract_json_block(text)
    if block is None:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("facts", []), list) or not isinstance(data.get("profile", {}), dict):
        return None
    return Extraction(
        facts=_parse_facts(data.get("facts")),
        profile=_parse_profile(data.get("profile")),
    )


async def extract(
    llm: LLMProvider,
    user_text: str,
    reply_text: str,
    *,
    timeout_s: float = 8.0,
    current_profile: dict[str, str] | None = None,
) -> Extraction | None:
    """抽不出/失败返回 None（fail-open）；成功但无事实返回 Extraction([], {})。

    `current_profile`：把旧档案带进提示词，模型才能在"用户推翻了旧设定"时输出空串撤销它。
    """
    conversation = f"用户：{user_text}\n角色：{reply_text}"
    if len(conversation) > MAX_INPUT_CHARS:
        conversation = conversation[:MAX_INPUT_CHARS]
    profile_text = (
        json.dumps(current_profile, ensure_ascii=False)
        if current_profile
        else "（无）"
    )
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACT_PROMPT.format(
                conversation=conversation, current_profile=profile_text
            ),
        },
    ]
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = await asyncio.wait_for(llm.generate(messages), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 - 抽取失败不得阻断回复（D8）
            logger.warning("事实抽取 LLM 调用失败（第 %d 次）：%s", attempt + 1, exc)
            continue
        parsed = parse_extraction(raw or "")
        if parsed is not None:
            return parsed
        logger.warning("事实抽取输出不是合法 JSON（第 %d 次），将重试", attempt + 1)
    return None
