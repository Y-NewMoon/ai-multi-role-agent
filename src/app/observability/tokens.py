"""token 估算（D3 · `cjk1_other4`）。

与 V2.3 §5#3 同口径：CJK 1 字 = 1 token；其余 4 字 = 1 token（向上取整）。
只用于成本与预算观测，不代表 provider 真实 usage；`usage_source` 固定 `"estimator"`。
"""

from __future__ import annotations

import math

# CJK 统一表意文字 + 扩展 A + 兼容表意文字 + 中日韩标点/假名/谚文
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # 日文假名
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 基本区
    (0xAC00, 0xD7AF),  # 谚文音节
    (0xF900, 0xFAFF),  # CJK 兼容表意文字
    (0xFF00, 0xFFEF),  # 全角标点/字母
    (0x20000, 0x2FA1F),  # CJK 扩展 B~F
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """CJK 1 字 = 1 token；其余 4 字 = 1 token（与 V2.3 §5#3 同口径）。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return cjk + math.ceil(other / 4)


def _content_text(content: object) -> str:
    """兼容 OpenAI 风格 content（str 或 [{type:text,text:...}] 片段）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算一组消息的 prompt token（content 文本 + 每条 4 token 结构开销）。"""
    total = 0
    for message in messages:
        total += estimate_tokens(_content_text(message.get("content")))
        total += 4
    return total
