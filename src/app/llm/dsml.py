"""DeepSeek DSML 文本工具调用兜底解析。

背景：个别轮次模型会把工具调用渲染成 DSML 标记并以 `content` 文本返回
（`finish_reason=stop`、`tool_calls` 为空），形如：

```
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="set_reminder">
<｜｜DSML｜｜ parameter name="when" string="true">2026-10-24T19:42:00+08:00</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>
```

若当成正文直接落库，会把乱码显示给用户，并且污染后续轮次的上下文（模型会继续
模仿）。这里把文本还原成 `ToolCall` 列表；解析不出调用时把文本清空，交给上层
「空回复兜底」重试，保证原文既不落库也不展示。

标记里的竖线可能是半角 `|` 或全角 `｜`（U+FF5C），解析前统一归一化。
"""

from __future__ import annotations

import json
import re
import uuid

from app.skills.base import ToolCall

# `<|DSML|` / `<｜｜DSML｜｜` → `<DSML `；闭合标签保留 `/`
_MARKER_RE = re.compile(r"<(/)?[|｜]+DSML[|｜]+\s*")
_CALLS_BLOCK_RE = re.compile(r"<DSML calls>(.*?)</DSML calls>", re.S)
_INVOKE_RE = re.compile(r'<DSML invoke name="([^"]+)">(.*?)</DSML invoke>', re.S)
_PARAM_RE = re.compile(r'<DSML parameter name="([^"]+)"[^>]*>(.*?)</DSML parameter>', re.S)
# 残留的孤立标记（例如只有 `<DSML invoke>` 没有闭合），用于兜底清洗
_STRAY_TAG_RE = re.compile(r"</?DSML[^>]*>")


def normalize_dsml(text: str) -> str:
    """把 DSML 标记归一化成 `<DSML ...>` 形式，便于统一解析。"""
    return _MARKER_RE.sub(lambda m: f"<{m.group(1) or ''}DSML ", text)


def has_dsml_marker(text: str) -> bool:
    """是否出现 DSML 工具调用标记（calls / invoke）。"""
    if not text:
        return False
    normalized = normalize_dsml(text)
    return "<DSML calls>" in normalized or "<DSML invoke" in normalized


def _parse_value(raw: str) -> object:
    """参数值按 JSON 还原（true/false/数字/字符串）；失败则保留原文本。"""
    text = raw.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_dsml_content(text: str) -> tuple[list[ToolCall], str, bool]:
    """解析 `content` 里的 DSML 工具调用。

    返回 `(calls, cleaned_content, had_marker)`：
    - 无标记：`([], 原文, False)`；
    - 有标记且解析出调用：`(calls, 去掉标记后的正文, True)`；
    - 有标记但解析不出调用：`([], "", True)` —— 上层应丢弃并走空回复兜底。
    """
    normalized = normalize_dsml(text)
    if "<DSML calls>" not in normalized and "<DSML invoke" not in normalized:
        return [], text, False

    calls: list[ToolCall] = []
    for block in _CALLS_BLOCK_RE.findall(normalized):
        for name, body in _INVOKE_RE.findall(block):
            arguments: dict[str, object] = {}
            for key, raw in _PARAM_RE.findall(body):
                arguments[key] = _parse_value(raw)
            calls.append(
                ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments)
            )

    cleaned = _CALLS_BLOCK_RE.sub("", normalized)
    cleaned = _STRAY_TAG_RE.sub("", cleaned).strip()
    # 有 invoke 但没被 calls 包裹时，上面的循环会漏掉：再扫一遍孤立 invoke
    if not calls:
        for name, body in _INVOKE_RE.findall(normalized):
            arguments = {
                key: _parse_value(raw) for key, raw in _PARAM_RE.findall(body)
            }
            calls.append(ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments))

    if not calls:
        return [], "", True
    return calls, cleaned, True
