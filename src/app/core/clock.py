"""统一时间入口（V3 M1 · ADR-0012）。

- `TZ` 固定 `Asia/Shanghai`（M1 不做多时区）；
- `now()`：`EVAL_FIXED_NOW`（进程环境变量）→ `settings.eval_fixed_now`（`.env` 可选）
  → 真实时间；业务代码不得再出现 `datetime.now(...)`；
- `now_iso()`：ISO 8601、秒级、带 `+08:00`；
- `parse_fixed_now()`：解析并校验冻结时刻；非法值抛 `ValueError`（启动即失败）。
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
ENV_KEY = "EVAL_FIXED_NOW"


def parse_fixed_now(raw: str | None) -> datetime | None:
    """把 `EVAL_FIXED_NOW` 解析为带时区的时间；空/缺省返回 None，非法抛 ValueError。

    接受 ISO 8601：`2026-10-25T10:35:00+08:00`、`2026-10-25T02:35:00Z`；
    无偏移按 `Asia/Shanghai` 解释。
    """
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"EVAL_FIXED_NOW 不是合法 ISO 8601: {text}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TZ)
    return moment.astimezone(TZ)


def fixed_now_raw() -> str | None:
    """当前生效的冻结时刻原文（环境变量优先，其次 settings）；未冻结为 None。"""
    raw = os.environ.get(ENV_KEY)
    if raw is None or not str(raw).strip():
        raw = _settings_fixed_now()
    return str(raw).strip() if raw is not None and str(raw).strip() else None


def now() -> datetime:
    """当前时间：冻结值优先，否则真实时间（唯一的“现在”入口）。"""
    parsed = parse_fixed_now(fixed_now_raw())
    if parsed is not None:
        return parsed
    return datetime.now(TZ)


def now_iso() -> str:
    return now().isoformat(timespec="seconds")


def validate_fixed_now() -> None:
    """启动时调用一次：非法 `EVAL_FIXED_NOW` 直接失败，不带到请求里。"""
    parse_fixed_now(fixed_now_raw())


def _settings_fixed_now() -> str | None:
    try:
        from app.config import get_settings

        return get_settings().eval_fixed_now
    except Exception:  # noqa: BLE001 - 配置缺失时视为未冻结
        return None
