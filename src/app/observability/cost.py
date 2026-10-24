"""成本核算（V2.4 · §2.4）。

公式逐字实现；单价默认值只是示例，部署时必须以账单为准覆盖 `.env`。
"""

from __future__ import annotations


def cost_cny(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """按每百万 token 单价折算人民币成本，保留 6 位小数（确定性）。"""
    return round(
        prompt_tokens / 1_000_000 * input_price_per_1m
        + completion_tokens / 1_000_000 * output_price_per_1m,
        6,
    )
