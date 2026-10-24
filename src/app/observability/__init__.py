"""可观测层（V2.4）：trace / token 估算 / 成本。

只做旁路记录，不改变 prompt、记忆写入与回复文本（不变量 2）。
"""

from app.observability.cost import cost_cny
from app.observability.tokens import estimate_messages_tokens, estimate_tokens
from app.observability.trace import (
    TraceRecorder,
    TraceStore,
    configure_trace_store,
    current_error,
    current_trace,
    is_degraded,
    mark_degraded,
    mark_error,
    reset_trace,
    start_trace,
)

__all__ = [
    "TraceRecorder",
    "TraceStore",
    "configure_trace_store",
    "cost_cny",
    "current_error",
    "current_trace",
    "estimate_messages_tokens",
    "estimate_tokens",
    "is_degraded",
    "mark_degraded",
    "mark_error",
    "reset_trace",
    "start_trace",
]
