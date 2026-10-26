"""每轮 trace：contextvar 隔离 + 按天 JSONL（V2.4 · §2.1/§2.2）。

设计要点：
- **加性**：trace 只旁路记录，不改变 prompt / 记忆写入 / 回复文本；写失败 fail-open。
- **不落全文**：只落 id / 计数 / 耗时 / 错误类型；错误 message 截断（默认 200 字）。
- **每 agent 轮一条**：私聊/闲聊一条；群聊每个发言人一条，共享 `parent_trace_id`。
- 文件：`{TRACE_DIR}/{YYYY-MM-DD}.jsonl`，UTF-8，append。
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TRACE_VERSION = "1.0"
PROMPT_VERSION_DEFAULT = "v2.6"

# 项目固定单一时区；trace 时间戳与按天文件名都以它为准，不读宿主时区。
TZ = ZoneInfo("Asia/Shanghai")


# ===== 全局 trace store（main 启动时注入；未注入时按 settings 惰性构造）=====

_store: "TraceStore | None" = None
_store_configured = False


def configure_trace_store(store: "TraceStore | None") -> None:
    """注入全局 TraceStore（main 启动 / 单测夹具）；传 None 表示关闭写入。"""
    global _store, _store_configured
    _store = store
    _store_configured = True


def _get_store() -> "TraceStore | None":
    global _store
    if not _store_configured and _store is None:
        try:  # 惰性从 settings 构造；配置缺失时不写 trace（no-op），不影响主流程
            from app.config import get_settings

            settings = get_settings()
            _store = TraceStore(
                directory=settings.trace_dir,
                enabled=settings.trace_enabled,
                sample_rate=settings.trace_sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 - 可观测不得影响启动
            logger.warning("trace store 初始化失败，已降级为 no-op：%s", exc)
            return None
    return _store


# ===== 降级标记（与 trace 同一生命周期）=====

_degraded: ContextVar[bool] = ContextVar("llm_degraded", default=False)
_error: ContextVar[dict | None] = ContextVar("trace_error", default=None)


def mark_degraded() -> None:
    """InstrumentedLLM 走超时兜底时置位；`update_memory` 据此跳过所有写入。"""
    _degraded.set(True)


def is_degraded() -> bool:
    return bool(_degraded.get())


def mark_error(error_type: str, message: str = "") -> None:
    """登记本轮错误类型/摘要（message 在 finish 时按配置截断）。"""
    _error.set({"type": str(error_type), "message": str(message)})


def current_error() -> dict | None:
    return _error.get()


def _now() -> datetime:
    """墙钟当前时间，固定 `Asia/Shanghai`（+08:00）。

    不能依赖宿主时区：CI（UTC）与部署机（+08:00）必须写出同一种偏移；
    也不读 `EVAL_FIXED_NOW` 之类的业务冻结时钟，否则
    `total_ms` / `finished_at` 不再反映真实耗时。
    """
    return datetime.now(TZ)


_git_sha_cache: str | None = None


def _git_sha() -> str:
    """启动期缓存 git short sha；失败写 `"unknown"`，不得 import eval。"""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = result.stdout.strip()
        _git_sha_cache = sha or "unknown"
    except Exception:  # noqa: BLE001 - 不是 git 仓库/无 git 时不得阻塞
        _git_sha_cache = "unknown"
    return _git_sha_cache


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


@dataclass
class TraceRecorder:
    """一轮一 recorder；no-op 实例的所有方法均为空操作。"""

    trace_id: str
    parent_trace_id: str | None = None
    mode: str = "private"
    session_id: int | None = None
    group_session_id: int | None = None
    agent_id: int = 0
    turn: int | None = None
    speaker_index: int | None = None
    prompt_version: str = PROMPT_VERSION_DEFAULT
    git_sha: str = "unknown"
    started_at: datetime = field(default_factory=_now)
    nodes: list[dict] = field(default_factory=list)
    retrieval_stats: dict = field(
        default_factory=lambda: {
            "lore_ids": [],
            "memory_count": 0,
            "rerank_top": None,
            "rewrite": {"used": False, "ms": 0},
        }
    )
    tools: list[dict] = field(default_factory=list)
    llm_stats: dict = field(
        default_factory=lambda: {
            "model": "",
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_cny": 0.0,
            "usage_source": "estimator",
            "wait_ms": 0,
        }
    )
    budget: dict | None = None
    memory_stats: dict | None = None
    first_token_ms: int | None = None
    # —— 内部状态 ——
    max_field_chars: int = 200
    _store: "TraceStore | None" = field(default=None, repr=False, compare=False)
    _noop: bool = field(default=False, repr=False, compare=False)
    _finished: bool = field(default=False, repr=False, compare=False)
    # R13：`turn_done` 前冻结计时/错误快照；之后只允许追加 usage，不延长 total_ms
    _stopped_at: datetime | None = field(default=None, repr=False, compare=False)
    _stop_error: dict | None = field(default=None, repr=False, compare=False)
    _stop_degraded: bool = field(default=False, repr=False, compare=False)

    # ===== 记录方法（签名逐字见 §2.2）=====

    def node(self, name: str, ms: int, ok: bool = True) -> None:
        if self._noop:
            return
        self.nodes.append({"name": str(name), "ms": int(ms), "ok": bool(ok)})

    def tool(self, name: str, ok: bool, ms: int, error_type: str | None = None) -> None:
        if self._noop:
            return
        self.tools.append(
            {
                "name": str(name),
                "ok": bool(ok),
                "ms": int(ms),
                "error_type": error_type,
            }
        )

    def retrieval(
        self,
        *,
        lore_ids: list[str],
        memory_count: int,
        rerank_top: int | None,
        rewrite_used: bool,
        rewrite_ms: int,
    ) -> None:
        if self._noop:
            return
        self.retrieval_stats = {
            "lore_ids": [str(i) for i in lore_ids],
            "memory_count": int(memory_count),
            "rerank_top": None if rerank_top is None else int(rerank_top),
            "rewrite": {"used": bool(rewrite_used), "ms": int(rewrite_ms)},
        }

    def llm(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_cny: float,
        usage_source: str = "estimator",
    ) -> None:
        if self._noop:
            return
        stats = self.llm_stats
        stats["model"] = str(model)
        stats["calls"] = int(stats["calls"]) + 1
        stats["prompt_tokens"] = int(stats["prompt_tokens"]) + int(prompt_tokens)
        stats["completion_tokens"] = int(stats["completion_tokens"]) + int(completion_tokens)
        stats["cost_cny"] = float(stats["cost_cny"]) + float(cost_cny)
        stats["usage_source"] = usage_source

    def llm_wait(self, wait_ms: int) -> None:
        """记录等全局信号量的累计毫秒（不影响回复，§2.1-13）。"""
        if self._noop:
            return
        self.llm_stats["wait_ms"] = int(self.llm_stats["wait_ms"]) + int(wait_ms)

    def memory_count(self, count: int) -> None:
        """只更新召回记忆条数（retrieve_memory 调用；lore 由 retrieve_lore 补全）。"""
        if self._noop:
            return
        self.retrieval_stats["memory_count"] = int(count)

    def memory_snapshot(
        self, *, namespace_entries: int, facts: int, profile_slots: int
    ) -> None:
        """C-03 观察：`TRACE_MEMORY_STATS=true` 时记录 namespace 规模（§2.1-14）。"""
        if self._noop:
            return
        self.memory_stats = {
            "namespace_entries": int(namespace_entries),
            "facts": int(facts),
            "profile_slots": int(profile_slots),
        }

    def set_budget(self, *, prompt_token_budget: int, estimated_prompt_tokens: int) -> None:
        """记录预算估算结果；只记录，不二次截断（§2.9 预算）。"""
        if self._noop:
            return
        self.budget = {
            "prompt_token_budget": int(prompt_token_budget),
            "estimated_prompt_tokens": int(estimated_prompt_tokens),
            "over_budget": int(estimated_prompt_tokens) > int(prompt_token_budget),
        }

    def first_token(self) -> None:
        if self._noop or self.first_token_ms is not None:
            return
        self.first_token_ms = self._elapsed_ms()

    def stop(self, *, error: dict | None = None, degraded: bool = False) -> None:
        """冻结本轮计时与错误/降级快照（在 `turn_done` 前调用）。

        R13：收敛动作必须发生在 `turn_done` 之前，客户端断开也要留下该条 trace；
        后续归属的调用（如终止决策）只追加 usage，`total_ms/finished_at` 不再变化。
        """
        if self._stopped_at is not None:
            return
        self._stopped_at = _now()
        self._stop_error = error if error is not None else current_error()
        self._stop_degraded = bool(degraded or is_degraded())

    def finish(self, *, error: dict | None = None, degraded: bool = False) -> None:
        """收敛并写一条 JSONL；重复调用只写一次；失败只 warning。"""
        if self._finished:
            return
        self._finished = True
        if self._noop or self._store is None:
            return
        if self._stopped_at is not None:
            finished_at = self._stopped_at
            if error is None:
                error = self._stop_error
            if not degraded:
                degraded = self._stop_degraded
        else:
            finished_at = _now()
        record = self._record(error=error, degraded=degraded, finished_at=finished_at)
        self._store.write(record)

    def _elapsed_ms(self) -> int:
        return int((_now() - self.started_at).total_seconds() * 1000)

    def _record(
        self, *, error: dict | None, degraded: bool, finished_at: datetime
    ) -> dict[str, Any]:
        llm = dict(self.llm_stats)
        llm["cost_cny"] = round(float(llm["cost_cny"]), 6)

        err: dict | None = None
        if error is not None:
            message = str(error.get("message") or "")
            err = {
                "type": str(error.get("type") or "error"),
                "message": message[: max(0, int(self.max_field_chars))],
            }

        total_ms = int((finished_at - self.started_at).total_seconds() * 1000)
        return {
            "trace_version": TRACE_VERSION,
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "mode": self.mode,
            "session_id": self.session_id,
            "group_session_id": self.group_session_id,
            "agent_id": self.agent_id,
            "turn": self.turn,
            "speaker_index": self.speaker_index,
            "prompt_version": self.prompt_version,
            "git_sha": self.git_sha,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(finished_at),
            "first_token_ms": self.first_token_ms,
            "total_ms": total_ms,
            "nodes": list(self.nodes),
            "retrieval": dict(self.retrieval_stats),
            "tools": list(self.tools),
            "llm": llm,
            "budget": self.budget,
            "memory_stats": self.memory_stats,
            "error": err,
            "degraded": bool(degraded),
        }


class TraceStore:
    """按天 append JSONL；进程内锁串行；写失败只 warning（fail-open）。"""

    def __init__(self, *, directory: str, enabled: bool = True, sample_rate: float = 1.0):
        self._directory = Path(directory)
        self._enabled = bool(enabled)
        self._sample_rate = float(sample_rate)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def should_sample(self) -> bool:
        if not self._enabled:
            return False
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        return random.random() < self._sample_rate

    def write(self, record: dict) -> None:
        """append 一行；任何异常都不得冒泡到请求链路。"""
        if not self._enabled:
            return
        try:
            started_at = str(record.get("started_at") or "")
            day = started_at[:10] if len(started_at) >= 10 else _now().date().isoformat()
            self._directory.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                with open(self._directory / f"{day}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - fail-open
            logger.warning("trace 写入失败（已忽略）：%s", exc)


# ===== contextvar 生命周期 =====

_current: ContextVar[TraceRecorder | None] = ContextVar("current_trace", default=None)


def start_trace(
    *,
    mode: str,
    session_id: int | None,
    agent_id: int,
    turn: int | None,
    group_session_id: int | None = None,
    speaker_index: int | None = None,
    parent_trace_id: str | None = None,
) -> tuple[TraceRecorder, Token]:
    """开启一轮 trace 并绑定到 contextvar；未采样/关闭时返回 no-op recorder。"""
    import uuid

    store = _get_store()
    noop = store is None or not store.should_sample()
    try:
        prompt_version = _prompt_version()
    except Exception:  # noqa: BLE001
        prompt_version = PROMPT_VERSION_DEFAULT

    recorder = TraceRecorder(
        trace_id=uuid.uuid4().hex,
        parent_trace_id=parent_trace_id,
        mode=mode,
        session_id=session_id,
        group_session_id=group_session_id,
        agent_id=int(agent_id),
        turn=turn,
        speaker_index=speaker_index,
        prompt_version=prompt_version,
        git_sha=_git_sha(),
        max_field_chars=_max_field_chars(store),
        _store=store,
        _noop=noop,
    )
    token = _current.set(recorder)
    # 降级标记与 trace 同生命周期：新一轮开始必须清零
    _degraded.set(False)
    _error.set(None)
    return recorder, token


def current_trace() -> TraceRecorder | None:
    return _current.get()


def reset_trace(token: Token) -> None:
    """必须放在 finally：不得把 trace / 降级标记泄漏到下一个请求。"""
    try:
        _current.reset(token)
    except ValueError:  # 跨 context reset 时兜底置空，避免泄漏
        _current.set(None)
    _degraded.set(False)
    _error.set(None)


def _prompt_version() -> str:
    from app.config import get_settings

    return get_settings().prompt_version


def _max_field_chars(store: TraceStore | None) -> int:
    try:
        from app.config import get_settings

        return int(get_settings().trace_max_field_chars)
    except Exception:  # noqa: BLE001
        return 200
