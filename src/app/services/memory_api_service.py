"""V3 M2：本机记忆接口的服务层。

四个动作直接调节点函数（不走 LangGraph、不碰 checkpointer/persona/lore/tools）：

- `turn`    ：滚动缓冲 + `update_memory`（语义记忆/抽取/摘要/L2）；
- `context` ：`retrieve_memory`（profile/facts/L2/summary/memories）；
- `forget`  ：按内容删语义记忆（不动 facts/profile/L2/summary）；
- `state`   ：条目/事实/档案/缓冲规模。

错误一律抛 `MemoryApiError`，由 `error_handlers` 映射成 §2.8 的错误信封。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.nodes.retrieve_memory import retrieve_memory
from app.agents.nodes.update_memory import update_memory
from app.config import Settings
from app.core import clock
from app.core.clock import TZ
from app.core.errors import AppError
from app.llm.base import LLMProvider
from app.memory.store import MemoryStore
from app.services.memory_api_buffer import (
    BufferOverflowError,
    BufferPersistError,
    MemoryApiBuffer,
    build_turn_messages,
)

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{1,64}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")
_MAX_QUERY_CHARS = 2000
_MAX_CONTENTS = 100
_MAX_LIMIT = 20


class MemoryApiError(AppError):
    """记忆接口业务异常：带错误码与 HTTP 状态，交由 memory 路径的处理器映射。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)


def _invalid(message: str) -> MemoryApiError:
    return MemoryApiError(422, "invalid_request", message)


def _parse_at(raw: str | None) -> str:
    """`at` 校验 + 归一化：无偏移按 Asia/Shanghai，输出带偏移的秒级 ISO 8601。"""
    if raw is None or not str(raw).strip():
        return clock.now_iso()
    text = str(raw).strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _invalid("at 不是合法的 ISO 8601 时间") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TZ)
    return moment.astimezone(TZ).isoformat(timespec="seconds")


def _fact_view(fact: dict) -> dict:
    """只暴露契约 §2.3 允许的字段，不泄漏内部 key。"""
    return {
        "text": fact.get("text", ""),
        "subject": fact.get("subject", ""),
        "predicate": fact.get("predicate", ""),
        "object": fact.get("object", ""),
        "importance": float(fact.get("importance", 0.5)),
        "updated_at": fact.get("updated_at"),
    }


class MemoryApiService:
    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        llm: LLMProvider,
        buffer: MemoryApiBuffer,
        settings: Settings,
    ) -> None:
        self._store = memory_store
        self._llm = llm
        self._buffer = buffer
        self._settings = settings
        # 幂等：进程内 TTL 缓存（不持久化，重启即失效；D9）
        self._idem: dict[tuple[str, str, str], tuple[float, dict]] = {}
        self._idem_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def _check_agent(self, agent: str) -> str:
        name = str(agent or "").strip()
        if not _ID_RE.match(name) or name.isdigit() or name not in self._settings.memory_api_agent_set:
            raise MemoryApiError(403, "agent_not_allowed", f"agent 不在白名单：{name or '(空)'}")
        return name

    @staticmethod
    def _check_session(session: str) -> str:
        name = str(session or "").strip()
        if not _ID_RE.match(name):
            raise _invalid("session 只允许 [A-Za-z0-9:_-]{1,64}")
        return name

    def _check_text(self, text: str, *, field: str) -> str:
        value = str(text or "").strip()
        limit = max(1, int(self._settings.memory_api_max_text_chars))
        if not value or len(value) > limit:
            raise _invalid(f"{field} strip 后长度必须在 1..{limit}")
        return value

    # ------------------------------------------------------------------
    # turn
    # ------------------------------------------------------------------
    async def turn(
        self,
        *,
        agent: str,
        session: str,
        user_text: str,
        replies: list[str],
        at: str | None,
        message_id: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        agent = self._check_agent(agent)
        session = self._check_session(session)
        user_text = self._check_text(user_text, field="user_text")
        max_replies = max(0, int(self._settings.memory_api_max_replies))
        reply_list = [self._check_text(r, field="replies[]") for r in (replies or [])]
        if len(reply_list) > max_replies:
            raise _invalid(f"replies 最多 {max_replies} 条")
        total = len(user_text) + sum(len(r) for r in reply_list)
        if total > max(1, int(self._settings.memory_api_max_text_chars)) * max(1, max_replies):
            raise _invalid("replies 总长超过上限")
        at_text = _parse_at(at)
        if message_id is not None and not _ID_RE.match(str(message_id)):
            raise _invalid("message_id 只允许 [A-Za-z0-9:_-]{1,64}")
        if idempotency_key is not None and not _IDEMPOTENCY_RE.match(str(idempotency_key)):
            raise _invalid("idempotency_key 只允许 [A-Za-z0-9:_-]{1,128}")

        # 2) 幂等命中 → 直接返回缓存，不写缓冲、不调 update_memory
        if idempotency_key is not None:
            cached = await self._cache_get((agent, session, str(idempotency_key)))
            if cached is not None:
                return {**cached, "idempotent": True}

        async with self._buffer.session_lock(agent, session):
            # R2：锁内二次查幂等 —— 同 key 并发时后到者拿锁即返回首次结果，
            # 否则两个请求都会在锁外错过缓存、各写一次。
            if idempotency_key is not None:
                cached = await self._cache_get((agent, session, str(idempotency_key)))
                if cached is not None:
                    return {**cached, "idempotent": True}

            slot = self._buffer.slot(agent, session)

            user_id = str(message_id) if message_id is not None else uuid.uuid4().hex
            new_messages = build_turn_messages(
                user_text=user_text,
                user_id=user_id,
                replies=reply_list,
                at=at_text,
            )

            # 4) 上限检查（R3：按本轮实际追加条数判断，不静默删）
            try:
                self._buffer.ensure_capacity(
                    agent, session, incoming=len(new_messages)
                )
            except BufferOverflowError as exc:
                raise MemoryApiError(503, "buffer_overflow", str(exc)) from exc

            # 5) message_id 去重
            if self._buffer.has_message(agent, session, user_id):
                raise MemoryApiError(
                    409, "duplicate_message_id", f"message_id 已存在于缓冲：{user_id}"
                )

            # 6) 追加本轮（用户 + 每条回复，全部带 id）；R4：整槽快照用于失败回滚
            slot_snapshot = {
                "messages": [dict(m) for m in slot["messages"]],
                "turn_count": int(slot.get("turn_count", 0)),
                "last_turn_at": slot.get("last_turn_at"),
            }

            def _restore_slot() -> None:
                slot["messages"] = slot_snapshot["messages"]
                slot["turn_count"] = slot_snapshot["turn_count"]
                slot["last_turn_at"] = slot_snapshot["last_turn_at"]

            slot["messages"].extend(new_messages)

            # 7) 调 update_memory —— 缓冲全部消息 + 本轮
            lc_messages = self._to_langchain(slot["messages"])
            if not reply_list:
                # R1（§2.2）：空 replies 只写用户记忆、跳过抽取。update_memory 在
                # replies 为空时会回退取最后一条 AIMessage；末尾放一条空 AIMessage
                # 哨兵即可阻止回退，且不改 update_memory 语义。哨兵不进缓冲。
                lc_messages.append(AIMessage(content=""))
            state = {
                "agent_id": agent,
                "session_id": session,
                "messages": lc_messages,
                "replies": list(reply_list),
                "idle_scope": "",
                "degraded": False,
            }
            facts_before = {str(f.get("key")) for f in await self._store.current_facts(agent, limit=1000)}
            profile_before = await self._store.get_profile(agent)
            summary_before = await self._store.get_summary(agent, session)
            l2_before = await self._store.get_l2(agent)
            l2_counter_before = await self._store.get_l2_counter(agent)

            try:
                result = await update_memory(
                    state,
                    self._store,
                    self._llm,
                    trigger_rounds=self._settings.summary_trigger_rounds,
                    chunk_rounds=self._settings.summary_chunk_rounds,
                    extract_enabled=self._settings.memory_extract_enabled,
                    extract_min_chars=self._settings.memory_extract_min_chars,
                    extract_timeout_s=self._settings.memory_extract_timeout_s,
                    fact_mode=self._settings.memory_fact_mode,
                    profile_enabled=self._settings.memory_profile_enabled,
                    l2_enabled=self._settings.memory_l2_enabled,
                    l2_every_merges=self._settings.memory_l2_every_merges,
                    l2_max_chars=self._settings.memory_l2_max_chars,
                )
            except Exception as exc:  # noqa: BLE001 - 统一映射为 502，不持久化本轮
                _restore_slot()
                logger.warning(
                    "记忆接口 turn 写入失败（agent=%s, session=%s）：%s",
                    agent,
                    session,
                    exc,
                )
                raise MemoryApiError(502, "memory_write_failed", "记忆写入失败") from exc

            # 8) 按 RemoveMessage.id 真删
            removal_ids = [
                str(m.id)
                for m in (result.get("messages") or [])
                if getattr(m, "id", None)
            ]
            removed, missing = self._buffer.remove_message_ids(agent, session, removal_ids)
            for mid in missing:
                logger.warning(
                    "RemoveMessage id 未在缓冲中找到（agent=%s, session=%s）：%s",
                    agent,
                    session,
                    mid,
                )

            # 9) turn_count / last_turn_at + 原子写
            slot["turn_count"] = int(slot.get("turn_count", 0)) + 1
            slot["last_turn_at"] = at_text
            try:
                await self._buffer.save()
            except BufferPersistError as exc:
                # R4：写盘失败时内存也必须回到本轮之前，否则重试会重复计数/重复追加
                _restore_slot()
                raise MemoryApiError(503, "buffer_persist_failed", str(exc)) from exc

            # 10) 计算结果字段
            facts_after = {
                str(f.get("key")) for f in await self._store.current_facts(agent, limit=1000)
            }
            profile_after = await self._store.get_profile(agent)
            summary_after = await self._store.get_summary(agent, session)
            l2_after = await self._store.get_l2(agent)
            l2_counter_after = await self._store.get_l2_counter(agent)

            payload = {
                "ok": True,
                "contract": "1",
                "agent": agent,
                "session": session,
                "rounds": int(slot["turn_count"]),
                "facts_added": len(facts_after - facts_before),
                "profile_changed": profile_before != profile_after,
                "summarized": removed > 0 or summary_after != summary_before,
                "l2_merged": l2_after != l2_before or l2_counter_after < l2_counter_before,
                "buffer_size": len(slot["messages"]),
                "idempotent": False,
            }
            # 11) 写幂等缓存
            if idempotency_key is not None:
                await self._cache_put((agent, session, str(idempotency_key)), payload)
            return payload

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    async def context(
        self,
        *,
        agent: str,
        session: str,
        query: str,
        limit: int | None,
    ) -> dict[str, Any]:
        agent = self._check_agent(agent)
        session = self._check_session(session)
        query_text = str(query or "")
        if len(query_text) > _MAX_QUERY_CHARS:
            raise _invalid(f"query 最多 {_MAX_QUERY_CHARS} 字符")
        recall_limit = int(limit) if limit is not None else int(self._settings.memory_recall_limit)
        if not 1 <= recall_limit <= _MAX_LIMIT:
            raise _invalid(f"limit 必须在 1..{_MAX_LIMIT}")

        state = {
            "messages": [HumanMessage(content=query_text)],
            "agent_id": agent,
            "session_id": session,
        }
        try:
            result = await retrieve_memory(
                state,
                self._store,
                limit=recall_limit,
                extract_enabled=self._settings.memory_extract_enabled,
                weighted=self._settings.memory_weighted_recall,
                profile_enabled=self._settings.memory_profile_enabled,
                l2_enabled=self._settings.memory_l2_enabled,
                w_cosine=self._settings.memory_w_cosine,
                w_recency=self._settings.memory_w_recency,
                w_importance=self._settings.memory_w_importance,
                halflife_days=self._settings.memory_recency_halflife_days,
                overscan=self._settings.memory_recall_overscan,
                now=clock.now(),
                debug=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆接口 context 检索失败（agent=%s）：%s", agent, exc)
            raise MemoryApiError(502, "retrieval_failed", "记忆检索失败") from exc

        return {
            "ok": True,
            "contract": "1",
            "agent": agent,
            "session": session,
            "profile": result.get("user_profile") or {},
            "boundaries": [],
            "facts": [_fact_view(f) for f in (result.get("recalled_facts") or [])],
            "l2": result.get("l2_portrait") or "",
            "memories": result.get("recalled_memories") or [],
            "summary": result.get("summary") or "",
            "debug": result.get("recalled_debug") or {"pool": 0, "top": []},
        }

    # ------------------------------------------------------------------
    # forget
    # ------------------------------------------------------------------
    async def forget(self, *, agent: str, contents: list[str]) -> dict[str, Any]:
        agent = self._check_agent(agent)
        raw = [str(c or "") for c in (contents or [])]
        if not 1 <= len(raw) <= _MAX_CONTENTS:
            raise _invalid(f"contents 条数必须在 1..{_MAX_CONTENTS}")
        seen: set[str] = set()
        unique: list[str] = []
        for content in raw:
            value = self._check_text(content, field="contents[]")
            if value not in seen:
                seen.add(value)
                unique.append(value)

        deleted = 0
        missing: list[str] = []
        try:
            for content in unique:
                if await self._store.delete(agent, content):
                    deleted += 1
                else:
                    missing.append(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆接口 forget 失败（agent=%s）：%s", agent, exc)
            raise MemoryApiError(502, "forget_failed", "记忆删除失败") from exc

        return {
            "ok": True,
            "contract": "1",
            "agent": agent,
            "deleted": deleted,
            "missing": missing,
        }

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    async def state(self, *, agent: str, session: str | None) -> dict[str, Any]:
        agent = self._check_agent(agent)
        session_name = self._check_session(session) if session is not None else None
        try:
            entries = await self._store.count_entries(agent)
            facts = len(await self._store.current_facts(agent, limit=1000))
            profile_slots = len(await self._store.get_profile(agent))
            l2_merges = await self._store.get_l2_counter(agent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆接口 state 读取失败（agent=%s）：%s", agent, exc)
            raise MemoryApiError(502, "state_failed", "记忆状态读取失败") from exc

        return {
            "ok": True,
            "contract": "1",
            "agent": agent,
            "session": session_name,
            "entries": entries,
            "facts": facts,
            "profile_slots": profile_slots,
            "boundaries": 0,
            "l2_merges": l2_merges,
            "buffer_size": self._buffer.size(agent, session_name),
            "last_turn_at": self._buffer.last_turn_at(agent, session_name),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _to_langchain(messages: list[dict]) -> list:
        out: list = []
        for message in messages:
            if message.get("role") == "user":
                at = message.get("at")
                out.append(
                    HumanMessage(
                        id=message["id"],
                        content=message["content"],
                        additional_kwargs={"at": at} if at else {},
                    )
                )
            else:
                out.append(AIMessage(id=message["id"], content=message["content"]))
        return out

    async def _cache_get(self, key: tuple[str, str, str]) -> dict | None:
        async with self._idem_lock:
            self._purge_idem()
            hit = self._idem.get(key)
            if hit is None:
                return None
            expires, payload = hit
            if expires <= time.monotonic():
                self._idem.pop(key, None)
                return None
            return dict(payload)

    async def _cache_put(self, key: tuple[str, str, str], payload: dict) -> None:
        async with self._idem_lock:
            self._purge_idem()
            ttl = max(1, int(self._settings.memory_api_idempotency_ttl_s))
            self._idem[key] = (time.monotonic() + ttl, dict(payload))
            cap = max(1, int(self._settings.memory_api_idempotency_max_entries))
            while len(self._idem) > cap:
                oldest = min(self._idem, key=lambda k: self._idem[k][0])
                self._idem.pop(oldest, None)

    def _purge_idem(self) -> None:
        now = time.monotonic()
        for key in [k for k, (expires, _) in self._idem.items() if expires <= now]:
            self._idem.pop(key, None)
