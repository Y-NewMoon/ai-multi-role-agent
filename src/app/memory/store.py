"""LangGraph Store 的薄封装——记忆读写的唯一入口。

接缝 5（方案 3.11A#5）：图内节点不访问业务表，记忆读写一律经注入的 Store。
会话隔离（接缝 2）：namespace 只由 agent_id 决定，不掺 session。

M2 起：
- 语义记忆：`search` 带 query 时走向量索引 top-k（见 main.py 注入的 index）；
- 情景记忆：滚动摘要以 key=`summary:{session_id}` 落在同一 namespace（方案 3.10），
  由 `put_summary` / `get_summary` 读写，并在 `search` 中剔除，避免重复占用 prompt 预算。

M6（读侧可见性）：群聊扇出会把**他人**的公屏发言写进本角色 namespace
（标 `visibility="group"` + `speaker_agent_id`）。这类条目的正文是别人的台词，
`search` 不再返回，理由见 `_is_other_group_line`。
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from langgraph.store.base import BaseStore

from app.memory.namespace import memory_namespace

logger = logging.getLogger(__name__)

SUMMARY_KEY_PREFIX = "summary:"
FACT_PREFIX = "fact:"
FACT_CUR_PREFIX = "fact_cur:"
PROFILE_PREFIX = "profile:"
L2_PORTRAIT_KEY = "l2:portrait"
L2_COUNTER_KEY = "l2:counter"

# V2.3 档案白名单（profile.py 复用同一份定义，避免两处漂移）
_PROFILE_SLOTS = ("称呼", "禁忌", "关系阶段", "重要日期")

# V2.3：这些 kind 不参与语义记忆召回（各有专门注入通道）
_NON_RECALL_KINDS = {"summary", "fact", "fact_cur", "profile", "l2", "l2_counter"}

# 语义召回时按 limit+overscan 取，再剔除摘要等条目，保证过滤后仍尽量凑够 limit。
_SEARCH_OVERSCAN = 10
_MAX_FACTS_PER_AGENT = 200


def utc_now_iso() -> str:
    """UTC ISO8601（V2.3 所有写入的 created_at / updated_at 统一口径）。"""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class FactWrite:
    fact_key: str
    created: bool  # False = 该 (subject,predicate,object) 已存在（幂等）
    superseded: list[str]  # 被本次取代的旧 fact key（keep_both 模式为空）


def _fact_key(agent_id: int | str, subject: str, predicate: str, obj: str) -> str:
    digest = hashlib.sha1(f"{agent_id}|{subject}|{predicate}|{obj}".encode("utf-8")).hexdigest()
    return f"{FACT_PREFIX}{digest[:24]}"


def _fact_cur_key(agent_id: int | str, subject: str, predicate: str) -> str:
    digest = hashlib.sha1(f"{agent_id}|{subject}|{predicate}".encode("utf-8")).hexdigest()
    return f"{FACT_CUR_PREFIX}{digest[:24]}"


def _profile_key(slot: str) -> str:
    return f"{PROFILE_PREFIX}{slot}"


def _is_other_group_line(value: dict, agent_id: int | str) -> bool:
    """是否为「群里他人公屏发言」的扇出副本？

    群聊扇出（`GroupService._fanout`）把每位成员的公屏发言写进其余成员各自的
    namespace（`visibility="group"`、`speaker_agent_id=发言人`）。这类条目正文是别人
    的台词，一旦混进 `【你记得的相关信息】`（compose_prompt）就会被模型当成"自己记
    得/说过的话"照抄，实测两名角色输出一字不差的重复回复并出现身份混淆。
    他人发言的可见性由**可归属**的渠道负责：本轮逐条注入（supervisor，带 `【角色名】`
    前缀）+ 群聊快照历史；语义召回只放本人的条目。
    """
    if value.get("visibility") != "group":
        return False
    speaker = value.get("speaker_agent_id")
    if speaker is None:
        return False  # 用户发言（无发言人）对所有成员都算"我收到的消息"，保留
    return str(speaker) != str(agent_id)


def _memory_key(agent_id: int | str, content: str) -> str:
    """以 (agent_id, content) 为单位生成稳定 key（接缝 4）。"""
    digest = hashlib.sha1(f"{agent_id}:{content}".encode("utf-8")).hexdigest()
    return digest[:32]


def _summary_key(session_id: int | str) -> str:
    """摘要 key 形如 summary:{session_id}（方案 3.10）：同会话重复摘要覆盖同一条。"""
    return f"{SUMMARY_KEY_PREFIX}{session_id}"


class MemoryStore:
    """封装 LangGraph Store，对外只暴露记忆语义。"""

    def __init__(self, store: BaseStore, *, fact_limit: int | None = None) -> None:
        self._store = store
        # 契约 §1.3#10：每 agent 事实上限可配（默认 200；延迟解析以便测试/配置覆盖）
        self._fact_limit = max(1, int(fact_limit if fact_limit is not None else _MAX_FACTS_PER_AGENT))

    async def put(
        self, agent_id: int | str, content: str, *, kind: str = "memory", **metadata: object
    ) -> None:
        """写入一条记忆，粒度 = (agent_id, content)。"""
        ns = memory_namespace(agent_id)
        value = {"text": content, "kind": kind, **metadata}
        await self._store.aput(ns, _memory_key(agent_id, content), value)

    async def put_summary(self, agent_id: int | str, session_id: int | str, text: str) -> None:
        """写入/覆盖该会话的滚动摘要（同 namespace，摘要文本同样会被向量化）。"""
        ns = memory_namespace(agent_id)
        await self._store.aput(ns, _summary_key(session_id), {"text": text, "kind": "summary"})

    async def get_summary(self, agent_id: int | str, session_id: int | str) -> str:
        """读取该会话的滚动摘要，无则返回空串。"""
        ns = memory_namespace(agent_id)
        item = await self._store.aget(ns, _summary_key(session_id))
        if item is None:
            return ""
        return str(item.value.get("text", ""))

    async def search(
        self, agent_id: int | str, query: str | None = None, limit: int = 5
    ) -> list[str]:
        """召回该角色的语义记忆文本。

        剔除三类条目：摘要（单独注入，方案 3.10）、V2.3 的 facts/profile/L2/计数器
        （各有专门注入通道）、**群内他人发言**的扇出副本（见 `_is_other_group_line`），
        以及已失效事实（`superseded_at` 非空）。
        返回类型与既有语义不变（`list[str]`），`golden.memory_hit_rate` 依赖它。
        """
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, query=query, limit=limit + _SEARCH_OVERSCAN)
        out: list[str] = []
        for item in items:
            value = item.value
            if not self._recallable(value, agent_id):
                continue
            text = value.get("text")
            if text:
                out.append(str(text))
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _recallable(value: dict, agent_id: int | str) -> bool:
        """是否可作为语义记忆候选（V2.3 统一过滤规则）。"""
        if value.get("kind") in _NON_RECALL_KINDS:
            return False
        if _is_other_group_line(value, agent_id):
            return False
        if value.get("superseded_at") is not None:
            return False
        return True

    async def search_scored(
        self, agent_id: int | str, *, query: str, limit: int = 5, overscan: int = 50
    ) -> list[dict]:
        """返回候选（含 cos / created_at / importance），供应用层加权重排。

        过滤规则同 `search`（kind → 群聊他人发言 → superseded）；候选最多 `overscan` 条。
        `score` 直接取 Store 的 cosine similarity（LangGraph 的 InMemory/Postgres 后端
        均返回 `1 - distance`，本类不做二次换算）。
        """
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, query=query, limit=max(limit, overscan) + _SEARCH_OVERSCAN)
        out: list[dict] = []
        for item in items:
            value = dict(item.value or {})
            if not self._recallable(value, agent_id):
                continue
            text = value.get("text")
            if not text:
                continue
            out.append(
                {
                    "key": item.key,
                    "text": str(text),
                    "role": value.get("role"),
                    "kind": value.get("kind", "memory"),
                    "score": item.score,
                    "importance": float(value.get("importance", 0.5)),
                    "created_at": value.get("created_at")
                    or (item.created_at.isoformat() if item.created_at else None),
                    "updated_at": value.get("updated_at")
                    or (item.updated_at.isoformat() if item.updated_at else None),
                }
            )
            if len(out) >= overscan:
                break
        return out

    # ------------------------------------------------------------------
    # V2.3 事实（fact / fact_cur）
    # ------------------------------------------------------------------
    async def put_fact(
        self,
        agent_id: int | str,
        *,
        subject: str,
        predicate: str,
        obj: str,
        importance: float = 0.5,
        confidence: float = 0.9,
        source_message_id: str | None = None,
        turn_id: str = "",
        mode: Literal["overwrite", "keep_both_with_note"] = "overwrite",
    ) -> FactWrite:
        """写入一条事实；同 (subject, predicate) 的新 object 按 mode 处理冲突。"""
        ns = memory_namespace(agent_id)
        now = utc_now_iso()
        fact_key = _fact_key(agent_id, subject, predicate, obj)
        cur_key = _fact_cur_key(agent_id, subject, predicate)
        existing = await self._store.aget(ns, fact_key)
        cur = await self._store.aget(ns, cur_key)
        cur_value = dict(cur.value or {}) if cur is not None else {}
        old_key = cur_value.get("fact_key")
        has_conflict = bool(old_key and old_key != fact_key)

        if existing is not None:
            # 幂等：同三元组已存在。若此前被失效、现在又被重新主张，则恢复为 active。
            if existing.value.get("superseded_at") is None:
                return FactWrite(fact_key=fact_key, created=False, superseded=[])
            superseded: list[str] = []
            if has_conflict and mode == "overwrite":
                old = await self._store.aget(ns, old_key)
                if old is not None and old.value.get("superseded_at") is None:
                    await self._store.aput(
                        ns, old_key, {**old.value, "superseded_at": now, "updated_at": now}
                    )
                    superseded.append(old_key)
            await self._store.aput(
                ns, fact_key, {**existing.value, "superseded_at": None, "updated_at": now}
            )
            await self._store.aput(
                ns,
                cur_key,
                {
                    "kind": "fact_cur",
                    "fact_key": fact_key,
                    "object": obj,
                    "supersedes": old_key if has_conflict else None,
                    "updated_at": now,
                },
            )
            return FactWrite(fact_key=fact_key, created=False, superseded=superseded)

        superseded = []
        if has_conflict and mode == "overwrite":
            old = await self._store.aget(ns, old_key)
            if old is not None and old.value.get("superseded_at") is None:
                await self._store.aput(
                    ns, old_key, {**old.value, "superseded_at": now, "updated_at": now}
                )
                superseded.append(old_key)

        await self._store.aput(
            ns,
            fact_key,
            {
                "text": f"{subject}{predicate}={obj}",
                "kind": "fact",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "importance": float(importance),
                "confidence": float(confidence),
                "source_message_id": source_message_id,
                "turn_id": turn_id,
                "superseded_at": None,
                "supersedes": old_key if has_conflict else None,
                "created_at": now,
                "updated_at": now,
            },
        )
        await self._store.aput(
            ns,
            cur_key,
            {
                "kind": "fact_cur",
                "fact_key": fact_key,
                "object": obj,
                "supersedes": old_key if has_conflict else None,
                "updated_at": now,
            },
        )
        await self._trim_facts(agent_id)
        return FactWrite(fact_key=fact_key, created=True, superseded=superseded)

    async def stats(self, agent_id: int | str) -> dict[str, int]:
        """C-03 观察（§2.1-14）：namespace 条目 / 未失效事实 / profile 槽位。

        与 `_all_facts` 同口径，单次 `asearch(limit=1000)` 封顶；调用方（retrieve_memory）
        只在 `TRACE_MEMORY_STATS=true` 时走这里，默认不产生额外扫描。
        """
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, limit=1000)
        facts = 0
        profile_slots = 0
        for item in items:
            value = dict(item.value or {})
            kind = value.get("kind")
            if kind == "fact" and value.get("superseded_at") is None:
                facts += 1
            elif kind == "profile":
                profile_slots += 1
        return {
            "namespace_entries": len(items),
            "facts": facts,
            "profile_slots": profile_slots,
        }

    async def current_facts(self, agent_id: int | str, *, limit: int = 20) -> list[dict]:
        """未失效事实，按 importance 降序、updated_at 降序。"""
        facts = await self._all_facts(agent_id)
        facts.sort(
            key=lambda f: (
                -float(f.get("importance", 0.5)),
                str(f.get("updated_at") or ""),
                str(f.get("key") or ""),
            )
        )
        return facts[: max(0, limit)]

    async def _all_facts(self, agent_id: int | str) -> list[dict]:
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, limit=1000)
        facts: list[dict] = []
        for item in items:
            value = dict(item.value or {})
            if value.get("kind") != "fact" or value.get("superseded_at") is not None:
                continue
            value["key"] = item.key
            facts.append(value)
        return facts

    async def _trim_facts(self, agent_id: int | str) -> None:
        """事实数 > `fact_limit` 时淘汰：**先删 superseded（审计），active 永远优先保留**。

        排序键第一维 `superseded_at is not None` → 已失效条目排在前面（先出局）；
        只有在 active 自身就超过上限时，才按 importance / updated_at 淘汰最弱的 active。
        这样同一 (subject, predicate) 反复改口时不会把 active 饿死。
        """
        ns = memory_namespace(agent_id)
        items = await self._store.asearch(ns, limit=1000)
        facts = [
            (item.key, dict(item.value or {}))
            for item in items
            if (item.value or {}).get("kind") == "fact"
        ]
        if len(facts) <= self._fact_limit:
            return
        facts.sort(
            key=lambda pair: (
                0 if pair[1].get("superseded_at") is not None else 1,
                float(pair[1].get("importance", 0.5)),
                str(pair[1].get("updated_at") or ""),
                pair[0],
            )
        )
        for key, _ in facts[: len(facts) - self._fact_limit]:
            await self._store.adelete(ns, key)

    # ------------------------------------------------------------------
    # V2.3 档案（profile）
    # ------------------------------------------------------------------
    async def put_profile(self, agent_id: int | str, slot: str, value: str) -> None:
        """写入/覆盖一个白名单槽位；空串视为清除该槽位。"""
        if slot not in _PROFILE_SLOTS:
            raise ValueError(f"非法 profile 槽位：{slot!r}")
        ns = memory_namespace(agent_id)
        text = (value or "").strip()
        if not text:
            await self._store.adelete(ns, _profile_key(slot))
            return
        await self._store.aput(
            ns,
            _profile_key(slot),
            {"kind": "profile", "text": text, "slot": slot, "updated_at": utc_now_iso()},
        )

    async def get_profile(self, agent_id: int | str) -> dict[str, str]:
        """读取 4 个白名单槽位；缺失的键不出现。"""
        ns = memory_namespace(agent_id)
        out: dict[str, str] = {}
        for slot in _PROFILE_SLOTS:
            item = await self._store.aget(ns, _profile_key(slot))
            if item is not None:
                text = str(item.value.get("text", "")).strip()
                if text:
                    out[slot] = text
        return out

    # ------------------------------------------------------------------
    # V2.3 L2 关系档案（按 agent，不掺 session）
    # ------------------------------------------------------------------
    async def put_l2(self, agent_id: int | str, text: str) -> None:
        ns = memory_namespace(agent_id)
        await self._store.aput(
            ns, L2_PORTRAIT_KEY, {"kind": "l2", "text": text, "updated_at": utc_now_iso()}
        )

    async def get_l2(self, agent_id: int | str) -> str:
        ns = memory_namespace(agent_id)
        item = await self._store.aget(ns, L2_PORTRAIT_KEY)
        if item is None:
            return ""
        return str(item.value.get("text", ""))

    async def bump_l2_counter(self, agent_id: int | str) -> int:
        """L1 合并计数 +1，返回新值（不存在则从 0 起）。"""
        ns = memory_namespace(agent_id)
        item = await self._store.aget(ns, L2_COUNTER_KEY)
        merges = int((item.value or {}).get("merges", 0)) if item is not None else 0
        merges += 1
        await self._store.aput(
            ns,
            L2_COUNTER_KEY,
            {"kind": "l2_counter", "merges": merges, "updated_at": utc_now_iso()},
        )
        return merges

    async def get_l2_counter(self, agent_id: int | str) -> int:
        ns = memory_namespace(agent_id)
        item = await self._store.aget(ns, L2_COUNTER_KEY)
        if item is None:
            return 0
        return int(item.value.get("merges", 0))

    async def reset_l2_counter(self, agent_id: int | str) -> None:
        """L2 合成成功后归零（失败时不调用）。"""
        ns = memory_namespace(agent_id)
        await self._store.aput(
            ns,
            L2_COUNTER_KEY,
            {"kind": "l2_counter", "merges": 0, "updated_at": utc_now_iso()},
        )

    async def delete(self, agent_id: int | str, content: str) -> bool:
        """按内容精确删除一条语义记忆（撤回消息时用；不影响该角色其余记忆）。

        key 由 (agent_id, content) 决定，故只删属于该内容的条目。
        M2：返回是否真的存在并删除（既有调用方忽略返回值，语义不变）。
        """
        ns = memory_namespace(agent_id)
        key = _memory_key(agent_id, content)
        item = await self._store.aget(ns, key)
        if item is None:
            return False
        await self._store.adelete(ns, key)
        return True

    async def count_entries(self, agent_id: int | str, *, page_size: int = 1000) -> int:
        """统计该 agent namespace 的全部条目数（分页 `asearch`，不受单次 1000 封顶）。

        `stats()` 的 `namespace_entries` 用单次 `asearch(limit=1000)` 封顶，
        M2 `GET /memory/state` 需要真实总数，故单独分页统计。
        """
        ns = memory_namespace(agent_id)
        size = max(1, int(page_size))
        total = 0
        offset = 0
        while True:
            items = await self._store.asearch(ns, limit=size, offset=offset)
            if not items:
                break
            total += len(items)
            offset += len(items)
        return total

    async def delete_all(self, agent_id: int | str) -> None:
        """清空该角色的整个 namespace（语义记忆 + 全部会话摘要）。

        用于删除角色时的级联清理；向量随 store_vectors 的 ON DELETE CASCADE 一并删除。
        """
        ns = memory_namespace(agent_id)
        # 单次 asearch 有 limit 上限（默认最多 1000 条），必须分页循环删至空，
        # 否则条目超过上限时会残留（实测 1005 条 → 残留 5 条）。
        while True:
            items = await self._store.asearch(ns, limit=1000)
            if not items:
                break
            for item in items:
                await self._store.adelete(item.namespace, item.key)

    async def delete_summary(self, agent_id: int | str, session_id: int | str) -> None:
        """删除某会话的滚动摘要（删除会话时调用；语义记忆属角色，不在此删）。"""
        ns = memory_namespace(agent_id)
        await self._store.adelete(ns, _summary_key(session_id))
