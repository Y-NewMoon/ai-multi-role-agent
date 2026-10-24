"""节点：拼装发给 LLM 的消息（system + 摘要 + 召回记忆 + 设定资料 + 近 N 轮）。

方案 3.9：总预算 8k token（可配），按优先级裁剪。
V2.1（契约 4.7）在原有段落之外新增 lore 段（always-on 设定 + 检索 lore）。
V2.3（契约 §2.8）新增三段：profile（常驻）、facts、L2；均为空时不插入任何字符
——保证 `memory_extract_enabled=false` 时与 V2.2 prompt 逐字一致。

裁剪顺序（§2.8-3）：① always-on lore → ② 滚动摘要 → ③ 语义记忆 + facts
→ ④ L2 → ⑤ profile → ⑥ 检索 lore → ⑦ 近 N 轮。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import BaseMessage

from app.agents.state import RolePlayState
from app.memory.facts import render_facts
from app.memory.profile import render_profile
from app.observability.trace import current_trace
from app.prompts.roleplay_system import build_idle_instruction, build_idle_nudge
from app.skills.registry import SkillRegistry

_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}

TZ = ZoneInfo("Asia/Shanghai")

LORE_HEADER = "【设定资料】"
# V2.2：工具白名单段（只放 name + when_to_use，不放 JSON Schema，契约 §2.4/§2.7）
TOOLS_HEADER = "【可用工具】"
# 工具轮的时间锚点：模型不知道今天几号，无法把「今晚8点35」换算成绝对时间
TIME_HEADER = "【当前时间】"
# V2.3：L2 关系档案段
L2_HEADER = "【你和这位客人的过往】"


def _to_chat_messages(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = _ROLE_MAP.get(getattr(m, "type", "human"), "user")
        out.append({"role": role, "content": str(m.content)})
    return out


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def _recent_rounds(messages: list[BaseMessage], rounds: int) -> list[BaseMessage]:
    """取最近 rounds 轮（一轮 = 一条用户消息 + 其后的角色回复）。

    M5：一轮可能含多条角色回复，故按「轮」而非「2N 条消息」计数。
    """
    if rounds <= 0:
        return list(messages)
    human_idx = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
    if len(human_idx) <= rounds:
        return list(messages)
    return list(messages[human_idx[len(human_idx) - rounds] :])


def _estimate_tokens(text: str) -> int:
    """无第三方 tokenizer 时的保守估算：CJK 约 1 字/token，其余约 4 字/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + (len(text) - cjk + 3) // 4


def _build_system(
    base: str,
    summary: str,
    memories: list[str],
    lore_block: str = "",
    tools_block: str = "",
    now_block: str = "",
    profile_block: str = "",
    facts_block: str = "",
    l2_block: str = "",
) -> str:
    """把 persona 基座与（可选的）档案/设定资料/摘要/记忆/事实/L2/工具清单拼成 system 文本。

    所有可选段均为空时与 V2.0/V2.1/V2.2 输出逐字一致（契约 §1.3-6）。
    新三段顺序固定：profile 在摘要之前；facts 紧随召回记忆；L2 在 facts 之后。
    """
    parts = [base]
    if lore_block:
        parts.append(lore_block)
    if profile_block:
        parts.append(profile_block)
    if summary:
        parts.append(f"【本会话摘要】\n{summary}")
    if memories:
        bullet = "\n".join(f"- {m}" for m in memories)
        parts.append(f"【你记得的相关信息】\n{bullet}")
    if facts_block:
        parts.append(facts_block)
    if l2_block:
        parts.append(l2_block)
    if tools_block:
        parts.append(tools_block)
    if now_block:
        parts.append(now_block)
    return "\n\n".join(p for p in parts if p)


def _now_block(now: datetime) -> str:
    """给启用工具的角色一个绝对时间锚点（仅在工具有效时插入）。"""
    weekday = "一二三四五六日"[now.weekday()]
    return (
        f"{TIME_HEADER}\n"
        f"现在是 {now.strftime('%Y-%m-%d %H:%M')}（Asia/Shanghai，周{weekday}）。"
        "用户说「今晚8点35」「明天早上7点」这类相对时间时，先按此换算成绝对时间。"
    )


def _tools_block(registry: SkillRegistry | None, allowed: list[str]) -> str:
    """白名单非空、已注册且可用时给出一段「名称 + 何时用」清单；否则返回空（逐字不变的保证）。"""
    if registry is None or not allowed:
        return ""
    skills = registry.resolve(allowed, only_available=True)
    if not skills:
        return ""
    lines = "\n".join(f"- {skill.name}：{skill.when_to_use}" for skill in skills)
    rule = "（必须依据工具返回结果作答；工具未返回成功时不得声称已完成，例如提醒没建成就不能说“已设置”。）"
    return f"{TOOLS_HEADER}\n{lines}\n{rule}"


# ---- lore 段（契约 4.7）----


def _lore_line(item: dict) -> str:
    """一行一个 chunk：`（header_path）content`。"""
    return f"（{item.get('header_path', '')}）{item.get('content', '')}"


def _lore_block(lore: list[dict]) -> str:
    if not lore:
        return ""
    return LORE_HEADER + "\n" + "\n".join(_lore_line(item) for item in lore)


def _item_tokens(item: dict) -> int:
    return _estimate_tokens(_lore_line(item))


def _is_always_on(item: dict) -> bool:
    return "always_on" in (item.get("routes") or [])


def _source_ref_of(item: dict) -> str:
    return str(item.get("source_ref", ""))


def _lowest_score_index(lore: list[dict]) -> int:
    """rerank 分最低者（并列按 source_ref 升序），保证确定性。"""
    return min(range(len(lore)), key=lambda k: (item_score(lore[k]), _source_ref_of(lore[k])))


def item_score(item: dict) -> float:
    try:
        return float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _largest_always_on_index(lore: list[dict]) -> int | None:
    """always-on 段里 source_ref 最大者的下标；无 always_on 项时返回 None。

    并列无关（source_ref 在库内唯一）；确定性优先。
    """
    idxs = [k for k, item in enumerate(lore) if _is_always_on(item)]
    if not idxs:
        return None
    return max(idxs, key=lambda k: _source_ref_of(lore[k]))


async def compose_prompt(
    state: RolePlayState,
    *,
    token_budget: int = 8000,
    recent_rounds: int = 10,
    lore_max_ratio: float = 0.20,
    lore_always_on_ratio: float = 0.08,
    profile_ratio: float = 0.10,
    l2_ratio: float = 0.05,
    skill_registry: SkillRegistry | None = None,
    allowed_skills: list[str] | None = None,
    now: datetime | None = None,
) -> dict:
    base = state.get("system_prompt", "")
    memories = list(state.get("recalled_memories") or [])
    summary = state.get("summary", "") or ""
    lore = list(state.get("recalled_lore") or [])
    # V2.3 三段（默认空；空时 _build_system 不插入任何字符）
    profile = dict(state.get("user_profile") or {})
    facts = list(state.get("recalled_facts") or [])
    l2 = state.get("l2_portrait") or ""
    profile_block = await render_profile(profile)
    facts_block = await render_facts(facts)
    l2_block = f"{L2_HEADER}\n{l2}" if l2 else ""
    # 白名单：优先取状态（服务层按角色注入），否则回落到图构造时的默认（契约 §2.4）
    allowed = list(state.get("allowed_skills") or allowed_skills or [])
    tools_block = _tools_block(skill_registry, allowed)
    # 时间锚点可注入（V3 M1 前先把不可复现的 datetime.now 收口成参数）
    now_block = _now_block(now or datetime.now(TZ)) if tools_block else ""

    # 自主闲聊：临时把"主动开口"指引拼到 system 末尾（不改人设基座，不影响正常轮）
    idle_scope = state.get("idle_scope") or ""
    if idle_scope:
        base = f"{base}\n\n{build_idle_instruction(idle_scope)}"

    recent = _to_chat_messages(_recent_rounds(state.get("messages", []), recent_rounds))

    def _system() -> str:
        return _build_system(
            base,
            summary,
            memories,
            _lore_block(lore),
            tools_block,
            now_block,
            profile_block,
            facts_block,
            l2_block,
        )

    def _total() -> int:
        return _estimate_tokens(_system()) + sum(_estimate_tokens(m["content"]) for m in recent)

    # 段配额（契约 4.7）：always-on ≤ 8%，lore 合计 ≤ 20%
    always_cap = int(token_budget * lore_always_on_ratio)
    lore_cap = int(token_budget * lore_max_ratio)
    while True:
        always = [k for k, item in enumerate(lore) if _is_always_on(item)]
        if not always or sum(_item_tokens(lore[k]) for k in always) <= always_cap:
            break
        lore.pop(_largest_always_on_index(lore))
    while lore and sum(_item_tokens(item) for item in lore) > lore_cap:
        lore.pop(_lowest_score_index(lore))

    # V2.3 段预算（§2.5 / §2.8-4）：profile ≤ profile_ratio；三段合计 ≤ profile + l2 + 10%
    profile_cap = int(token_budget * profile_ratio)
    l2_cap = int(token_budget * l2_ratio)
    combo_cap = int(token_budget * (profile_ratio + l2_ratio + 0.10))
    while profile and _estimate_tokens(profile_block) > profile_cap:
        # 从低优先级槽位开始丢（保持白名单顺序的前缀）
        last = next((slot for slot in reversed(tuple(profile)) if slot in profile), None)
        if last is None:
            break
        profile.pop(last)
        profile_block = await render_profile(profile)
    if l2_block and _estimate_tokens(l2_block) > l2_cap:
        l2, l2_block = "", ""
    while (
        facts
        and _estimate_tokens(profile_block)
        + _estimate_tokens(facts_block)
        + _estimate_tokens(l2_block)
        > combo_cap
    ):
        # 先删 importance 最低的 fact（并列删最后一条，确定性）
        worst = min(
            range(len(facts)),
            key=lambda k: (float(facts[k].get("importance", 0.5)), k),
        )
        facts.pop(worst)
        facts_block = await render_facts(facts)
    if (
        _estimate_tokens(profile_block) + _estimate_tokens(l2_block) > combo_cap
    ):
        l2, l2_block = "", ""
    if _estimate_tokens(profile_block) > combo_cap:
        profile, profile_block = {}, ""

    # 全局超预算裁剪顺序（§2.8-3）：① always-on → ② 摘要 → ③ 记忆 + facts
    # → ④ L2 → ⑤ profile → ⑥ 检索 lore → ⑦ 轮次
    while _total() > token_budget:
        idx = _largest_always_on_index(lore)
        if idx is None:
            break
        lore.pop(idx)
    if _total() > token_budget and summary:
        summary = ""
    if _total() > token_budget and memories:
        memories = []
    while _total() > token_budget and facts:
        worst = min(
            range(len(facts)),
            key=lambda k: (float(facts[k].get("importance", 0.5)), k),
        )
        facts.pop(worst)
        facts_block = await render_facts(facts)
    if _total() > token_budget and l2_block:
        l2, l2_block = "", ""
    if _total() > token_budget and profile_block:
        profile, profile_block = {}, ""
    while _total() > token_budget and lore:
        lore.pop(_lowest_score_index(lore))
    while _total() > token_budget and len(recent) > 1:
        recent.pop(0)

    prompt = [{"role": "system", "content": _system()}]
    prompt.extend(recent)
    # 闲聊轮：透传给 LLM 的消息若以 assistant 回合收尾，模型易退化为「只输出内心独白、无正文」。
    # 末尾补一条合成 user 提示（不进状态/记忆/快照），使 prompt 以 user 收尾即可纠正。
    if idle_scope:
        prompt.append({"role": "user", "content": build_idle_nudge(idle_scope)})
    # V2.4：只记录预算估算结果，不二次截断（裁剪已在上文完成）
    recorder = current_trace()
    if recorder is not None:
        recorder.set_budget(
            prompt_token_budget=token_budget,
            estimated_prompt_tokens=sum(
                _estimate_tokens(str(m.get("content") or "")) for m in prompt
            ),
        )
    return {"prompt": prompt}
