"""supervisor 调度硬约束测试（ADR-0004 §3.4）——纯逻辑 + 轻量替身，不需真库。"""

from app.agents.supervisor import (
    GroupMember,
    Supervisor,
    choose_next,
    parse_decision,
    strip_speaker_prefix,
)


# ===== 决策解析 =====

def test_parse_decision_speak():
    assert parse_decision("2", {1, 2, 3}) == ("speak", 2)
    assert parse_decision("  3 号 ", {1, 2, 3}) == ("speak", 3)


def test_parse_decision_end_and_invalid():
    assert parse_decision("END", {1, 2}) == ("end", None)
    assert parse_decision("本轮结束", {1, 2}) == ("invalid", None)
    assert parse_decision("99", {1, 2}) == ("invalid", None)  # 不在成员里
    assert parse_decision("", {1, 2}) == ("invalid", None)


# ===== 选择逻辑（硬约束）=====

def test_choose_next_respects_max_turns_and_one_per_person():
    assert choose_next(candidate=None, order=[1, 2, 3], spoken=[1, 2], last_speaker=2, max_turns=2) is None
    assert choose_next(candidate=None, order=[1, 2], spoken=[1, 2], last_speaker=2, max_turns=4) is None


def test_choose_next_accepts_valid_candidate():
    assert choose_next(candidate=3, order=[1, 2, 3], spoken=[1], last_speaker=1, max_turns=4) == 3


def test_choose_next_forces_change_on_consecutive_same():
    # LLM 连续选同一人 → 强制换人（退化轮询取下一个未发言者）
    assert choose_next(candidate=1, order=[1, 2, 3], spoken=[1], last_speaker=1, max_turns=4) == 2


def test_choose_next_falls_back_when_candidate_already_spoken_or_invalid():
    assert choose_next(candidate=1, order=[1, 2, 3], spoken=[1], last_speaker=1, max_turns=4) == 2
    assert choose_next(candidate=None, order=[1, 2, 3], spoken=[1], last_speaker=1, max_turns=4) == 2


# ===== supervisor.run（用替身图/LLM 验证调度序列）=====

class _StubGraph:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def astream(self, state, *, config, stream_mode):
        yield ("values", {"reply": self._reply})


class _FixedDecisionLLM:
    """决策恒返回同一编号，用于验证"连续同人强制换人"。"""

    def __init__(self, decision: str) -> None:
        self._decision = decision

    async def generate(self, messages):
        return self._decision

    async def astream(self, messages):  # 不被调用
        yield ""


def _members(*ids: int) -> list[GroupMember]:
    return [GroupMember(agent_id=i, name=f"角色{i}", persona_prompt="人设", background="") for i in ids]


async def _run_order(max_turns: int, decision: str) -> list[int]:
    sup = Supervisor(
        llm=_FixedDecisionLLM(decision),
        graph_factory=lambda: _StubGraph("回复"),
        members=_members(1, 2, 3),
        max_turns=max_turns,
    )
    return [d["agent_id"] for e, d in [x async for x in sup.run(group_session_id=1, user_message="hi")] if e == "speaker"]


async def test_supervisor_forces_different_speakers_each_turn():
    # 决策一直是 1：第一轮 1 发言，之后被强制换人 → 1,2,3
    assert await _run_order(max_turns=4, decision="1") == [1, 2, 3]


async def test_supervisor_respects_max_turns():
    assert await _run_order(max_turns=2, decision="1") == [1, 2]


async def test_supervisor_end_after_first_speaker_ends_turn():
    # 首轮先兜底出一人；随后 LLM 返回 END → 结束
    sup = Supervisor(
        llm=_FixedDecisionLLM("END"),
        graph_factory=lambda: _StubGraph("回复"),
        members=_members(1, 2),
        max_turns=4,
    )
    events = [x async for x in sup.run(group_session_id=1, user_message="hi")]
    speakers = [d["agent_id"] for e, d in events if e == "speaker"]
    assert speakers == [1]  # 至少一人回复用户


# ===== supervisor.run_idle（自主闲聊：公平轮转 / 每人≤N / 累计≤M）=====


class _IdleStubGraph:
    """闲聊图替身：values 快照带 replies（run_idle 只读它）。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies

    async def astream(self, state, *, config, stream_mode):
        yield ("values", {"replies": list(self._replies), "reply": "\n".join(self._replies)})


def _idle_supervisor(*, replies: list[str], members: list[GroupMember]) -> Supervisor:
    # run_idle 不使用 LLM 决策，决策替身仅用于满足构造签名
    return Supervisor(
        llm=_FixedDecisionLLM("1"),
        graph_factory=lambda: _IdleStubGraph(replies),
        members=members,
        max_turns=4,
    )


async def _idle_events(sup, *, max_total: int, turn_max: int, rotate_after: int | None = None):
    return [
        x
        async for x in sup.run_idle(
            group_session_id=1,
            max_total=max_total,
            turn_max_messages=turn_max,
            rotate_after=rotate_after,
        )
    ]


async def test_run_idle_rotates_from_last_speaker():
    # 上一位发言人 1 排到末尾 → 顺序 2,3,1（保证多次点击从不同成员开始）
    sup = _idle_supervisor(replies=["闲聊"], members=_members(1, 2, 3))
    events = await _idle_events(sup, max_total=10, turn_max=2, rotate_after=1)
    assert [d["agent_id"] for e, d in events if e == "speaker"] == [2, 3, 1]


async def test_run_idle_without_anchor_uses_member_order():
    sup = _idle_supervisor(replies=["闲聊"], members=_members(1, 2, 3))
    events = await _idle_events(sup, max_total=10, turn_max=2)
    assert [d["agent_id"] for e, d in events if e == "speaker"] == [1, 2, 3]


async def test_run_idle_caps_messages_per_turn():
    # 每人最多 turn_max_messages 条（超出部分丢弃）
    sup = _idle_supervisor(replies=["a", "b", "c"], members=_members(1, 2))
    events = await _idle_events(sup, max_total=10, turn_max=2)
    turns = {d["agent_id"]: d["replies"] for e, d in events if e == "turn_done"}
    assert turns == {1: ["a", "b"], 2: ["a", "b"]}


async def test_run_idle_respects_total_budget():
    # 累计预算 3：第一位发 2 条，第二位只补 1 条即达上限 → 顺序 [1,2]
    sup = _idle_supervisor(replies=["a", "b"], members=_members(1, 2, 3))
    events = await _idle_events(sup, max_total=3, turn_max=2)
    order = [d["agent_id"] for e, d in events if e == "speaker"]
    turns = {d["agent_id"]: d["replies"] for e, d in events if e == "turn_done"}
    assert order == [1, 2]
    assert turns == {1: ["a", "b"], 2: ["a"]}


# ===== M6：思考链透传与对齐 =====


class _ThinkingGraph:
    """带思考链的图替身：values 快照同时给出 replies 与 thinkings。"""

    def __init__(self, replies: list[str], thinkings: list[str]) -> None:
        self._replies = replies
        self._thinkings = thinkings

    async def astream(self, state, *, config, stream_mode):
        yield (
            "values",
            {
                "replies": list(self._replies),
                "reply": "\n".join(self._replies),
                "thinkings": list(self._thinkings),
            },
        )


def _thinking_supervisor(*, replies: list[str], thinkings: list[str], members) -> Supervisor:
    return Supervisor(
        llm=_FixedDecisionLLM("1"),
        graph_factory=lambda: _ThinkingGraph(replies, thinkings),
        members=members,
        max_turns=4,
    )


async def test_supervisor_turn_done_carries_thinkings_aligned_to_replies():
    sup = _thinking_supervisor(
        replies=["甲", "乙"], thinkings=["想甲", "想乙"], members=_members(1)
    )
    events = [x async for x in sup.run(group_session_id=1, user_message="hi")]
    done = [d for e, d in events if e == "turn_done"]
    assert len(done) == 1
    assert done[0]["replies"] == ["甲", "乙"]
    assert done[0]["thinkings"] == ["想甲", "想乙"]


async def test_supervisor_turn_done_pads_missing_thinkings():
    # 思考条数不足 → 以消息条数为准补齐空串
    sup = _thinking_supervisor(replies=["甲", "乙"], thinkings=["只有一条"], members=_members(1))
    events = [x async for x in sup.run(group_session_id=1, user_message="hi")]
    done = [d for e, d in events if e == "turn_done"][0]
    assert done["thinkings"] == ["只有一条", ""]


async def test_run_idle_truncates_thinkings_with_kept_replies():
    # 每人最多 2 条 → 第 3 条消息及其思考一并被丢弃，思考与保留的消息对齐
    sup = _thinking_supervisor(
        replies=["a", "b", "c"], thinkings=["ta", "tb", "tc"], members=_members(1)
    )
    events = await _idle_events(sup, max_total=10, turn_max=2)
    turns = {d["agent_id"]: d for e, d in events if e == "turn_done"}
    assert turns[1]["replies"] == ["a", "b"]
    assert turns[1]["thinkings"] == ["ta", "tb"]


# ===== 群聊「说话人前缀」剥离（防剧本化串味）=====

def test_strip_speaker_prefix_removes_member_prefix():
    names = {"角色1", "角色2"}
    assert strip_speaker_prefix("【角色1】你好呀", names) == "你好呀"
    assert strip_speaker_prefix("【角色2】（笑了笑）在呢", names) == "（笑了笑）在呢"  # 仅剥前缀
    assert strip_speaker_prefix("【角色1】【角色2】正文", names) == "正文"  # 连续前缀逐个剥
    assert strip_speaker_prefix("没有前缀", names) == "没有前缀"
    assert strip_speaker_prefix("【路人】你好", names) == "【路人】你好"  # 非成员名不剥
    assert strip_speaker_prefix("【角色1】   ", names) == "【角色1】   "  # 剥空则保留原文


async def test_supervisor_strips_speaker_prefix_from_replies():
    # 模型模仿注入格式、输出带【角色名】前缀 → turn_done 前被剥离
    sup = Supervisor(
        llm=_FixedDecisionLLM("1"),
        graph_factory=lambda: _StubGraph("【角色1】你按错的时候我也看见了。"),
        members=_members(1, 2),
        max_turns=4,
    )
    events = [x async for x in sup.run(group_session_id=1, user_message="hi")]
    first = [d for e, d in events if e == "turn_done"][0]
    assert first["agent_id"] == 1
    assert first["replies"] == ["你按错的时候我也看见了。"]


async def test_run_idle_strips_speaker_prefix():
    sup = _idle_supervisor(replies=["【角色1】先说两句"], members=_members(1, 2))
    events = await _idle_events(sup, max_total=10, turn_max=2)
    turns = {d["agent_id"]: d["replies"] for e, d in events if e == "turn_done"}
    assert turns == {1: ["先说两句"], 2: ["先说两句"]}
