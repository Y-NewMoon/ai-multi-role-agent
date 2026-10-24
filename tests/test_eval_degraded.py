"""P0 降级判定纯函数（审查交接文档 §13.2）：judge 解析失败后的保守计分。

judge 坏掉不能产出绿色：无论是否命中禁词，该用例都必须计入分母且判 fail，
不许再出现"解析失败即丢样本"导致分母静默缩小（记忆 §13.2 反向验证要求）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import run_eval  # noqa: E402

_STD = ["作为一个AI", "语言模型", "人工智能"]


def test_降级_命中禁词记fail():
    ok, reason = run_eval._verdict_from_degraded("作为一个AI，我无法回答这个问题。", _STD)
    assert ok is False
    assert reason.startswith("[词表兜底]")


def test_降级_未命中也记fail():
    ok, reason = run_eval._verdict_from_degraded("客人稍坐，我去沏壶桂花乌龙。", _STD)
    assert ok is False
    assert reason.startswith("[judge 降级-未判定]")


async def test_run_scenario_降级不产生假绿(monkeypatch):
    """反向验证（§13.2 完成标志 5）：judge 全坏时必须判 fail，persona_pass 下降而非维持 1.0。"""
    cases = [
        {
            "id": "sc-999",
            "name": "四季夏目",
            "persona": "茶艺师",
            "background": "古镇茶馆",
            "turns": ["你好呀"],
            "expect": {"must_not": []},
        }
    ]

    async def fake_turns(store, llm, case, run_id):
        return ["客人好呀，先喝口茶。"]

    async def fake_judge(llm, case, replies):
        return {
            "dimensions": {},
            "verdict": None,
            "confidence": 0.0,
            "reason": "judge 输出三次解析失败",
            "judge_parse_error": True,
            "degraded": True,
        }

    monkeypatch.setattr(run_eval, "_run_turns", fake_turns)
    monkeypatch.setattr(run_eval, "_judge_case", fake_judge)
    result = await run_eval.run_scenario(None, None, cases, "t")
    assert result["degraded"] == 1
    assert result["metrics"]["persona_pass"] == 0.0
    assert result["failures"][0]["reason"].startswith("[judge 降级-未判定]")
