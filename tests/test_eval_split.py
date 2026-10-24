"""切分确定性与比例（契约 4.4）：同一 id 永远同一侧，比例落在 60:40 ~ 80:20。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from split import SEED, split_of  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def _all_case_ids() -> list[str]:
    ids: list[str] = []
    for path in sorted((EVAL_DIR / "suites").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.append(__import__("json").loads(line)["id"])
    return ids


def test_split_种子固定():
    assert SEED == 20261015


# 钉死的分侧（跨机器/跨版本必须一致）：改这个映射 = 改测试集切分，等于破坏可比性。
_EXPECTED_SIDES = {
    "iso-001": "dev",
    "mem-002": "test",
    "sc-007": "dev",
    "adv-013": "dev",
    "rl-020": "test",
}


def test_split_确定性():
    """硬编码分侧（T12 #5）：不再用恒真断言 split_of(x)==split_of(x)。"""
    for cid, side in _EXPECTED_SIDES.items():
        assert split_of(cid) == side, f"{cid} 的分侧变了：期望 {side}，实际 {split_of(cid)}"


def test_split_不同id可不同侧():
    sides = {split_of(f"demo-{i}") for i in range(50)}
    assert sides == {"dev", "test"}


def test_split_全量用例比例():
    ids = _all_case_ids()
    assert len(ids) >= 220, f"套件用例不足 220 条：{len(ids)}"
    dev = sum(1 for cid in ids if split_of(cid) == "dev")
    ratio = dev / len(ids)
    assert 0.60 <= ratio <= 0.80, f"dev 比例越界：{ratio:.3f}（dev={dev}/{len(ids)}）"
