"""golden.memory_hit_rate 的空分母口径（V2.5 收尾修复）。

`--sample 4` 可能抽不到任何 memory 用例；此时分母为 0，指标应为 null（未测），
不能写成 0.0（会被误读为“全部未命中”）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import metrics  # noqa: E402


def test_未抽到memory用例返回None():
    # 对应 run_golden 中 mem_hits=[] 的情况（例如 golden --sample 4 只抽到 forbid/isolation）
    assert metrics.pass_rate([]) is None


def test_两条memory用例全命中返回1():
    assert metrics.pass_rate([True, True]) == 1.0
