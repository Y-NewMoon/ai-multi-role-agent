"""T1：token 估算与成本公式确定性单测。"""

from __future__ import annotations

import pytest

from app.observability.cost import cost_cny
from app.observability.tokens import estimate_messages_tokens, estimate_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("你好", 2),
        ("abcd", 1),
        ("abcde", 2),
        ("你好abcd", 3),
        ("hello world", 3),  # 11 个非 CJK -> ceil(11/4)=3
    ],
)
def test_estimate_tokens(text, expected):
    assert estimate_tokens(text) == expected


def test_estimate_messages_tokens_supports_str_and_parts():
    messages = [
        {"role": "system", "content": "你好"},  # 2 + 4
        {"role": "user", "content": [{"type": "text", "text": "abcd"}]},  # 1 + 4
        {"role": "assistant", "content": None},  # 0 + 4
    ]
    assert estimate_messages_tokens(messages) == 15


def test_cost_formula_exact():
    got = cost_cny(
        prompt_tokens=3100,
        completion_tokens=210,
        input_price_per_1m=2.0,
        output_price_per_1m=8.0,
    )
    assert got == pytest.approx(0.00788, abs=1e-9)


def test_cost_is_deterministic():
    kwargs = dict(
        prompt_tokens=12345,
        completion_tokens=678,
        input_price_per_1m=2.0,
        output_price_per_1m=8.0,
    )
    first = cost_cny(**kwargs)
    for _ in range(10):
        assert cost_cny(**kwargs) == first
    assert first == round(first, 6)


def test_cost_sums_match_report_total():
    parts = [
        cost_cny(
            prompt_tokens=p,
            completion_tokens=c,
            input_price_per_1m=2.0,
            output_price_per_1m=8.0,
        )
        for p, c in [(1000, 10), (2000, 20), (500, 5)]
    ]
    assert round(sum(parts), 6) == pytest.approx(0.00728, abs=1e-9)
