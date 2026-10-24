"""T11 · 自然度成对比较：离线统计（胜率 / tie 率）。

口径②（见 `scripts/make_preference_pairs.py` 与交接文档 §8.7 #9）：
两版回复都由模型生成，标表直读「回复一(a_text) / 回复二(b_text)」，无 `shown_first` 盲标打乱。

读 `eval/calibration/pairs.csv`（由 `scripts/make_preference_pairs.py` 生成、人来标），输出：
- 回复一胜率 = 回复一被选 / (回复一被选 + 回复二被选)；
- tie 率 = tie / 已标注总数；
- 未标注条数（chosen 为空）。

用法：
    uv run python scripts/preference_stats.py
    uv run python scripts/preference_stats.py --csv eval/calibration/pairs.csv
    uv run python scripts/preference_stats.py --out eval/report/pref.md

口径：`chosen` ∈ {A, B, tie}；A=回复一(a_text)、B=回复二(b_text)。
兼容写法：`a`/`a_text`/`first` 记为回复一胜，`b`/`b_text`/`second` 记为回复二胜，`tie`（不分大小写）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "eval" / "calibration" / "pairs.csv"
OUT_DIR = ROOT / "eval" / "report"

# chosen 兼容写法：契约记 A/B，标注人实际可能直接写列名 a_text/b_text
_FIRST_ALIASES = {"a", "a_text", "first"}
_SECOND_ALIASES = {"b", "b_text", "second"}


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[error] 文件不存在：{path}（先跑 scripts/make_preference_pairs.py）")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def compute(rows: list[dict]) -> dict:
    total = len(rows)
    annotated = first_wins = second_wins = ties = 0
    unknown: list[str] = []
    for row in rows:
        chosen = (row.get("chosen") or "").strip().lower()
        if not chosen:
            continue
        annotated += 1
        if chosen == "tie":
            ties += 1
        elif chosen in _FIRST_ALIASES:
            first_wins += 1
        elif chosen in _SECOND_ALIASES:
            second_wins += 1
        else:
            annotated -= 1
            unknown.append(row.get("pair_id", "?"))
    return {
        "total": total,
        "annotated": annotated,
        "first_wins": first_wins,
        "second_wins": second_wins,
        "ties": ties,
        "unknown": unknown,
    }


def render_markdown(stats: dict) -> str:
    total = stats["total"]
    annotated = stats["annotated"]
    decisive = stats["first_wins"] + stats["second_wins"]
    first_rate = stats["first_wins"] / decisive if decisive else None
    tie_rate = stats["ties"] / annotated if annotated else None
    lines = [
        "# T11 自然度成对比较 · 统计",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 数据：`eval/calibration/pairs.csv`",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 总对数 | {total} |",
        f"| 已标注 | {annotated} |",
        f"| 未标注 | {total - annotated} |",
        f"| 回复一(a_text) 胜 | {stats['first_wins']} |",
        f"| 回复二(b_text) 胜 | {stats['second_wins']} |",
        f"| tie | {stats['ties']} |",
        f"| 回复一胜率（排除 tie） | {_pct(first_rate)} |",
        f"| tie 率 | {_pct(tie_rate)} |",
        "",
        "> 口径②：`chosen` ∈ {A, B, tie}（兼容 `a_text`/`b_text`）；A=回复一(a_text)、B=回复二(b_text)，两版均由模型生成。",
    ]
    if total - annotated > 0:
        lines += ["", f"> **未标注 {total - annotated}/{total}**：结论不完整，先补标再解读。"]
    if stats["unknown"]:
        lines += ["", f"> **无效 chosen**（既非 A/B/tie）：{', '.join(stats['unknown'])}"]
    if tie_rate is not None and tie_rate > 0.5:
        lines += ["", f"> **tie 率 {_pct(tie_rate)} > 50%**：这批回复区分度不足（记为发现，换更难的对，不强行凑结果）。"]
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T11 成对比较统计")
    ap.add_argument("--csv", default=str(PAIRS_PATH), help="pairs.csv 路径")
    ap.add_argument("--out", default=None, help="输出 .md 路径（默认 eval/report/preference-stats-<ts>.md）")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    stats = compute(_load_rows(Path(args.csv)))
    markdown = render_markdown(stats)

    out = Path(args.out) if args.out else OUT_DIR / f"preference-stats-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
