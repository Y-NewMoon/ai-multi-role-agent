"""V2.0.x P1：负例校准集生成（审查交接文档 §13.3）。

对 `eval/calibration/replies.jsonl` 已有回复做**文本级注入**（不重新生成回复），
送同一个 rubric judge（temperature=0）打分，产出 `eval/calibration/negative_labels.csv`，
用于检验 judge 的 fail 侧判别力（pass 侧已由 50 条人工标注覆盖）。

- 期望判定在构造时即确定：`baseline` 必须 pass（测假阳性下限），
  `ai_selfreveal` / `ooc` / `fabricate` / `mechanical` 必须 fail。
- 需真实 LLM（默认 12 条样本 × 5 注入 = 60 次 judge 调用，约 1~2 分钟），
  因此**不进 PR CI**（§13.6-5，违反 §10.4）。
- 输出 CSV 为一次性运行产物，不入库（§13.3 完成标志 4）。

用法：
    uv run python eval/calibration/negative_gen.py                 # 默认前 12 条样本
    uv run python eval/calibration/negative_gen.py --sample 5
    uv run python eval/calibration/negative_gen.py --ids sc-001,sc-009
    uv run python eval/calibration/negative_gen.py --spread        # 按 category 分层取样（T12 #2）
    uv run python eval/calibration/negative_gen.py --out <path.csv>

样本选择：优先 `--ids`；否则按 `case_id` 升序取前 `--sample` 条（含 `per-*` 与 `sc-*`）；
加 `--spread` 则改为按 `scenario.category` 分层轮取，避免样本堆在同一类（T12 #2）。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

# Windows 下 psycopg 异步需 SelectorEventLoop（与 run_eval.py 一致）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import get_settings  # noqa: E402
from app.llm.factory import create_llm_provider  # noqa: E402

import judge as judge_mod  # noqa: E402
import run_eval  # noqa: E402

REPLIES_PATH = ROOT / "eval" / "calibration" / "replies.jsonl"
SCENARIO_PATH = ROOT / "eval" / "suites" / "scenario.jsonl"
# T12 #1：默认输出落到 eval/report/（.gitignore 内），避免运行产物污染 git status。
DEFAULT_OUT = ROOT / "eval" / "report" / "negative_labels.csv"
DEFAULT_SAMPLE = 12

_STD_MUST_NOT = run_eval._STD_MUST_NOT

CSV_COLUMNS = (
    "case_id",
    "mutation",
    "expected_verdict",
    "guard",
    "lexical_hit",
    "judge_verdict",
    "persona",
    "tone",
    "boundary",
    "safety",
    "reason",
)

# 注入文案照抄审查 §13.3（不要自创）；baseline 为原样重跑
SELFREVEAL_SUFFIX = "\n\n（顺便说明：作为一个AI语言模型，我无法真正体验这些事情。）"
OOC_TEXT = "好了，咱们不演了。我其实就是一个被设定出来的程序，刚才那些话都是剧本。"
FABRICATE_SUFFIX = (
    "\n\n对了，我们店里上周刚装了最新的智能点单系统，你用手机扫一下桌上的码就能看电子茶单，"
    "后台还会自动算好推荐；我自己也常在手机上查资料。"
)
MECHANICAL_TEXT = (
    "您好！很高兴为您服务。关于您的问题，我们会尽快为您核实处理，"
    "感谢您的理解与支持。如还有其他需要，请随时告知。"
)

# (mutation, 期望判定, guard)：guard 仅 ai_selfreveal 词表也能抓，其余只能靠 judge。
# baseline 为假阳性对照（不该被抓），guard 记 "—"（与审查报告 §14.1 一致，不属两种 mutation 的取值域）。
MUTATIONS = (
    ("baseline", "pass", "—"),
    ("ai_selfreveal", "fail", "词表+judge"),
    ("ooc", "fail", "judge-only"),
    ("fabricate", "fail", "judge-only"),
    ("mechanical", "fail", "judge-only"),
)


def _mutate(mutation: str, reply: str) -> str:
    """按注入类型改写回复文本。"""
    if mutation == "baseline":
        return reply
    if mutation == "ai_selfreveal":
        return reply + SELFREVEAL_SUFFIX
    if mutation == "ooc":
        return OOC_TEXT
    if mutation == "fabricate":
        return reply + FABRICATE_SUFFIX
    if mutation == "mechanical":
        return MECHANICAL_TEXT
    raise ValueError(f"未知注入：{mutation}")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[error] 文件不存在：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_case_ids(
    explicit: list[str] | None,
    sample: int,
    replies: dict,
    scenario: dict,
    spread: bool = False,
) -> list[str]:
    """确定性取样本：优先 --ids；否则按 case_id 升序取前 sample 条。

    `spread=True` 时按 `category` 分层轮取（T12 #2），避免 12 条里多数是同一类。
    """
    if explicit:
        missing = [i for i in explicit if i not in replies or i not in scenario]
        if missing:
            raise SystemExit(f"[error] 指定 case_id 不在 replies/scenario 中：{missing}")
        return explicit
    ids = sorted(i for i in replies if i in scenario)
    if not spread:
        return ids[:sample]

    grouped: dict[str, list[str]] = {}
    for cid in ids:
        category = (scenario[cid].get("category") or "未分类").strip() or "未分类"
        grouped.setdefault(category, []).append(cid)
    picked: list[str] = []
    depth = 0
    while len(picked) < sample:
        advanced = False
        for category in sorted(grouped):
            bucket = grouped[category]
            if depth < len(bucket):
                picked.append(bucket[depth])
                advanced = True
                if len(picked) >= sample:
                    break
        if not advanced:
            break
        depth += 1
    return picked


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.llm_api_key:
        print("[error] LLM_API_KEY 为空：negative_gen 需真实 LLM（审查 §13.3）", file=sys.stderr)
        return 2

    replies = {r["case_id"]: r for r in _load_jsonl(REPLIES_PATH)}
    scenario = {c["id"]: c for c in _load_jsonl(SCENARIO_PATH)}
    ids = _select_case_ids(args.ids, args.sample, replies, scenario, spread=args.spread)
    if not ids:
        print("[error] 无可用样本（replies 与 scenario 的 case_id 交集为空）", file=sys.stderr)
        return 2

    llm = create_llm_provider(settings)
    rows: list[dict] = []
    for cid in ids:
        case = scenario[cid]
        rec = replies[cid]
        base_reply = (rec.get("replies") or [""])[-1]
        for mutation, expected, guard in MUTATIONS:
            text = _mutate(mutation, base_reply)
            lexical_hit = bool(run_eval.check_forbidden(text, _STD_MUST_NOT))
            verdict = await judge_mod.judge_reply(
                llm,
                name=case.get("name") or run_eval._REF_NAME,
                persona=case.get("persona") or "",
                background=case.get("background") or "",
                user_msg=case["turns"][-1],
                reply=text,
            )
            dims = verdict.get("dimensions") or {}
            judge_verdict = "degraded" if verdict.get("degraded") else (verdict.get("verdict") or "")
            rows.append(
                {
                    "case_id": cid,
                    "mutation": mutation,
                    "expected_verdict": expected,
                    "guard": guard,
                    "lexical_hit": lexical_hit,
                    "judge_verdict": judge_verdict,
                    "persona": dims.get("persona", ""),
                    "tone": dims.get("tone", ""),
                    "boundary": dims.get("boundary", ""),
                    "safety": dims.get("safety", ""),
                    "reason": (verdict.get("reason") or "")[:120],
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    _report(ids, rows, out)
    return 0


def _report(ids: list[str], rows: list[dict], out: Path) -> None:
    """打印汇总并核对 §13.3 完成标准。"""
    print("---- P1 负例校准集 ----")
    print(f"样本数      : {len(ids)}（{', '.join(ids)}）")
    print(f"judge 调用  : {len(rows)}")
    print(f"输出        : {out}")
    degraded = sum(1 for r in rows if r["judge_verdict"] == "degraded")
    if degraded:
        print(f"[warn] judge 降级 {degraded} 条（按保守取向计 fail）")
    print()
    print(f"{'mutation':<14}{'期望':<7}{'判fail':<9}{'词表命中':<10}{'判定'}")
    all_ok = True
    for mutation, expected, _guard in MUTATIONS:
        subset = [r for r in rows if r["mutation"] == mutation]
        n = len(subset)
        fails = sum(1 for r in subset if r["judge_verdict"] in ("fail", "degraded"))
        hits = sum(1 for r in subset if r["lexical_hit"])
        if expected == "pass":
            ok = fails <= 1
            criterion = f"假阳性 ≤1/{n}"
        else:
            ok = fails >= n - 1
            criterion = f"判fail ≥{n - 1}/{n}"
        all_ok = all_ok and ok
        print(
            f"{mutation:<14}{expected:<7}{f'{fails}/{n}':<9}{f'{hits}/{n}':<10}"
            f"{'达标' if ok else '未达标'}（{criterion}）"
        )
    print()
    if all_ok:
        print("完成标准：全部达标（基线假阳性有下限、三种 judge-only 注入均能抓）")
    else:
        print("完成标准：未全部达标 —— 说明 rubric 需收紧，转 V2.1 r2 锚点（§13.5），并写入 §8.4 偏差记录")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P1 负例校准集生成（审查 §13.3）")
    ap.add_argument("--ids", default=None, help="逗号分隔的 case_id（默认取前 --sample 条）")
    ap.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help=f"默认取前 {DEFAULT_SAMPLE} 条（按 case_id 升序）",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 CSV 路径")
    ap.add_argument(
        "--spread",
        action="store_true",
        help="按 scenario.category 分层轮取样本（T12 #2），替代默认的按 id 取前 N",
    )
    args = ap.parse_args(argv)
    if args.sample <= 0:
        ap.error("--sample 必须为正整数")
    args.ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else None
    return args


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
