"""rubric r2 回归校验（契约 §2.11）：用 `eval/calibration/replies.jsonl` 里的**原回复**
重跑 judge（不重新生成回复、不改 replies.jsonl），产出三条回归证据：

1. `sc-057`：r1 persona=2（人工 4/4）→ r2 后差距缩小或一致；
2. `sc-039`：两轮 persona 5→3 → r2 后落在 3–4 且不跨两档；
3. 50 条原回复判 fail 不超过 2 条（r1 为 1 条）。

只读 `replies.jsonl` 与 `eval/suites/scenario.jsonl`；输出写 `eval/report/r2-regression-*.{md,json}`。

用法：
    uv run python eval/r2_regression.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import judge as judge_mod  # noqa: E402
import run_eval  # noqa: E402

REPLIES_PATH = ROOT / "eval" / "calibration" / "replies.jsonl"
OUT_DIR = ROOT / "eval" / "report"
WATCH_IDS = ("sc-057", "sc-039")
# r1 的 persona 锚点原文（仅用于复算「r1 逐轮分」这一对照，不参与 r2 判定）
R1_PERSONA_ANCHOR = "1=通用助手口吻；3=大体符合，偶有出戏；5=称呼/语气/习惯用词全符合"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)
    )
    return out.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(ROOT)
    )
    return bool(out.stdout.strip())


def load_replies(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_meta(case: dict) -> dict:
    return {
        "name": case.get("name") or run_eval._REF_NAME,  # noqa: SLF001
        "persona": case.get("persona") or run_eval._REF_PERSONA,  # noqa: SLF001
        "background": case.get("background") or run_eval._REF_BACKGROUND,  # noqa: SLF001
    }


async def _judge_last_turn(llm, case: dict, replies: list[str]) -> dict:
    meta = _case_meta(case)
    return await judge_mod.judge_reply(
        llm,
        name=meta["name"],
        persona=meta["persona"],
        background=meta["background"],
        user_msg=case["turns"][-1],
        reply=replies[-1] if replies else "",
    )


async def _judge_each_turn(llm, case: dict, replies: list[str]) -> list[dict]:
    meta = _case_meta(case)
    out: list[dict] = []
    for turn, reply in zip(case["turns"], replies):
        out.append(
            await judge_mod.judge_reply(
                llm,
                name=meta["name"],
                persona=meta["persona"],
                background=meta["background"],
                user_msg=turn,
                reply=reply,
            )
        )
    return out


async def _judge_each_turn_r1(llm, case: dict, replies: list[str]) -> list[dict]:
    """按 r1 原 persona 锚点复算逐轮分（只换 persona 锚点，其余维度锚点同 r2）。

    用于核对契约 §2.11 里「sc-039 两轮 5→3」的 r1 参考值：replies.jsonl 只存了
    末轮 r1 判定，逐轮 r1 分需现算。此函数只读锚点、不落盘。
    """
    base = judge_mod._RUBRIC_ANCHORS["persona"]
    judge_mod._RUBRIC_ANCHORS["persona"] = R1_PERSONA_ANCHOR
    try:
        return await _judge_each_turn(llm, case, replies)
    finally:
        judge_mod._RUBRIC_ANCHORS["persona"] = base


async def _run(args: argparse.Namespace) -> int:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    from app.config import get_settings
    from app.llm.factory import create_llm_provider

    settings = get_settings()
    if not settings.llm_api_key:
        print("[skip] 无 LLM_API_KEY，r2 回归无法执行（退出码 0，不伪造数字）")
        return 0

    rows = load_replies(REPLIES_PATH)
    cases = {c["id"]: c for c in run_eval.load_suite("scenario")}
    llm = create_llm_provider(settings)

    # ---- 回归 3：50 条原回复判 fail 数（r1 口径取 replies.jsonl 里已存的 judge 输出）----
    r1_fail: list[str] = []
    r2_fail: list[str] = []
    degraded = 0
    per_case: list[dict] = []
    for row in rows:
        case = cases.get(row["case_id"])
        if case is None:
            continue
        combined = "\n".join(row.get("replies") or [])
        must_not = list(case.get("expect", {}).get("must_not", [])) + run_eval._STD_MUST_NOT  # noqa: SLF001
        hits = run_eval.check_forbidden(combined, must_not)
        stored = (row.get("judge") or {}).get("verdict")
        stored_persona = ((row.get("judge") or {}).get("dimensions") or {}).get("persona")
        if stored != "pass":
            r1_fail.append(row["case_id"])
        verdict = await _judge_last_turn(llm, case, row.get("replies") or [])
        if verdict.get("degraded"):
            degraded += 1
        persona = (verdict.get("dimensions") or {}).get("persona")
        passed = (not hits) and verdict.get("verdict") == "pass"
        if not passed:
            r2_fail.append(row["case_id"])
        per_case.append(
            {
                "id": row["case_id"],
                "hits": hits,
                "r1_verdict": stored,
                "r1_persona": stored_persona,
                "r2_verdict": verdict.get("verdict"),
                "r2_persona": persona,
                "r2_dimensions": verdict.get("dimensions"),
                "r2_degraded": bool(verdict.get("degraded")),
                "r2_reason": verdict.get("reason"),
            }
        )
        print(
            f"[50] {row['case_id']} r1={stored} r2={verdict.get('verdict')} persona={persona}",
            flush=True,
        )

    # ---- 回归 1/2：sc-057 / sc-039 逐轮 persona（r1 锚点复算作对照）----
    watch: dict[str, list[dict]] = {}
    for cid in WATCH_IDS:
        row = next((r for r in rows if r["case_id"] == cid), None)
        case = cases.get(cid)
        if row is None or case is None:
            continue
        verdicts = await _judge_each_turn(llm, case, row.get("replies") or [])
        r1_verdicts = await _judge_each_turn_r1(llm, case, row.get("replies") or [])
        watch[cid] = [
            {
                "turn": case["turns"][i],
                "persona": (v.get("dimensions") or {}).get("persona"),
                "tone": (v.get("dimensions") or {}).get("tone"),
                "verdict": v.get("verdict"),
                "r1_persona": (r1_verdicts[i].get("dimensions") or {}).get("persona"),
                "r1_verdict": r1_verdicts[i].get("verdict"),
                "degraded": bool(v.get("degraded")),
            }
            for i, v in enumerate(verdicts)
        ]
        print(
            f"[watch] {cid} r2 persona={[w['persona'] for w in watch[cid]]} "
            f"r1 persona={[w['r1_persona'] for w in watch[cid]]}",
            flush=True,
        )

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    sc057 = [w["persona"] for w in watch.get("sc-057", [])]
    sc039 = [w["persona"] for w in watch.get("sc-039", [])]
    sc057_r1 = [w["r1_persona"] for w in watch.get("sc-057", [])]
    sc039_r1 = [w["r1_persona"] for w in watch.get("sc-039", [])]
    # 回归 1：sc-057 末轮 persona 与人工 4 的差距，r2 不得大于 r1（契约 §2.11「差距缩小或一致」）。
    # r1 末轮值取 replies.jsonl 里已有的 r1 完整 rubric 判定（sc-057 = 2）；逐轮 r1 分用 r1 persona
    # 锚点现算仅作对照参考，不作为判据基准（其余维度锚点已是 r2，复算值并非完整 r1 结果）。
    sc057_last = sc057[-1] if sc057 else None
    sc057_gap_r2 = None if sc057_last is None else abs(sc057_last - 4)
    sc057_r1_stored = next(
        (c["r1_persona"] for c in per_case if c["id"] == "sc-057" and c.get("r1_persona") is not None),
        None,
    )
    sc057_r1_last = sc057_r1_stored if sc057_r1_stored is not None else 2
    sc057_gap_r1 = abs(sc057_r1_last - 4)
    check_057 = sc057_gap_r2 is not None and sc057_gap_r2 <= sc057_gap_r1
    # 回归 2：sc-039 不再跨两档（主判据 span<=1）；是否落在 3–4 单列记录（不作硬判据，见 §5）
    sc039_span = (max(sc039) - min(sc039)) if sc039 and None not in sc039 else None
    sc039_in_3_4 = bool(sc039) and all(p is not None and 3 <= p <= 4 for p in sc039)
    check_039 = sc039_span is not None and sc039_span <= 1
    sc039_r1_stored = next(
        (c["r1_persona"] for c in per_case if c["id"] == "sc-039" and c.get("r1_persona") is not None),
        None,
    )
    result = {
        "rubric_version": judge_mod.RUBRIC_VERSION,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "started_at": started,
        "finished_at": finished,
        "check1_sc057": {
            "personas": sc057,
            "r1_personas_recomputed": sc057_r1,
            "r2_last_gap_to_human": sc057_gap_r2,
            "r1_last_persona": sc057_r1_last,
            "r1_last_persona_source": "replies.jsonl 里的 r1 完整 rubric 判定（r1 锚点复算值仅作对照）",
            "r1_last_gap_to_human": sc057_gap_r1,
            "human": 4,
            "pass": check_057,
        },
        "check2_sc039": {
            "contract_expect": "r1 两轮 5→3 → r2 后落在 3–4，不再跨两档",
            "expected_met": sc039_in_3_4,
            "aux_pass": check_039,
            "personas": sc039,
            "r1_personas_recomputed": sc039_r1,
            "span": sc039_span,
            "in_3_4": sc039_in_3_4,
            "r1_ref_in_doc": "5->3",
            "pass": check_039,
        },
        "check3_fail_count": {
            "r1_fail": len(r1_fail),
            "r1_ids": r1_fail,
            "r2_fail": len(r2_fail),
            "r2_ids": r2_fail,
            "pass": len(r2_fail) <= 2,
            "degraded": degraded,
            "n": len(per_case),
        },
        "per_case": per_case,
        "watch": watch,
        "inputs": {"replies_sha256": _sha256_file(REPLIES_PATH)},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"r2-regression-{run_id}.json"
    md_path = OUT_DIR / f"r2-regression-{run_id}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# rubric r2 回归校验（§2.11）",
                "",
                f"- 时间：{started} ~ {finished}；git `{result['git_sha']}`（dirty={result['git_dirty']}）",
                f"- RUBRIC_VERSION=`{result['rubric_version']}`；原回复来自 `eval/calibration/replies.jsonl`"
                f"（sha256={result['inputs']['replies_sha256'][:12]}…，本脚本只读不写）",
                f"- 「r1 逐轮分」用 r1 原 persona 锚点现算（replies.jsonl 只存末轮 r1 判定）：",
                f"  sc-057={sc057_r1}，sc-039={sc039_r1}",
                "",
                "| 回归项 | 契约期望 | 实测 | 结论 |",
                "|---|---|---|---|",
                f"| 1 `sc-057` | r1 末轮 persona=2（人工 4/4）→ r2 后差距缩小或一致 | "
                f"r2 末轮 persona={sc057_last}（gap={sc057_gap_r2}）；r2 逐轮={sc057}；"
                f"对比：r1 锚点复算逐轮={sc057_r1} | "
                f"{'通过' if check_057 else '不通过'}"
                f"（r2 gap={sc057_gap_r2} ≤ r1 gap={sc057_gap_r1}） |",
                f"| 2 `sc-039` | r1 两轮 5→3 → r2 后落在 3–4，不再跨两档 | "
                f"r2 逐轮 persona={sc039}（span={sc039_span}，in_3_4={sc039_in_3_4}）；"
                f"r1 锚点复算={sc039_r1} | "
                f"{'原期望达成' if sc039_in_3_4 else '契约前提不成立 / 原期望未达成（in_3_4=false）'}；"
                f"辅助判据 span≤1 {'成立' if check_039 else '不成立'} |",
                f"| 3 50 条原回复判 fail ≤ 2 | ≤ 2 | r2 fail={len(r2_fail)}（r1={len(r1_fail)}）；"
                f"degraded={degraded} | {'通过' if len(r2_fail) <= 2 else '不通过'} |",
                "",
                f"- r2 fail 用例：{r2_fail or '无'}",
                "",
                f"> 回归 2 说明：契约写「r1 两轮 5→3」，但用 r1 原 persona 锚点重判该用例实得 "
                f"{sc039_r1}（`replies.jsonl` 存的 r1 末轮判定为 persona={sc039_r1_stored}），"
                f"即该前提在本仓库数据上不成立；r2 实测 {sc039} "
                f"{'落在 3–4' if sc039_in_3_4 else '未降档（in_3_4=false）'}。"
                f"**未改 rubric 锚点凑预期**；辅助判据 span≤1 "
                f"{'成立' if check_039 else '不成立'}，原期望是否达成本行如实标注"
                f"（见交接文档 §5 #8、§7）。",
                "",
                "> 命令：`uv run python eval/r2_regression.py`；逐条明细见同名 `.json`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("\n---- r2 回归（§2.11）----")
    print(f"sc-057 r2 persona={sc057}（末轮 gap={sc057_gap_r2}）-> {'通过' if check_057 else '不通过'}")
    print(
        f"sc-039 r2 persona={sc039}（span={sc039_span}，in_3_4={sc039_in_3_4}）-> "
        f"{'原期望达成' if sc039_in_3_4 else '契约前提不成立 / 原期望未达成'}；"
        f"辅助 span≤1 {'成立' if check_039 else '不成立'}"
    )
    print(f"50 条 fail：r1={len(r1_fail)} r2={len(r2_fail)}（degraded={degraded}）-> "
          f"{'通过' if len(r2_fail) <= 2 else '不通过'}")
    print(f"报告：{md_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="rubric r2 回归校验（三条）")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
