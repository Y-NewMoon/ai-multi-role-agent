"""T10（可选）· judge 负例校准集：labels_fail.csv + replies_fail.jsonl。

背景（V2.0 复审建议项 6）：`labels.csv` 的 50 条样本**全为 pass**，fail 侧零方差，
导致 judge 与人工的 κ 无解释力（患病率悖论）。本脚本补 fail 侧样本，让 `eval/calibrate.py`
产出**非退化**的 2×2 混淆矩阵与有值的 κ。

做法：
- 从 `scenario` 里取 20 条**未被 labels.csv 标注**的样本（与 pass 侧不重号），
  对每条回复做文本级负例注入（出戏/编造/生硬/自曝），送同一 rubric judge 打分；
- 生成 `eval/calibration/replies_fail.jsonl`：
    50 条既有（`replies.jsonl` 原样）+ 20 条负例（含 judge 输出）；
- 生成 `eval/calibration/labels_fail.csv`：
    case_id,human_score_1,human_score_2,final_label,notes
    50 条 pass（沿用 `labels.csv` 的人工标注）+ 20 条 fail（构造负例，score=1、final_label=fail）。

**口径说明（写进 §8.7）**：fail 侧 20 条是**构造**负例（非人工逐条标注）；合并成 70 行是为了让
`calibrate.py` 的混淆矩阵 pass/fail 两侧都非空（否则 κ 退化）。若要求 labels_fail.csv 只放 20 条 fail 行，
需要确认口径。

需真实 LLM（20 次 judge 调用）；产物为运行产物，**不入库前先确认**（默认写 eval/calibration/）。

用法：
    uv run python eval/calibration/make_fail_labels.py
    uv run python eval/calibration/make_fail_labels.py --sample 20
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
sys.path.insert(0, str(ROOT / "eval" / "calibration"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import run_eval  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.factory import create_llm_provider  # noqa: E402

# 复用 negative_gen 的注入文案与改写逻辑（单一定义来源）
import negative_gen  # noqa: E402

CALIB_DIR = ROOT / "eval" / "calibration"
LABELS_PATH = CALIB_DIR / "labels.csv"
REPLIES_PATH = CALIB_DIR / "replies.jsonl"
OUT_LABELS = CALIB_DIR / "labels_fail.csv"
OUT_REPLIES = CALIB_DIR / "replies_fail.jsonl"

LABEL_COLUMNS = ["case_id", "human_score_1", "human_score_2", "final_label", "notes"]
# 负例注入类型（取 negative_gen 的非 baseline 项）
NEG_MUTATIONS = tuple(m for m, expected, _g in negative_gen.MUTATIONS if expected == "fail")
DEFAULT_SAMPLE = 20


def _load_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _pick_fail_ids(sample: int, labels: list[dict]) -> list[str]:
    labelled = {r["case_id"] for r in labels}
    cases = run_eval.load_suite("scenario")
    rest = sorted(c["id"] for c in cases if c["id"] not in labelled)
    if len(rest) < sample:
        raise SystemExit(f"[error] 未标注样本不足：need {sample}, got {len(rest)}")
    return rest[:sample]


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.llm_api_key:
        print("[error] LLM_API_KEY 为空：T10 需真实 LLM", file=sys.stderr)
        return 2

    labels = _load_labels(LABELS_PATH)
    base_replies = {r["case_id"]: r for r in _load_replies(REPLIES_PATH)}
    cases = {c["id"]: c for c in run_eval.load_suite("scenario")}
    fail_ids = _pick_fail_ids(args.sample, labels)

    llm = create_llm_provider(settings)
    # 为 20 条未标注样本**真实生成**回复（非采集侧），再对末轮回复做负例注入：
    # 这样负例来自真实模型文本，符合"出戏/答非所问/违背人设"的构造意图。
    store, storage_label, _, pool = await run_eval._make_store(
        inmemory=True, force_fake_embeddings=True
    )
    new_rows: list[dict] = []
    try:
        for i, cid in enumerate(fail_ids):
            case = cases[cid]
            try:
                replies = await run_eval._run_turns(store, llm, case, "failcalib")
            except Exception as exc:  # noqa: BLE001 - 单条失败跳过，不中断
                print(f"  [{i + 1}/{len(fail_ids)}] {cid} 生成失败，跳过：{exc}")
                continue
            mutation = NEG_MUTATIONS[i % len(NEG_MUTATIONS)]
            mutated = negative_gen._mutate(mutation, replies[-1] if replies else "")
            verdict = await run_eval._judge_case(llm, case, [mutated])
            new_rows.append(
                {
                    "case_id": cid,
                    "name": case.get("name") or run_eval._REF_NAME,
                    "turns": case["turns"],
                    "replies": [mutated],
                    "judge": verdict,
                    "mutation": mutation,
                }
            )
            print(f"  [{i + 1}/{len(fail_ids)}] {cid} ({mutation}) 判定={verdict.get('verdict')}")
    finally:
        if pool is not None:
            await pool.close()
    print(f"[info] 生成存储：{storage_label}")

    # replies_fail.jsonl = 既有 50 条 + 20 条负例
    out_replies = list(base_replies.values()) + [
        {k: v for k, v in row.items() if k != "mutation"} for row in new_rows
    ]
    OUT_REPLIES.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_replies) + "\n",
        encoding="utf-8",
    )

    # labels_fail.csv = 50 条 pass（沿用）+ 20 条 fail（构造）
    out_labels = list(labels) + [
        {
            "case_id": row["case_id"],
            "human_score_1": "1",
            "human_score_2": "1",
            "final_label": "fail",
            "notes": f"构造负例（{row['mutation']}），期望 judge 判 fail",
        }
        for row in new_rows
    ]
    with OUT_LABELS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(out_labels)

    print(f"[OK] {OUT_REPLIES}（{len(out_replies)} 条）")
    print(f"[OK] {OUT_LABELS}（{len(out_labels)} 行：{len(labels)} pass + {len(new_rows)} fail）")
    print("下一步：uv run python eval/calibrate.py --labels eval/calibration/labels_fail.csv "
          "--replies eval/calibration/replies_fail.jsonl")
    return 0


def _load_replies(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T10 负例校准集生成")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help=f"负例条数，默认 {DEFAULT_SAMPLE}")
    return ap.parse_args(argv)


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
