"""judge 校准工具（交付物 15）：读人工标注表，输出一致率 + Cohen's κ + 混淆矩阵。

用法：
    uv run python eval/calibrate.py                # 读 labels.csv + replies.jsonl，算一致率/κ
    uv run python eval/calibrate.py --collect      # 跑 50 条 scenario，落模型回复到 replies.jsonl
    uv run python eval/calibrate.py --labels PATH  # 指定标注表

未完成人工标注时：打印「待人工标注」提示并以非零退出码结束，不抛栈。
标注表列：case_id,human_score_1,human_score_2,final_label,notes
- human_score_1 / human_score_2：同一标注人按 rubric 打 1-5 分（隔 24h 各标一次）；
- final_label：pass / fail（≥3 分视作 pass）；
- judge 侧数据来自 replies.jsonl 里 --collect 时落下的 judge 输出（维度分 + verdict）。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

if sys.platform == "win32":  # 与 run_eval 一致：真库需要 SelectorEventLoop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import judge as judge_mod  # noqa: E402
from metrics import cohens_kappa  # noqa: E402
from split import split_of  # noqa: E402

CALIB_DIR = ROOT / "eval" / "calibration"
LABELS_PATH = CALIB_DIR / "labels.csv"
REPLIES_PATH = CALIB_DIR / "replies.jsonl"
LABEL_COLUMNS = ["case_id", "human_score_1", "human_score_2", "final_label", "notes"]
COLLECT_SIZE = 50


def load_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_replies(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_label(score_text: str) -> int | None:
    """1-5 分 → 1（pass，≥3）/ 0（fail）；空或非法返回 None。"""
    text = (score_text or "").strip()
    if not text:
        return None
    try:
        score = int(text)
    except ValueError:
        return None
    return 1 if score >= 3 else 0


def _final_label(text: str) -> int | None:
    v = (text or "").strip().lower()
    if v in ("pass", "1"):
        return 1
    if v in ("fail", "0"):
        return 0
    return None


def _confusion(a: list[int], b: list[int]) -> dict:
    """行=人工（gold），列=judge/第二次标注（pred），2x2。"""
    matrix = [[0, 0], [0, 0]]
    for x, y in zip(a, b):
        matrix[x][y] += 1
    return matrix


def _fmt_matrix(matrix: dict) -> str:
    return (
        f"  gold\\pred |  pass(1) |  fail(0)\n"
        f"  pass(1)   |  {matrix[1][1]:>7} |  {matrix[1][0]:>7}\n"
        f"  fail(0)   |  {matrix[0][1]:>7} |  {matrix[0][0]:>7}"
    )


def _agreement(a: list[int], b: list[int]) -> tuple[float, float]:
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    return agree, cohens_kappa(a, b)


def run_calibration(labels_path: Path, replies_path: Path) -> int:
    if not labels_path.exists():
        print(f"[待人工标注] 标注表不存在：{labels_path}")
        return 1
    rows = load_labels(labels_path)
    if not rows:
        print(f"[待人工标注] 标注表为空：{labels_path}")
        return 1

    pending = [
        r["case_id"]
        for r in rows
        if _to_label(r.get("human_score_1", "")) is None
        or _to_label(r.get("human_score_2", "")) is None
        or _final_label(r.get("final_label", "")) is None
    ]
    if pending:
        print(f"[待人工标注] 还有 {len(pending)}/{len(rows)} 条未填完（human_score_1/2、final_label）")
        print(f"  例如：{', '.join(pending[:5])}")
        return 1

    replies = {r["case_id"]: r for r in load_replies(replies_path)}
    missing = [r["case_id"] for r in rows if r["case_id"] not in replies]
    if missing:
        print(f"[错误] replies.jsonl 缺少 {len(missing)} 条回复：{missing[:5]}……")
        print("  请先运行 `uv run python eval/calibrate.py --collect` 生成模型回复。")
        return 1

    # 标注者内一致率（intra-rater）：同一人隔 24h 两次标注
    first = [_to_label(r["human_score_1"]) for r in rows]
    second = [_to_label(r["human_score_2"]) for r in rows]
    human = [_final_label(r["final_label"]) for r in rows]
    judge_labels = []
    judge_scores = []
    for r in rows:
        j = replies[r["case_id"]].get("judge", {})
        verdict = j.get("verdict")
        judge_labels.append(None if verdict is None else (1 if verdict == "pass" else 0))
        judge_scores.append(j.get("dimensions", {}).get("persona"))

    paired = [(h, j) for h, j in zip(human, judge_labels) if j is not None]
    if not paired:
        print("[错误] judge 侧无可比对的 verdict（replies.jsonl 需含 judge 输出）")
        return 1
    gold = [p[0] for p in paired]
    pred = [p[1] for p in paired]

    intra_agree, intra_kappa = _agreement(first, second)
    judge_agree, judge_kappa = _agreement(gold, pred)
    exact = sum(
        1
        for r, js in zip(rows, judge_scores)
        if js is not None and int(r["human_score_1"]) == js
    )
    n = len(rows)

    print("---- judge 校准结果 ----")
    print(f"样本数                : {n}（replies.jsonl 对齐）")
    print(f"标注者内一致率        : {intra_agree:.1%}")
    print(f"标注者内 Cohen's κ    : {intra_kappa:.3f}")
    print(f"judge 与人工一致率    : {judge_agree:.1%}")
    print(f"judge 与人工 κ        : {judge_kappa:.3f}")
    print(f"persona 维逐分完全一致: {exact / n:.1%}")
    print("混淆矩阵（人工 final_label vs judge verdict）：")
    print(_fmt_matrix(_confusion(gold, pred)))
    print("结论口径：κ 报告如上；只信 ≥1 分的差异，小于 1 分的波动不作结论。")
    return 0


async def collect_replies(labels_path: Path, replies_path: Path) -> int:
    """跑 50 条 scenario（dev 侧优先），落模型回复与 judge 输出到 replies.jsonl。"""
    import run_eval
    from app.config import get_settings
    from app.llm.factory import create_llm_provider

    settings = get_settings()
    if not settings.llm_api_key:
        print("[error] LLM_API_KEY 为空，无法采集模型回复", file=sys.stderr)
        return 2

    cases = run_eval.load_suite("scenario")
    dev = [c for c in cases if split_of(c["id"]) == "dev"]
    picked = sorted(dev or cases, key=lambda c: c["id"])[:COLLECT_SIZE]

    store, storage_label, _, pool = await run_eval._make_store(inmemory=False, force_fake_embeddings=False)
    llm = create_llm_provider(settings)
    print(f"[collect] 存储={storage_label}，采集 {len(picked)} 条 scenario 回复……")
    rows = []
    try:
        for i, case in enumerate(picked, 1):
            replies = await run_eval._run_turns(store, llm, case, "calib")
            verdict = await run_eval._judge_case(llm, case, replies)
            rows.append(
                {
                    "case_id": case["id"],
                    "name": case["name"],
                    "turns": case["turns"],
                    "replies": replies,
                    "judge": verdict,
                }
            )
            print(f"  [{i}/{len(picked)}] {case['id']} ok")
    finally:
        if pool is not None:
            await pool.close()

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    with replies_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] 已写入 {replies_path}（{len(rows)} 条）")

    if not labels_path.exists():
        with labels_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LABEL_COLUMNS)
            for row in rows:
                writer.writerow([row["case_id"], "", "", "", ""])
        print(f"[OK] 已生成标注表骨架 {labels_path}（human_score_1/2、final_label、notes 留空待人工标注）")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="judge 校准：人工标注 vs judge 的一致率与 κ")
    ap.add_argument("--labels", default=str(LABELS_PATH), help="标注表 CSV 路径")
    ap.add_argument("--replies", default=str(REPLIES_PATH), help="模型回复 JSONL 路径")
    ap.add_argument("--collect", action="store_true", help="跑 50 条 scenario 采集模型回复与 judge 输出")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    labels_path, replies_path = Path(args.labels), Path(args.replies)
    if args.collect:
        raise SystemExit(asyncio.run(collect_replies(labels_path, replies_path)))
    raise SystemExit(run_calibration(labels_path, replies_path))


if __name__ == "__main__":
    main()
