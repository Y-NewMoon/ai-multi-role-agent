"""T11 · 自然度成对比较：生成对 + 标注表（口径②：两版均由模型生成）。

偏差说明（见交接文档 §8.7 #9 的裁定记录）：
契约 §5 T11 原定 B = 人工手写「理想回复」（未写前用 `MECHANICAL_TEXT` 占位），
并做固定 seed 的盲标打乱（`shown_first`）。经确认改为**口径②**：
- A / B 两版**都由模型对同一 case 生成**（同人设、同用户话、独立采样），不再用客服模板占位；
- 去掉 `shown_first` 盲标打乱，标注表直读「回复一(a_text) / 回复二(b_text)」。
本脚本因此与契约原文存在偏差，已记入交接文档 §8.4 / §8.7。

本脚本做的事：
- 从 `eval/suites/scenario.jsonl` 按 id 升序取 20 条；
- A = 模型回复 v1（`eval/calibration/replies.jsonl` 的 `replies[-1]`，已有）；
- B = 模型回复 v2（**本次对同一 case 重新采样生成**的另一版）；
- 写 `eval/calibration/pairs.csv`（不覆盖已填标注，除非 `--force`）。

用法：
    uv run python scripts/make_preference_pairs.py                 # 20 条，已存在则不覆盖
    uv run python scripts/make_preference_pairs.py --sample 10
    uv run python scripts/make_preference_pairs.py --force         # 覆盖（会丢已填的 chosen/note）

CSV 列：`pair_id,case_id,a_text,b_text,chosen,note`
- `a_text` = 模型回复 v1；
- `b_text` = 模型回复 v2（另一版）；
- `chosen` = 标注人填：`A`（回复一更好）/ `B`（回复二更好）/ `tie`（无法区分）；**必须允许 tie**；
- `note` = 可选一两句理由。

> v2 为模型采样结果，**非确定性**：重跑可能得到不同措辞；交付后以落盘的 `pairs.csv` 为准。
> 需真实 LLM：`LLM_API_KEY` 为空则报错退出（两版都要真生成）。
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

# Windows 下 psycopg 异步需 SelectorEventLoop（与 run_eval 一致）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SCENARIO_PATH = ROOT / "eval" / "suites" / "scenario.jsonl"
REPLIES_PATH = ROOT / "eval" / "calibration" / "replies.jsonl"
PAIRS_PATH = ROOT / "eval" / "calibration" / "pairs.csv"

DEFAULT_SAMPLE = 20

CSV_COLUMNS = ("pair_id", "case_id", "a_text", "b_text", "chosen", "note")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[error] 文件不存在：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pick(sample: int, replies_path: Path) -> tuple[list[dict], dict]:
    scenario = {c["id"]: c for c in _load_jsonl(SCENARIO_PATH)}
    replies = {r["case_id"]: r for r in _load_jsonl(replies_path)}
    ids = sorted(i for i in scenario if i in replies)[:sample]
    if not ids:
        raise SystemExit("[error] scenario 与 replies 的交集为空，无可用样本")
    return [scenario[i] for i in ids], replies


async def build_pairs(sample: int, replies_path: Path) -> list[dict]:
    """生成 20 对：A=已有 v1，B=对同一 case 重新采样生成的 v2。"""
    import run_eval
    from app.config import get_settings
    from app.llm.factory import create_llm_provider

    cases, replies = _pick(sample, replies_path)
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit("[error] LLM_API_KEY 为空，口径② 需真模型生成第二版回复")

    llm = create_llm_provider(settings)
    # 与 collect_replies 同条件：真库 + 真嵌入（每 case 结束即清记忆，故内容一致）
    store, storage_label, embed_label, pool = await run_eval._make_store(
        inmemory=False, force_fake_embeddings=False
    )
    print(f"[generate] 存储={storage_label} 嵌入={embed_label}，生成 {len(cases)} 对的 v2……")
    pairs: list[dict] = []
    try:
        for idx, case in enumerate(cases, start=1):
            v2 = (await run_eval._run_turns(store, llm, case, "pref") or [""])[-1]
            v1 = (replies[case["id"]].get("replies") or [""])[-1]
            pairs.append(
                {
                    "pair_id": f"p-{idx:03d}",
                    "case_id": case["id"],
                    "a_text": v1,
                    "b_text": v2,
                    "chosen": "",
                    "note": "",
                }
            )
            print(f"  [{idx}/{len(cases)}] {case['id']} ok")
    finally:
        if pool is not None:
            await pool.close()
    return pairs


def _write(pairs: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(pairs)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T11 自然度成对比较：生成 pairs.csv（口径②）")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help=f"取前 N 条，默认 {DEFAULT_SAMPLE}")
    ap.add_argument("--out", default=str(PAIRS_PATH), help="输出 CSV 路径")
    ap.add_argument("--replies", default=str(REPLIES_PATH), help="v1 回复来源（replies.jsonl）")
    ap.add_argument("--force", action="store_true", help="已存在时覆盖（会丢已填的 chosen/note）")
    args = ap.parse_args(argv)
    if args.sample <= 0:
        ap.error("--sample 必须为正整数")
    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"[error] {out} 已存在；避免覆盖已填标注，加 --force 才覆盖"
        )

    pairs = asyncio.run(build_pairs(args.sample, Path(args.replies)))
    _write(pairs, out)

    print("---- T11 成对比较准备（口径②：两版均由模型生成）----")
    print(f"生成对    : {len(pairs)}（{', '.join(p['case_id'] for p in pairs)}）")
    print(f"输出      : {out}")
    print()
    print("标注说明：")
    print("  1. 每条给出两版模型回复：回复一(a_text) / 回复二(b_text)；")
    print("  2. 直读更自然、更像真人的一版，填 chosen ∈ {A, B, tie}（A=回复一，B=回复二）；")
    print("     必须允许 tie（两版区分不出来就填 tie，不要强迫二选一）；note 可写一句理由；")
    print("  3. 填完跑：uv run python scripts/preference_stats.py")


if __name__ == "__main__":
    main()
