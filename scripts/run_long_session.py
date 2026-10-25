"""V2.3 长会话曲线（契约 §2.11 / C4）：100 轮合成历史，只调 `compose_prompt`，0 次 LLM。

- 用合成历史（脚本内生成 100 轮固定对话）直接调 `compose_prompt`，输出第 10/20/50/100 轮
  的 `prompt_token`（`_estimate_tokens`）与各段占比；
- 用桩模拟 L2 与 facts：按 `memory_l2_every_merges` 的节奏放入一段固定 300 字桩文本作为 L2、
  第 3 轮写入 2 条桩 facts，使曲线包含这三段的真实体积；`usage.llm_calls` 恒为 0；
- 第 100 轮必须仍能检出第 3 轮写入的 facts 与 L2 段；
- 第 15 轮检查点：第 1 轮关键内容必须仍能到达模型（摘要或近轮窗口）；
- 两种模式：`--engine v22`（等价 `memory_extract_enabled=false`，摘要触发 20 轮）/
  `--engine v23`（当前默认行为，摘要触发取配置）；
- `--extra-src DIR`：把旧版 `src` 插到 `sys.path` 最前，用于交叉校验 v22 等价性（硬要求）。

用法：
    uv run python scripts/run_long_session.py --engine v23 --out .tmp/long-v23.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "report"

# 第 1 轮的关键内容（15 轮检查点用）与桩文本标记
ROUND1_TEXT = "第1轮：我小时候在西湖边长大，最喜欢雨天。"
ROUND1_MARKER = "西湖边"
L2_MARKER = "关系档案桩"
FACT_TEXTS = ("用户养了猫=叫团子", "用户喜欢的茶=普洱")
CHECKPOINTS = (10, 20, 50, 100)
CHECKPOINT_15 = 15

_SECTION_HEADERS = (
    ("time", "【当前时间】"),
    ("lore", "【设定资料】"),
    ("profile", "【客人档案】"),
    ("summary", "【本会话摘要】"),
    ("memories", "【你记得的相关信息】"),
    ("facts", "【你记得的事】"),
    ("l2", "【你和这位客人的过往】"),
    ("tools", "【可用工具】"),
)

BASE_PROMPT = (
    "你是古镇茶馆「一盏春」的茶艺师，说话温柔、慢条斯理，喜欢用茶来打比方，"
    "称呼用户为「客人」，遇到不懂的事会坦率承认。"
)


def _l2_stub(max_chars: int = 300) -> str:
    text = (
        "（关系档案桩）我们认识很久了：客人常来茶馆，喜欢靠窗的位置。"
        "我们在雨天聊过西湖，也聊过猫和旧书；客人说过怕打雷，我答应留一盏灯。"
    )
    while len(text) < max_chars:
        text += "客人来的时候，我总会先温杯，再慢慢续水。"
    return text[:max_chars]


def _stub_summary(existing: str, old_texts: list[str]) -> str:
    """确定性的摘要桩：把旧轮原文并入摘要（截断，模拟真实摘要的体积）。"""
    merged = "；".join([part for part in [existing, *old_texts] if part])
    return merged[-1200:]


def _split_sections(system: str) -> dict[str, str]:
    """按固定 header 切分 system 文本；无 header 的内容归入 base。"""
    positions = sorted(
        (system.find(header), name, header)
        for name, header in _SECTION_HEADERS
        if header in system
    )
    first = positions[0][0] if positions else len(system)
    out = {"base": system[:first].strip()}
    for index, (pos, name, _header) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(system)
        out[name] = system[pos:end].strip()
    return out


def _build_state(
    engine: str,
    messages,
    summary: str,
    profile: dict,
    facts: list[dict],
    l2: str,
    *,
    last_user_at: str | None = None,
) -> dict:
    state: dict = {
        "system_prompt": BASE_PROMPT,
        "messages": list(messages),
        "recalled_memories": [],
        "recalled_lore": [],
        "summary": summary,
        "agent_id": 1,
        "session_id": 1,
        "last_user_at": last_user_at or "",
    }
    if engine == "v23":
        state.update(
            {
                "user_profile": dict(profile),
                "recalled_facts": list(facts),
                "l2_portrait": l2,
            }
        )
    return state


async def _snapshot(
    compose_prompt,
    estimate_tokens,
    engine: str,
    round_no: int,
    state: dict,
    *,
    token_budget: int,
    recent_rounds: int,
    now=None,
) -> dict:
    kwargs: dict = {"token_budget": token_budget, "recent_rounds": recent_rounds}
    if now is not None:
        kwargs["now"] = now
    if engine == "v23":
        # 当前默认值与 compose_prompt 的默认值一致；显式传入便于报告自证
        kwargs.update({"profile_ratio": 0.10, "l2_ratio": 0.05})
    result = await compose_prompt(state, **kwargs)
    prompt = result["prompt"]
    system = str(prompt[0]["content"])
    recent = prompt[1:]
    recent_text = "\n".join(str(m["content"]) for m in recent)
    total = sum(estimate_tokens(str(m["content"])) for m in prompt)
    sections = _split_sections(system)
    section_tokens = {name: estimate_tokens(text) for name, text in sections.items()}
    section_tokens["recent"] = sum(estimate_tokens(str(m["content"])) for m in recent)
    section_tokens = {k: v for k, v in section_tokens.items() if v}
    return {
        "round": round_no,
        "prompt_token": total,
        "sections": section_tokens,
        "ratio": {k: round(v / total, 4) for k, v in section_tokens.items()} if total else {},
        "system": system,
        "recent_text": recent_text,
        "recent_count": len(recent),
    }


async def _run(args: argparse.Namespace, engine: str) -> int:
    from app.agents.nodes.compose_prompt import _estimate_tokens, compose_prompt
    from app.config import get_settings
    from langchain_core.messages import AIMessage, HumanMessage

    import report as report_mod

    try:
        # V3 M1：当前版本走统一时钟（支持 --now / EVAL_FIXED_NOW）
        from app.core import clock

        base_now = clock.now()
        eval_fixed_now = clock.fixed_now_raw()
        tz = clock.TZ
    except ImportError:
        # --extra-src 交叉校验用的是旧版 src（无 app.core.clock）：退回真实时间
        tz = ZoneInfo("Asia/Shanghai")
        base_now = datetime.now(tz)
        eval_fixed_now = None

    settings = get_settings()
    prompt_version = getattr(settings, "prompt_version", None)
    # R1：manifest 的运行时刻用墙钟；base_now 只是合成消息的“模拟时刻”
    started_dt = datetime.now(tz)
    run_id = started_dt.strftime("%Y%m%d-%H%M%S")
    started_at = started_dt.isoformat(timespec="seconds")

    # v22 = V2.2 行为：摘要触发 20、无 profile/facts/L2；v23 = 当前配置默认行为
    trigger = 20 if engine == "v22" else settings.summary_trigger_rounds
    chunk = settings.summary_chunk_rounds
    recent_rounds = settings.prompt_recent_rounds
    every_merges = getattr(settings, "memory_l2_every_merges", 3)
    l2_max_chars = getattr(settings, "memory_l2_max_chars", 300)

    messages: list = []
    summary = ""
    merges = 0
    profile = {"称呼": "阿林", "禁忌": "不喝浓茶"} if engine == "v23" else {}
    facts: list[dict] = []
    l2 = ""

    checkpoints: dict[str, dict] = {}
    checkpoint_15: dict | None = None
    last_snapshot: dict | None = None

    for round_no in range(1, args.rounds + 1):
        # V3 M1：合成时间按 30 分钟递进，末轮 == now；at 只给用户消息（契约 §2.3）
        user_at = base_now - timedelta(minutes=30 * (args.rounds - round_no))
        last_user_at = (
            (user_at - timedelta(minutes=30)).isoformat(timespec="seconds")
            if round_no > 1
            else ""
        )
        user_text = ROUND1_TEXT if round_no == 1 else (
            f"第{round_no}轮：今天又去了城南的旧书摊，翻到一本讲茶器的旧书。"
        )
        reply_text = f"第{round_no}轮：我给您续了茶，听您慢慢说。"
        messages.append(
            HumanMessage(
                content=user_text,
                additional_kwargs={"at": user_at.isoformat(timespec="seconds")},
            )
        )
        messages.append(AIMessage(content=reply_text))

        if engine == "v23" and round_no == 3:
            for index, text in enumerate(FACT_TEXTS):
                facts.append(
                    {
                        "text": text,
                        "importance": 0.8 - index * 0.1,
                        "updated_at": base_now.isoformat(),
                        "created_at": base_now.isoformat(),
                    }
                )

        # 模拟滚动摘要：用户消息数 > trigger 时压缩最旧 chunk 轮（与 update_memory 同口径）
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        old_texts: list[str] = []
        if human_count > trigger:
            cut = 0
            seen = 0
            for index, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    seen += 1
                    if seen == chunk + 1:
                        cut = index
                        break
            old = messages[:cut] if cut else list(messages)
            old_texts = [str(m.content) for m in old if isinstance(m, HumanMessage)]
            summary = _stub_summary(summary, old_texts)
            messages = messages[cut:]
            merges += 1
            if engine == "v23" and merges % max(1, every_merges) == 0:
                l2 = _l2_stub(l2_max_chars)

        if round_no in CHECKPOINTS or round_no == CHECKPOINT_15:
            state = _build_state(
                engine,
                messages,
                summary,
                profile,
                facts,
                l2,
                last_user_at=last_user_at,
            )
            snapshot = await _snapshot(
                compose_prompt,
                _estimate_tokens,
                engine,
                round_no,
                state,
                token_budget=settings.prompt_token_budget,
                recent_rounds=recent_rounds,
                now=base_now,
            )
            where = None
            if ROUND1_MARKER in snapshot["system"]:
                where = "summary" if ROUND1_MARKER in summary else "other-section"
            elif ROUND1_MARKER in snapshot["recent_text"]:
                where = "recent"
            if round_no in CHECKPOINTS:
                checkpoints[str(round_no)] = {
                    "prompt_token": snapshot["prompt_token"],
                    "sections": snapshot["sections"],
                    "ratio": snapshot["ratio"],
                }
            if round_no == CHECKPOINT_15:
                checkpoint_15 = {
                    "round": 15,
                    "prompt_token": snapshot["prompt_token"],
                    "round1_injected": where is not None,
                    "where": where,
                    "summary_present": bool(summary),
                }
            last_snapshot = snapshot

    assert last_snapshot is not None
    round100_text = last_snapshot["system"]
    result = {
        "schema_version": "1.0",
        "engine": engine,
        "extra_src": args.extra_src,
        "rounds": args.rounds,
        "config": {
            "token_budget": settings.prompt_token_budget,
            "recent_rounds": recent_rounds,
            "summary_trigger_rounds": trigger,
            "summary_chunk_rounds": chunk,
            "l2_every_merges": every_merges if engine == "v23" else None,
            "prompt_version": prompt_version,
            "eval_fixed_now": eval_fixed_now,
        },
        "checkpoints": checkpoints,
        "checkpoint_15": checkpoint_15,
        "round_100": {
            "contains_facts": all(text in round100_text for text in FACT_TEXTS),
            "contains_l2": L2_MARKER in round100_text,
            "contains_round1": ROUND1_MARKER in round100_text,
        },
        "usage": {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "judge_cache_hits": 0,
            "judge_cache_misses": 0,
        },
    }

    out_path = Path(args.out) if args.out else OUT_DIR / f"longsession-{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = report_mod.build_manifest(
        run_id=run_id,
        command="uv run python scripts/run_long_session.py --engine "
        + engine
        + (f" --out {args.out}" if args.out else "")
        + (f" --extra-src {args.extra_src}" if args.extra_src else "")
        + (f" --now {args.now}" if args.now else ""),
        started_at=started_at,
        finished_at=datetime.now(tz).isoformat(timespec="seconds"),
        env={
            "python": sys.version.split()[0],
            "database": None,
            "llm_model": None,
            "embedding_model": None,
            "rerank_model": None,
            "prompt_version": prompt_version,
            "eval_fixed_now": eval_fixed_now,
        },
        inputs={
            "suite_sha256": None,
            "fixtures_sha256": hashlib.sha256(
                (ROOT / "scripts" / "run_long_session.py").read_bytes()
            ).hexdigest(),
        },
        outputs={"report_sha256": report_mod.file_sha256(out_path), "cases_sha256": None},
        usage=result["usage"],
        budget_section="4.6#6",
        budget_prompt_tokens=0,
    )
    manifest_path = report_mod.write_manifest(manifest, out_path.with_suffix(".manifest.json"))

    print(f"---- long-session（engine={engine}）----")
    for key in ("10", "20", "50", "100"):
        row = checkpoints.get(key)
        if row:
            print(f"round {key:>3}: prompt_token={row['prompt_token']} ratio={row['ratio']}")
    print(f"checkpoint 15 : {checkpoint_15}")
    print(f"round 100     : {result['round_100']}")
    print(f"输出：{out_path}")
    print(f"manifest：{manifest_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.3 长会话曲线（不调 LLM）")
    ap.add_argument("--engine", choices=("v22", "v23"), default="v23")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 eval/report/longsession-<run_id>.json）")
    ap.add_argument("--extra-src", default=None, help="把旧版 src 插到 sys.path 最前（交叉校验用）")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument(
        "--now",
        default=None,
        help="冻结时钟（ISO 8601；等价设置 EVAL_FIXED_NOW）",
    )
    args = ap.parse_args()

    # 导入 app 之前确定 sys.path：当前 src 在后，--extra-src 在最前（硬要求）
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval"))
    if args.extra_src:
        sys.path.insert(0, str(Path(args.extra_src).resolve()))

    if args.now:
        # 先校验（非法值直接报错，不静默回退），再交给 clock 的统一入口
        from datetime import datetime as _datetime

        _datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        os.environ["EVAL_FIXED_NOW"] = args.now

    import asyncio

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(_run(args, args.engine)))


if __name__ == "__main__":
    main()
