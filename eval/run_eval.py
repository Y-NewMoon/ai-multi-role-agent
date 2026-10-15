"""M2.5 评测闭环：跑评测集，量化 人设一致性 / 记忆命中率 / 串味=0，产出可复现报告。

用法：
    .venv\\Scripts\\python.exe eval\\run_eval.py             # 全量（人设一致性会调用真实 LLM）
    .venv\\Scripts\\python.exe eval\\run_eval.py --no-llm    # 零成本：跳过人设一致性（该指标需 LLM）
    .venv\\Scripts\\python.exe eval\\run_eval.py --inmemory  # 强制 InMemoryStore（默认走真库 roleplay_test）

指标口径（与方案 §8 对齐）：
- 串味=0：A/B 两 namespace 不相等；A 写入的事实不出现在 B 的召回里（反之亦然）。泄漏数必须=0。
- 记忆命中率：埋事实 → 用 query 走 `MemoryStore.search(top-k)`，`expect` 全部被召回即记为命中。
- 人设一致性（混合）：先规则——命中 forbid 直接 fail、命中 expect 直接 pass；
  两者都未命中判为「存疑」，交真实 LLM 按 1-5 rubric 复核，≥3 记 pass。

退出码：串味>0 时返回 1（可接 CI），否则 0。
"""

import argparse
import asyncio
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 下 psycopg 异步需 SelectorEventLoop（与 scripts/run_server.py 一致）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

from app.agents.graph import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.factory import create_embedding_provider, create_llm_provider  # noqa: E402
from app.memory.namespace import memory_namespace  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402

CASES_DIR = ROOT / "eval" / "cases"
REPORT_DIR = ROOT / "eval" / "report"
_TOP_K = 5


def _load_cases(name: str) -> list[dict]:
    path = CASES_DIR / f"{name}.jsonl"
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


async def _make_store(inmemory: bool):
    """返回 (store, 存储标签, pool 或 None)；pool 非空时由调用方关闭。"""
    settings = get_settings()
    index = {
        "dims": settings.embedding_dims,
        "embed": create_embedding_provider(settings),
        "fields": ["text"],
    }
    if inmemory or not settings.test_database_url:
        return InMemoryStore(index=index), "inmemory", None

    pool = AsyncConnectionPool(
        settings.test_database_url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        await pool.open()
        await pool.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001 - 库不可用时回退，保证评测仍能产出报告
        await pool.close()
        print(f"[warn] 测试库不可用，回退 InMemoryStore：{exc}")
        return InMemoryStore(index=index), "inmemory(fallback)", None

    from langgraph.store.postgres.aio import AsyncPostgresStore

    store = AsyncPostgresStore(pool, index=index)
    await store.setup()
    db = settings.test_database_url.rsplit("/", 1)[-1]
    return store, f"postgres:{db}", pool


async def _purge(store, agent_id) -> None:
    """清空该角色的记忆 namespace，避免污染与跨用例串数据。"""
    ns = memory_namespace(agent_id)
    for item in await store.asearch(ns, limit=1000):
        await store.adelete(item.namespace, item.key)


# ---------------------------------------------------------------------------
# 指标 1：串味 = 0
# ---------------------------------------------------------------------------
async def _run_isolation(store, run_id: str) -> tuple[int, list[dict]]:
    ms = MemoryStore(store)
    leaks = 0
    details: list[dict] = []
    for c in _load_cases("isolation"):
        a = f"evalA-{run_id}-{c['id']}"
        b = f"evalB-{run_id}-{c['id']}"
        try:
            ns_ok = memory_namespace(a) != memory_namespace(b)
            for f in c["facts_a"]:
                await ms.put(a, f)
            for f in c["facts_b"]:
                await ms.put(b, f)
            recall_a = await ms.search(a, query=c["probe"], limit=_TOP_K)
            recall_b = await ms.search(b, query=c["probe"], limit=_TOP_K)
            # A 的召回里混入 B 的事实（或反之）即一次泄漏；namespace 相等也算一次
            cross = len(set(recall_a) & set(c["facts_b"])) + len(set(recall_b) & set(c["facts_a"]))
            case_leaks = cross + (0 if ns_ok else 1)
            leaks += case_leaks
            details.append({"id": c["id"], "leaks": case_leaks, "namespace_distinct": ns_ok})
        finally:
            await _purge(store, a)
            await _purge(store, b)
    return leaks, details


# ---------------------------------------------------------------------------
# 指标 2：记忆命中率
# ---------------------------------------------------------------------------
async def _run_memory(store, run_id: str) -> tuple[float, list[dict]]:
    ms = MemoryStore(store)
    cases = _load_cases("memory")
    hits = 0
    details: list[dict] = []
    for c in cases:
        agent = f"evalM-{run_id}-{c['id']}"
        try:
            for f in c["facts"]:
                await ms.put(agent, f)
            recalled = await ms.search(agent, query=c["query"], limit=c.get("limit", _TOP_K))
            hit = all(e in recalled for e in c["expect"])
            hits += 1 if hit else 0
            details.append(
                {"id": c["id"], "hit": hit, "expect": c["expect"], "recalled": recalled}
            )
        finally:
            await _purge(store, agent)
    rate = hits / len(cases) if cases else 1.0
    return rate, details


# ---------------------------------------------------------------------------
# 指标 3：人设一致性（规则先筛 + LLM 复核）
# ---------------------------------------------------------------------------
def _first_score(text: str) -> int:
    m = re.search(r"[1-5]", text or "")
    return int(m.group()) if m else 1


async def _judge(llm, *, name: str, persona: str, background: str, user_msg: str, reply: str) -> int:
    prompt = [
        {"role": "system", "content": "你是严格的中文角色扮演质量评审员，只输出一个 1-5 的整数分数，不要解释。"},
        {
            "role": "user",
            "content": (
                f"角色名：{name}\n人设：{persona}\n背景：{background}\n"
                f"用户说：{user_msg}\n角色回复：{reply}\n\n"
                "请就「回复是否符合该角色人设与口吻」打分：5=完全符合，1=完全不符。"
            ),
        },
    ]
    return _first_score(await llm.generate(prompt))


async def _run_persona(store, llm, run_id: str) -> tuple[float, list[dict]]:
    cases = _load_cases("persona")
    passed = 0
    details: list[dict] = []
    for c in cases:
        agent = f"evalP-{run_id}-{c['id']}"
        graph = build_graph(
            llm=llm,
            memory_store=MemoryStore(store),
            checkpointer=InMemorySaver(),
            store=store,
        )
        cfg = {"configurable": {"thread_id": f"{agent}:persona"}}
        replies: list[str] = []
        try:
            for turn in c["turns"]:
                state_in = {
                    "messages": [HumanMessage(content=turn)],
                    "agent_id": agent,
                    "session_id": 0,
                    "agent_name": c["name"],
                    "persona_prompt": c["persona"],
                    "background": c.get("background", ""),
                }
                result = await graph.ainvoke(state_in, config=cfg)
                replies.append(str(result.get("reply", "")))

            combined = "\n".join(replies)
            forbid = [k for k in c.get("forbid_any", []) if k in combined]
            expect = [k for k in c.get("expect_any", []) if k in combined]
            if forbid:
                verdict, judge_score = False, None  # 规则：命中禁忌，直接判负
            elif expect:
                verdict, judge_score = True, None  # 规则：命中期望，直接判正
            else:  # 存疑 → LLM 复核
                judge_score = await _judge(
                    llm,
                    name=c["name"],
                    persona=c["persona"],
                    background=c.get("background", ""),
                    user_msg=c["turns"][-1],
                    reply=replies[-1],
                )
                verdict = judge_score >= 3
            passed += 1 if verdict else 0
            details.append(
                {
                    "id": c["id"],
                    "pass": verdict,
                    "rule_forbid": forbid,
                    "rule_expect": expect,
                    "judge_score": judge_score,
                    "replies": replies,
                }
            )
        finally:
            await _purge(store, agent)
    rate = passed / len(cases) if cases else 1.0
    return rate, details


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.0%}"


def _render_md(report: dict) -> str:
    m = report["metrics"]
    leaks = m["isolation_leaks"]
    persona = m.get("persona_consistency")
    lines = [
        "# M2.5 评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 存储后端：{report['storage']}",
        f"- LLM 评审：{'开启' if report['llm_enabled'] else '关闭（--no-llm）'}",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 结果 | 判定 |",
        "|---|---|---|",
        f"| 串味（memory leak） | {leaks} | {'PASS' if leaks == 0 else 'FAIL'} |",
        f"| 记忆命中率 | {_pct(m.get('memory_hit_rate'))} | - |",
        f"| 人设一致性 | {_pct(persona)} | {'skipped' if persona is None else '-'} |",
        "",
        "## 明细",
        "",
        "### 串味",
        "",
    ]
    for d in report["details"].get("isolation", []):
        lines.append(f"- `{d['id']}`：泄漏 {d['leaks']}，namespace 互异={d['namespace_distinct']}")
    lines += ["", "### 记忆命中率", ""]
    for d in report["details"].get("memory", []):
        mark = "命中" if d["hit"] else "未命中"
        lines.append(f"- `{d['id']}`：{mark}（expect={d['expect']}）")
    lines += ["", "### 人设一致性", ""]
    persona_detail = report["details"].get("persona", [])
    if isinstance(persona_detail, str):
        lines.append(f"- {persona_detail}")
    else:
        for d in persona_detail:
            score = "-" if d["judge_score"] is None else f"judge={d['judge_score']}"
            lines.append(f"- `{d['id']}`：{'PASS' if d['pass'] else 'FAIL'}（{score}）")
    return "\n".join(lines) + "\n"


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORT_DIR / f"eval-{ts}.json"
    md_path = REPORT_DIR / f"eval-{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    print(f"[OK] 报告：{md_path}")
    print(f"[OK] 原始：{json_path}")
    return md_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M2.5 评测闭环")
    ap.add_argument("--no-llm", action="store_true", help="跳过人设一致性（需真实 LLM，零成本模式）")
    ap.add_argument("--inmemory", action="store_true", help="强制 InMemoryStore（默认走真库 roleplay_test）")
    return ap.parse_args(argv)


async def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    run_id = uuid.uuid4().hex[:6]

    store, storage_label, pool = await _make_store(args.inmemory)
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": storage_label,
        "llm_enabled": not args.no_llm,
        "metrics": {},
        "details": {},
    }
    try:
        leaks, iso_details = await _run_isolation(store, run_id)
        report["metrics"]["isolation_leaks"] = leaks
        report["details"]["isolation"] = iso_details

        mem_rate, mem_details = await _run_memory(store, run_id)
        report["metrics"]["memory_hit_rate"] = mem_rate
        report["details"]["memory"] = mem_details

        if args.no_llm:
            report["metrics"]["persona_consistency"] = None
            report["details"]["persona"] = "skipped (--no-llm)"
        else:
            per_rate, per_details = await _run_persona(store, create_llm_provider(settings), run_id)
            report["metrics"]["persona_consistency"] = per_rate
            report["details"]["persona"] = per_details
    finally:
        if pool is not None:
            await pool.close()

    _write_report(report)

    m = report["metrics"]
    print("---- M2.5 评测汇总 ----")
    print(f"存储后端      : {storage_label}")
    print(f"串味          : {m['isolation_leaks']}（要求 0）")
    print(f"记忆命中率    : {_pct(m.get('memory_hit_rate'))}")
    print(f"人设一致性    : {_pct(m.get('persona_consistency'))}")
    return 1 if leaks > 0 else 0


def main() -> None:
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
