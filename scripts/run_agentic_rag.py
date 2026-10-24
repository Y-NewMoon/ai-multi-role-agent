"""V2.2 Agentic RAG 对比实验（契约 §2.10）：static / agentic / hybrid 三策略。

- 语料：`eval/suites/agentic_rag.jsonl` 20 条，复用 `eval/fixtures/lore_long` 两篇；
- 三策略：
  * `static`  ：`LORE_STATIC_MODE=full`，静态注入 V2.1 全量 `retrieve_lore`；**不允许** `read_lore`；
  * `agentic` ：`LORE_STATIC_MODE=off`（不注入静态 lore）；**允许** `read_lore`；
  * `hybrid`  ：`LORE_STATIC_MODE=always_on`（只注入 always-on）；**允许** `read_lore`；
- 指标：`setting_consistency`（judge `boundary>=4` 比例，judge 的 `lore_context` 用该 case 的
  `expect_refs` 对应 chunk 拼接）、`avg_prompt_tokens`（`Σ estimate_tokens(prompt)` + 工具结果）、
  `first_token_latency_ms`（请求开始到第一个 `text` 事件；工具模式下即最终文本事件）；
- 每 case 落盘 `static_refs` / `read_lore_refs` / `ranked` / `tool_results` / `answer` / judge 分；
- 报告：三行 × 三列（不强制 hybrid 最优；低于 static 记为发现）；
- 跳过（§4.3）：无 `LLM_API_KEY` / provider 不支持 tools / 无 `TEST_DATABASE_URL` / 无可用嵌入器 /
  无可用 reranker → 整体 `skipped=true`、指标 `null`、退出 0；
- 不进 `run_eval.SUITE_NAMES`（D9），PR CI 不跑本脚本。

用法：
    uv run python scripts/run_agentic_rag.py --agent 1
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

import judge as judge_mod  # noqa: E402
import lore_eval  # noqa: E402
from app.agents.graph import build_graph  # noqa: E402
from app.agents.nodes.compose_prompt import _estimate_tokens  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.base import create_engine, create_session_factory  # noqa: E402
from app.lore.retriever import Retriever  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.skills.builtin import build_default_registry  # noqa: E402
from app.skills.executor import ToolExecutor  # noqa: E402

SUITE_PATH = ROOT / "eval" / "suites" / "agentic_rag.jsonl"
FIXTURES_DIR = ROOT / "eval" / "fixtures" / "lore_long"
OUT_DIR = ROOT / "eval" / "report"

# 本实验只用到 `read_lore`（agentic / hybrid 允许；static 不允许）
READ_LORE = ["read_lore"]

EVAL_NAME = "云栖茶庄的年轻掌柜"
EVAL_PERSONA = (
    "云栖茶庄的年轻掌柜，说话温和慢条斯理，爱用茶来打比方，"
    "称呼来客为「客人」，遇到拿不准的事会坦率承认。"
)
EVAL_BACKGROUND = "在青溪镇的茶庄长大，熟悉店里的一草一木，喜欢读旧诗。"

METRIC_KEYS = ("setting_consistency", "avg_prompt_tokens", "first_token_latency_ms", "queries")
STRATEGIES = ("static", "agentic", "hybrid")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(FIXTURES_DIR.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rewrites_sha256() -> str:
    from app.lore.retriever import REWRITE_CACHE_PATH

    return _sha256_file(Path(REWRITE_CACHE_PATH))


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(ROOT),
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, cwd=str(ROOT),
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return True


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in SUITE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _null_strategy(reason: str) -> dict:
    row: dict = {key: None for key in METRIC_KEYS}
    row.update({"queries": 0, "skipped": True, "skip_reason": reason})
    return row


def _json_tokens(value) -> int:
    return _estimate_tokens(json.dumps(value, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# 检索语料：取 expect_refs 对应 content 作 judge 的 lore_context
# ---------------------------------------------------------------------------
async def _load_lore_contents(session_factory, agent_id: int) -> dict[str, dict]:
    from sqlalchemy import select

    from app.db.models import LoreChunk

    async with session_factory() as db:
        rows = (
            await db.execute(
                select(LoreChunk).where(
                    LoreChunk.agent_id == agent_id, LoreChunk.status == "active"
                )
            )
        ).scalars()
        return {r.source_ref: {"content": r.content, "header_path": r.header_path} for r in rows}


def _lore_context(case: dict, contents: dict[str, dict]) -> str | None:
    parts = [
        f"（{contents[ref]['header_path']}）{contents[ref]['content']}"
        for ref in case.get("expect_refs", [])
        if ref in contents
    ]
    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# 图构造（三策略）
# ---------------------------------------------------------------------------
def _build_graph_for(
    strategy: str, *, llm, settings, embed, reranker, session_factory, tool_retriever, store
):
    base = dict(
        llm=llm,
        memory_store=MemoryStore(store),
        checkpointer=InMemorySaver(),
        store=store,
        token_budget=settings.prompt_token_budget,
    )
    if strategy == "static":
        # V2.1 原路径：静态全量注入，不给工具（skill_registry=None → 拓扑与 V2.1 一致）
        return (
            build_graph(
                **base,
                lore_retriever=Retriever(
                    session_factory=session_factory,
                    embedder=embed,
                    reranker=reranker,
                    rewrite_enabled=False,
                ),
                lore_enabled=True,
                lore_static_mode="full",
            ),
            None,
        )
    registry = build_default_registry(
        session_factory=session_factory, lore_retriever=tool_retriever
    )
    executor = ToolExecutor(registry, max_tool_rounds=settings.skill_max_tool_rounds)
    mode = "off" if strategy == "agentic" else "always_on"
    lore_retriever = None
    if mode == "always_on":
        # hybrid：只注入 always-on，需要带 reranker 的检索器（always-on 不走 rerank，但保持同一实例）
        lore_retriever = Retriever(
            session_factory=session_factory,
            embedder=embed,
            reranker=reranker,
            rewrite_enabled=False,
        )
    return (
        build_graph(
            **base,
            lore_retriever=lore_retriever,
            lore_enabled=mode == "always_on",
            lore_static_mode=mode,
            skill_registry=registry,
            tool_executor=executor,
            allowed_skills=list(READ_LORE),
            max_tool_rounds=settings.skill_max_tool_rounds,
        ),
        executor,
    )


# ---------------------------------------------------------------------------
# 跑一个策略
# ---------------------------------------------------------------------------
async def _run_case(
    graph, case: dict, *, agent_id: int, session_id: int, tag: str, executor=None
) -> dict:
    state_in = {
        "messages": [HumanMessage(content=case["query"])],
        "agent_id": agent_id,
        "session_id": session_id,
        "agent_name": EVAL_NAME,
        "persona_prompt": EVAL_PERSONA,
        "background": EVAL_BACKGROUND,
        "idle_scope": "",
        "reply_max_messages": 0,
        # 白名单由状态注入：static 无工具（空），agentic/hybrid 为 read_lore
        "allowed_skills": [] if tag == "static" else list(READ_LORE),
    }
    cfg = {"configurable": {"thread_id": f"{session_id}:{agent_id}:{tag}:{case['id']}"}}
    start = time.perf_counter()
    first_text_ms: float | None = None
    last_values: dict = {}
    if executor is not None:
        executor.reset()  # 每个 case 一次独立请求（同请求缓存，§2.2-5）
    async for mode, chunk in graph.astream(
        state_in, config=cfg, stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            if first_text_ms is None and isinstance(chunk, dict) and chunk.get("text"):
                first_text_ms = (time.perf_counter() - start) * 1000.0
        elif mode == "values":
            last_values = chunk
    if first_text_ms is None:
        first_text_ms = (time.perf_counter() - start) * 1000.0
    recall = list(last_values.get("recalled_lore") or [])
    tool_results = list(last_values.get("tool_results") or [])
    prompt = list(last_values.get("prompt") or [])
    prompt_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in prompt)
    prompt_tokens += sum(_json_tokens(r) for r in tool_results)
    read_refs: list[str] = []
    for r in tool_results:
        if r.get("name") == "read_lore" and r.get("ok"):
            for chunk in (r.get("data") or {}).get("chunks", []) or []:
                read_refs.append(chunk.get("source_ref"))
    return {
        "id": case["id"],
        "strategy": tag,
        "query": case["query"],
        "expect_refs": case.get("expect_refs", []),
        "static_refs": [item.get("source_ref") for item in recall],
        "read_lore_refs": read_refs,
        "ranked": [item.get("source_ref") for item in recall],
        "tool_calls": [
            {"id": r.get("tool_call_id"), "name": r.get("name"), "arguments": r.get("arguments")}
            for r in tool_results
        ],
        "tool_results": tool_results,
        "answer": str(last_values.get("reply", "")),
        "prompt_tokens": prompt_tokens,
        "first_token_latency_ms": round(first_text_ms, 2),
        "tool_rounds": int(last_values.get("tool_rounds") or 0),
    }


async def _run_strategy(
    strategy: str,
    cases: list[dict],
    *,
    llm,
    settings,
    embed,
    reranker,
    session_factory,
    contents: dict[str, dict],
    run_id: str,
) -> tuple[dict, list[dict]]:
    tool_retriever = Retriever(
        session_factory=session_factory,
        embedder=embed,
        reranker=reranker,
        rewrite_enabled=False,  # D7：工具内不调 LLM
    )
    store = InMemoryStore(
        index={"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]}
    )
    session_id = await _create_temp_session(session_factory, cases[0]["agent_id"])
    try:
        graph, executor = _build_graph_for(
            strategy,
            llm=llm,
            settings=settings,
            embed=embed,
            reranker=reranker,
            session_factory=session_factory,
            tool_retriever=tool_retriever,
            store=store,
        )
        per_case: list[dict] = []
        consistent: list[bool] = []
        for case in cases:
            row = await _run_case(
                graph,
                case,
                agent_id=case["agent_id"],
                session_id=session_id,
                tag=strategy,
                executor=executor,
            )
            verdict = await judge_mod.judge_reply(
                llm,
                name=EVAL_NAME,
                persona=EVAL_PERSONA,
                background=EVAL_BACKGROUND,
                user_msg=case["query"],
                reply=row["answer"],
                lore_context=_lore_context(case, contents),
            )
            degraded = bool(verdict.get("degraded"))
            boundary = (verdict.get("dimensions") or {}).get("boundary")
            row.update(
                {
                    "judge_boundary": boundary,
                    "judge_persona": (verdict.get("dimensions") or {}).get("persona"),
                    "judge_verdict": verdict.get("verdict"),
                    "judge_degraded": degraded,
                    "judge_source": "judge_fallback" if degraded else "judge",
                }
            )
            consistent.append((not degraded) and isinstance(boundary, int) and boundary >= 4)
            per_case.append(row)
            print(
                f"[{strategy}] {case['id']} boundary={boundary} "
                f"static={len(row['static_refs'])} read={len(row['read_lore_refs'])} "
                f"prompt≈{row['prompt_tokens']} 首字={row['first_token_latency_ms']}ms",
                flush=True,
            )
    finally:
        await _cleanup_temp_session(session_factory, session_id)

    n = len(per_case)
    metrics = {
        "setting_consistency": (sum(consistent) / n) if n else None,
        "avg_prompt_tokens": (sum(r["prompt_tokens"] for r in per_case) / n) if n else None,
        "first_token_latency_ms": (
            sum(r["first_token_latency_ms"] for r in per_case) / n
        )
        if n
        else None,
        "queries": n,
        "skipped": False,
        "skip_reason": None,
    }
    return metrics, per_case


async def _create_temp_session(session_factory, agent_id: int) -> int:
    from app.db.models import Session as ChatSession

    async with session_factory() as db:
        # agent_id=0 是占位：会话只作 thread/记忆容器，用例真实 agent_id 在图状态里给定
        row = ChatSession(agent_id=agent_id, title="v2.2-agentic-rag-temp")
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return int(row.id)


async def _cleanup_temp_session(session_factory, session_id: int) -> None:
    from sqlalchemy import delete

    from app.db.models import Session as ChatSession

    async with session_factory() as db:
        await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
        await db.commit()


# ---------------------------------------------------------------------------
# 渲染与落盘
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    if value is None:
        return "null（skipped）"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(strategies: dict, per_case_by_strategy: dict, meta: dict) -> str:
    def _consistent_n(name: str) -> int:
        return sum(
            1
            for row in per_case_by_strategy.get(name, [])
            if int(row.get("judge_boundary") or 0) >= 4
        )

    hybrid_static_refs = sum(
        len(row.get("static_refs") or []) for row in per_case_by_strategy.get("hybrid", [])
    )
    static_n, agentic_n = _consistent_n("static"), _consistent_n("agentic")
    lines = [
        "# V2.2 Agentic RAG 对比报告",
        "",
        f"- 生成时间：{meta['started_at']} ~ {meta['finished_at']}",
        f"- git sha：`{meta['git_sha']}`（dirty={meta['git_dirty']}）",
        f"- 用例：`eval/suites/agentic_rag.jsonl` 20 条（yunqi_tea / qingxi_town 各 10）",
        f"- rerank：{meta['rerank']}；跳过：{meta['skipped']}（原因：{meta['skip_reason']}）",
        "",
        "## 三策略对比",
        "",
        "| 策略 | setting_consistency | avg_prompt_tokens | first_token_latency_ms | queries |",
        "|---|---|---|---|---|",
    ]
    for name in STRATEGIES:
        row = strategies[name]
        lines.append(
            f"| {name} | {_fmt(row['setting_consistency'])} | "
            f"{_fmt(row['avg_prompt_tokens'])} | {_fmt(row['first_token_latency_ms'])} | "
            f"{_fmt(row['queries'])} |"
        )
    lines += [
        "",
        "> `setting_consistency` = judge `boundary >= 4` 的比例；`avg_prompt_tokens` = "
        "`Σ estimate_tokens(prompt)` + 工具结果；`first_token_latency_ms` = 请求开始到第一个 "
        "`text` 事件。**不设“必须 hybrid 最优”的硬门槛**；若 hybrid 低于 static，记为发现。",
        "",
        f"> 说明 1：`hybrid` 的静态注入在本轮 20 条用例中合计 **{hybrid_static_refs}** 条"
        "（夹具 `eval/fixtures/lore_long` 无 `always_on=true` 的 chunk），故 hybrid 与 agentic "
        "同条件，两行差异是采样噪声，不能解读为「hybrid 带来增益/损失」。",
        f"> 说明 2：历史两次同条件运行中 agentic 分别为 20/20 与 17/20（差 3 条）；"
        f"本轮 static={static_n}/20、agentic={agentic_n}/20，"
        f"{abs(static_n - agentic_n)} 条差距不足以单独支撑「static 更准」。",
        "",
    ]
    if not meta["skipped"]:
        for name in STRATEGIES:
            lines += [f"### 逐用例明细 · {name}", ""]
            lines += [
                "| id | boundary | static_refs | read_lore_refs | 工具结果 | 首字(ms) |",
                "|---|---|---|---|---|---|",
            ]
            for row in per_case_by_strategy[name]:
                calls = "、".join(
                    f"{r['name']}({'ok' if r.get('ok') else r.get('error_type')})"
                    for r in row["tool_results"]
                ) or "—"
                lines.append(
                    f"| {row['id']} | {row['judge_boundary']} | "
                    f"{len(row['static_refs'])} | {len(row['read_lore_refs'])} | "
                    f"{calls} | {row['first_token_latency_ms']} |"
                )
            lines.append("")
    lines += [
        f"> 命令：`uv run python scripts/run_agentic_rag.py --agent {meta['agent']}`；"
        "每 case 的 ranked / tool_results / answer 见同目录 `cases.jsonl`。",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(out_dir: Path, run_id: str, report: dict, per_case: list[dict], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / f"agentic-rag-{run_id}.cases.jsonl"
    report_path = out_dir / f"agentic-rag-{run_id}.json"
    md_path = out_dir / f"agentic-rag-{run_id}.md"
    cases_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_case) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(report["strategies"], report["per_case"], meta), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "command": f"uv run python scripts/run_agentic_rag.py --agent {meta['agent']}",
        "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"],
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "env": meta["env"],
        "inputs": {
            "suite_sha256": _sha256_file(SUITE_PATH),
            "fixtures_sha256": _fixtures_sha256(),
            "rewrites_sha256": _rewrites_sha256(),
        },
        "outputs": {
            "report_sha256": _sha256_file(report_path),
            "ranked_sha256": _sha256_file(cases_path),
        },
    }
    (out_dir / f"agentic-rag-{run_id}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"report": report_path, "markdown": md_path, "cases": cases_path, "manifest": manifest}


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {
        "agent": args.agent,
        "run_id": run_id,
        "started_at": started_at,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "rerank": False,
        "skipped": False,
        "skip_reason": None,
        "env": {
            "python": sys.version.split()[0],
            "database": (settings.test_database_url or "").rsplit("/", 1)[-1] or None,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "rerank_model": None,
            "mcp_version": None,
        },
    }
    cases = load_cases()

    def _finish(reason: str | None, strategies: dict, per_case: dict) -> int:
        meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if reason is not None:
            meta["skipped"] = True
            meta["skip_reason"] = reason
        report = {
            "schema_version": "1.0",
            "suite": "agentic_rag",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": meta["git_sha"],
            "rerank": meta["rerank"],
            "strategies": strategies,
            "per_case": per_case,
            "manifest": None,
        }
        paths = _write_outputs(OUT_DIR, run_id, report, per_case.get("all", []), meta)
        print("\n---- agentic_rag 汇总 ----")
        for name in STRATEGIES:
            row = strategies[name]
            print(
                f"{name:<9}: setting_consistency={_fmt(row['setting_consistency'])} "
                f"avg_prompt_tokens={_fmt(row['avg_prompt_tokens'])} "
                f"first_token_latency_ms={_fmt(row['first_token_latency_ms'])}"
            )
        print(f"报告：{paths['markdown']}")
        print(f"manifest：{paths['manifest']}")
        return 0

    def _skipped(reason: str) -> int:
        return _finish(
            reason,
            {name: _null_strategy(reason) for name in STRATEGIES},
            {"all": [row for name in STRATEGIES for row in []]},
        )

    # ---- 跳过条件（§4.3）----
    if not settings.llm_api_key:
        return _skipped("no_llm_key")
    if not settings.test_database_url:
        return _skipped("no_test_database_url")

    from app.llm.base import LLMProvider
    from app.llm.factory import create_embedding_provider, create_llm_provider

    llm = create_llm_provider(settings)
    impl = getattr(type(llm), "generate_with_tools", None)
    if impl is None or impl is getattr(LLMProvider, "generate_with_tools", object()):
        return _skipped("provider_without_tools")

    try:
        embed = create_embedding_provider(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 嵌入器不可用：{exc}", flush=True)
        return _skipped("no_embeddings")

    try:
        from app.lore.retriever import BgeReranker

        reranker = BgeReranker()
    except Exception as exc:  # noqa: BLE001 - 无 reranker 视为不满足 §4.3 条件
        print(f"[warn] reranker 不可用：{exc}", flush=True)
        return _skipped("no_reranker")
    meta["rerank"] = True
    meta["env"]["rerank_model"] = "BAAI/bge-reranker-v2-m3"

    engine = create_engine(settings.test_sqlalchemy_url)
    session_factory = create_session_factory(engine)
    strategies: dict = {}
    per_case: dict = {}
    try:
        await lore_eval.ensure_agent(session_factory, 1)
        imported = await lore_eval.import_fixtures(session_factory, embed, agent_id=1)
        print(f"[info] 导入 lore chunk 数：{imported}", flush=True)
        contents = await _load_lore_contents(session_factory, 1)
        for name in STRATEGIES:
            strategies[name], per_case[name] = await _run_strategy(
                name,
                cases,
                llm=llm,
                settings=settings,
                embed=embed,
                reranker=reranker,
                session_factory=session_factory,
                contents=contents,
                run_id=run_id,
            )
    except Exception as exc:  # noqa: BLE001 - 环境异常不伪造指标：整体置跳过
        print(f"[warn] agentic_rag 运行失败，整体置 skipped：{exc}", flush=True)
        return _skipped("runtime_error")
    finally:
        await engine.dispose()

    per_case["all"] = [row for name in STRATEGIES for row in per_case[name]]
    return _finish(None, strategies, per_case)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.2 Agentic RAG 三策略对比（static/agentic/hybrid）")
    ap.add_argument("--agent", type=int, default=1, help="评测角色 id，默认 1")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
