"""V2.2 `skill_use` 评测（契约 §2.12）：工具触发 / 参数 / 失败矩阵 + manifest。

- 语料：`eval/suites/skill_use.jsonl` 30 条（5 工具 × 5 + no-tool 5）；
- 指标：`trigger_accuracy`（工具名正确数 / 25）、`argument_accuracy`（期望参数子集正确数 / 25，
  未调用算错）、`no_tool_accuracy`（正确不调用数 / 5）、`unexpected_error_rate`（非预期
  `error_type` 的工具调用数 / 总调用数）、`avg_rounds`、`queries`、`skipped`、`skip_reason`；
- 每 case 落盘 `tool_calls` / `tool_results` / `final_reply`（供审查方独立重算）；
- 跳过（§4.3）：无 `LLM_API_KEY` / provider 不支持 tools / 无 `TEST_DATABASE_URL` / 无可用嵌入器
  → 整体 `skipped=true`、指标 `null`、退出 0（不伪造数字）；
- 评测前 `ensure_agent` + 建临时会话（`set_reminder` 需要 `session_id`），跑完清理该会话与其 reminders；
- 不进 `run_eval.SUITE_NAMES`（D9），PR CI 不跑本脚本。

用法：
    uv run python scripts/run_skill_eval.py --agent 1
"""

from __future__ import annotations

import os

# 与 ablate_lore.py 同一约定：必须在任何把 huggingface_hub 拉进来的导入之前设置。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
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

import lore_eval  # noqa: E402
from app.agents.graph import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.base import create_engine, create_session_factory  # noqa: E402
from app.lore.retriever import Retriever  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.skills.builtin import build_default_registry  # noqa: E402
from app.skills.builtin.web_search import FakeSearchProvider  # noqa: E402
from app.skills.executor import ToolExecutor  # noqa: E402

SUITE_PATH = ROOT / "eval" / "suites" / "skill_use.jsonl"
FIXTURES_DIR = ROOT / "eval" / "fixtures" / "lore_long"
OUT_DIR = ROOT / "eval" / "report"

# 本角色在评测里绑定全部 5 个工具（真实场景白名单来自 Agent.allowed_skills）
ALL_SKILLS = ["get_current_time", "roll_dice", "web_search", "set_reminder", "read_lore"]

# 评测专用人设（与 fixture 语料同主题；不进业务库，只注入图状态）
EVAL_NAME = "云栖茶庄的年轻掌柜"
EVAL_PERSONA = (
    "云栖茶庄的年轻掌柜，说话温和慢条斯理，爱用茶来打比方，"
    "称呼来客为「客人」，遇到拿不准的事会坦率承认。"
)
EVAL_BACKGROUND = "在青溪镇的茶庄长大，熟悉店里的一草一木，喜欢读旧诗。"

# 假搜索结果（不访问网络；§2.12「评测用 fake provider」）
FAKE_RESULTS = [
    {"title": "BGE-M3 模型说明", "url": "https://example.com/bge-m3", "snippet": "BGE-M3 支持多语言与长文本检索。"},
    {"title": "多语言嵌入综述", "url": "https://example.com/embedding", "snippet": "多语言嵌入模型的对比与选型建议。"},
    {"title": "检索评测方法", "url": "https://example.com/retrieval", "snippet": "hit@k 与 MRR 等检索指标的常见口径。"},
]

METRIC_KEYS = (
    "trigger_accuracy",
    "argument_accuracy",
    "no_tool_accuracy",
    "unexpected_error_rate",
    "avg_rounds",
    "queries",
    "skipped",
    "skip_reason",
)


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures_sha256() -> str:
    """`lore_long` 两篇语料的合并哈希（按文件名排序）。"""
    digest = hashlib.sha256()
    for path in sorted(FIXTURES_DIR.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def _null_metrics(reason: str) -> dict:
    row: dict = {key: None for key in METRIC_KEYS}
    row.update({"queries": 0, "skipped": True, "skip_reason": reason})
    return row


def _arg_subset_ok(expected: dict, actual: dict) -> bool:
    """期望参数是实际参数的子集（逐键相等）；未调用工具时由调用方判错。"""
    return all(actual.get(key) == value for key, value in (expected or {}).items())


def _expected_error_types(case: dict) -> set[str]:
    """本 case 允许出现的 `error_type`（默认空：一切失败都算非预期）。"""
    return set(case.get("expect", {}).get("error_types", []) or [])


def score_case(case: dict, tool_results: list[dict]) -> dict:
    """单 case 判定：触发 / 参数 / no_tool / 轮数。"""
    expect = case.get("expect", {})
    calls = [
        {"id": r.get("tool_call_id"), "name": r.get("name"), "arguments": r.get("arguments") or {}}
        for r in tool_results
    ]
    if expect.get("no_tool"):
        return {"triggered": False, "trigger_ok": not tool_results, "argument_ok": None}
    want_tool = expect.get("tool")
    triggered = want_tool in [c["name"] for c in calls]
    argument_ok = False
    if triggered:
        first = next(c for c in calls if c["name"] == want_tool)
        argument_ok = _arg_subset_ok(expect.get("args") or {}, first["arguments"])
    return {"triggered": triggered, "trigger_ok": triggered, "argument_ok": argument_ok}


def summarize(cases: list[dict], per_case: list[dict]) -> dict:
    """由每 case 明细算 8 项指标（分母按 §2.12：25 / 25 / 5）。"""
    tool_cases = [c for c in cases if not c.get("expect", {}).get("no_tool")]
    no_tool_cases = [c for c in cases if c.get("expect", {}).get("no_tool")]
    by_id = {row["id"]: row for row in per_case}

    trigger_ok = sum(1 for c in tool_cases if by_id[c["id"]]["trigger_ok"])
    arg_ok = sum(1 for c in tool_cases if by_id[c["id"]]["argument_ok"])
    no_tool_ok = sum(1 for c in no_tool_cases if by_id[c["id"]]["trigger_ok"])

    total_calls = 0
    unexpected = 0
    rounds: list[int] = []
    for c in cases:
        row = by_id[c["id"]]
        allowed_errors = _expected_error_types(c)
        for record in row["tool_results"]:
            total_calls += 1
            err = record.get("error_type")
            if err is not None and err not in allowed_errors:
                unexpected += 1
        rounds.append(int(row.get("rounds") or 0))

    return {
        "trigger_accuracy": (trigger_ok / len(tool_cases)) if tool_cases else None,
        "argument_accuracy": (arg_ok / len(tool_cases)) if tool_cases else None,
        "no_tool_accuracy": (no_tool_ok / len(no_tool_cases)) if no_tool_cases else None,
        "unexpected_error_rate": (unexpected / total_calls) if total_calls else 0.0,
        "avg_rounds": (sum(rounds) / len(rounds)) if rounds else None,
        "queries": len(cases),
        "skipped": False,
        "skip_reason": None,
    }


def _fmt(value) -> str:
    if value is None:
        return "null（skipped）"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(metrics: dict, per_case: list[dict], meta: dict) -> str:
    lines = [
        "# V2.2 skill_use 评测报告",
        "",
        f"- 生成时间：{meta['started_at']} ~ {meta['finished_at']}",
        f"- git sha：`{meta['git_sha']}`（dirty={meta['git_dirty']}）",
        f"- 用例：`eval/suites/skill_use.jsonl` {metrics['queries']} 条（5 工具 × 5 + no-tool 5）",
        f"- 工具后端：web_search=fake；read_lore=真 Retriever（rewrite 关闭）；set_reminder=真库",
        f"- 跳过：{metrics['skipped']}（原因：{metrics['skip_reason']}）",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 值 | 门禁（§4.2） |",
        "|---|---|---|",
        f"| trigger_accuracy | {_fmt(metrics['trigger_accuracy'])} | ≥ 0.85 |",
        f"| argument_accuracy | {_fmt(metrics['argument_accuracy'])} | ≥ 0.80 |",
        f"| no_tool_accuracy | {_fmt(metrics['no_tool_accuracy'])} | ≥ 0.90 |",
        f"| unexpected_error_rate | {_fmt(metrics['unexpected_error_rate'])} | ≤ 0.05 |",
        f"| avg_rounds | {_fmt(metrics['avg_rounds'])} | — |",
        f"| queries | {_fmt(metrics['queries'])} | 30 |",
        "",
        "## 逐用例明细",
        "",
        "| id | 期望 | 触发 | 参数 | 工具调用 | 轮数 | 最终回复（截断） |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in per_case:
        calls = "、".join(
            f"{r['name']}({'ok' if r.get('ok') else r.get('error_type')})"
            for r in row["tool_results"]
        ) or "—"
        expect = row["expect"]
        want = "no_tool" if expect.get("no_tool") else expect.get("tool")
        reply = (row.get("final_reply") or "").replace("\n", " ")[:40]
        lines.append(
            f"| {row['id']} | {want} | {row['trigger_ok']} | "
            f"{row['argument_ok'] if row['argument_ok'] is not None else '—'} | "
            f"{calls} | {row['rounds']} | {reply} |"
        )
    lines += [
        "",
        "> 命令：`uv run python scripts/run_skill_eval.py --agent %d`；"
        "每 case 的 tool_calls / tool_results / final_reply 见同目录 `cases.jsonl`。"
        % meta["agent"],
        "",
    ]
    return "\n".join(lines)


def _write_outputs(out_dir: Path, run_id: str, report: dict, per_case: list[dict], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / f"skill-use-{run_id}.cases.jsonl"
    report_path = out_dir / f"skill-use-{run_id}.json"
    md_path = out_dir / f"skill-use-{run_id}.md"
    cases_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_case) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report["metrics"], per_case, meta), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "command": f"uv run python scripts/run_skill_eval.py --agent {meta['agent']}",
        "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"],
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "env": meta["env"],
        "inputs": {
            "suite_sha256": _sha256_file(SUITE_PATH),
            "fixtures_sha256": _fixtures_sha256(),
        },
        "outputs": {
            "report_sha256": _sha256_file(report_path),
            "ranked_sha256": _sha256_file(cases_path),
        },
    }
    (out_dir / f"skill-use-{run_id}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"report": report_path, "markdown": md_path, "cases": cases_path, "manifest": manifest}


# ---------------------------------------------------------------------------
# 真跑
# ---------------------------------------------------------------------------
async def _create_temp_session(session_factory, agent_id: int) -> int:
    """建一个评测临时会话（`reminders.session_id` 有 FK 到 `sessions.id`）。"""
    from app.db.models import Session as ChatSession

    async with session_factory() as db:
        row = ChatSession(agent_id=agent_id, title="v2.2-skill-eval-temp")
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return int(row.id)


async def _cleanup_temp_session(session_factory, session_id: int) -> None:
    """删临时会话（reminders 有 ON DELETE CASCADE，一并清掉）。"""
    from sqlalchemy import delete

    from app.db.models import Session as ChatSession

    async with session_factory() as db:
        await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
        await db.commit()


async def _run_cases(
    graph, cases: list[dict], *, agent_id: int, session_id: int, run_id: str, executor=None
) -> list[dict]:
    per_case: list[dict] = []
    for case in cases:
        if executor is not None:
            executor.reset()  # 每个 case 一次独立请求（同请求缓存，§2.2-5）
        state_in = {
            "messages": [HumanMessage(content=case["query"])],
            "agent_id": agent_id,
            "session_id": session_id,
            "agent_name": EVAL_NAME,
            "persona_prompt": EVAL_PERSONA,
            "background": EVAL_BACKGROUND,
            "idle_scope": "",
            "reply_max_messages": 0,
            "allowed_skills": list(ALL_SKILLS),
        }
        # 每 case 独立 thread：避免用例之间经工作记忆互相污染
        cfg = {"configurable": {"thread_id": f"{session_id}:{agent_id}:{case['id']}:{run_id}"}}
        result = await graph.ainvoke(state_in, config=cfg)
        tool_results = list(result.get("tool_results") or [])
        verdict = score_case(case, tool_results)
        per_case.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expect": case.get("expect", {}),
                "tool_calls": [
                    {"id": r.get("tool_call_id"), "name": r.get("name"), "arguments": r.get("arguments")}
                    for r in tool_results
                ],
                "tool_results": tool_results,
                "final_reply": str(result.get("reply", "")),
                "rounds": int(result.get("tool_rounds") or 0),
                **verdict,
            }
        )
        print(
            f"[case] {case['id']} 触发={verdict['trigger_ok']} "
            f"参数={verdict['argument_ok']} 调用数={len(tool_results)}",
            flush=True,
        )
    return per_case


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

    def _finish(metrics: dict, per_case: list[dict]) -> int:
        meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report = {
            "schema_version": "1.0",
            "suite": "skill_use",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": meta["git_sha"],
            "metrics": metrics,
            "manifest": None,
        }
        paths = _write_outputs(OUT_DIR, run_id, report, per_case, meta)
        report["manifest"] = str(paths["manifest"])
        print("\n---- skill_use 汇总 ----")
        for key in METRIC_KEYS:
            print(f"{key:<22}: {_fmt(metrics[key])}")
        print(f"报告：{paths['markdown']}")
        print(f"manifest：{paths['manifest']}")
        return 0

    # ---- 跳过条件（§4.3）----
    if not settings.llm_api_key:
        return _finish(_null_metrics("no_llm_key"), [])
    if not settings.test_database_url:
        return _finish(_null_metrics("no_test_database_url"), [])

    from app.llm.base import LLMProvider
    from app.llm.factory import create_embedding_provider, create_llm_provider

    llm = create_llm_provider(settings)
    impl = getattr(type(llm), "generate_with_tools", None)
    if impl is None or impl is getattr(LLMProvider, "generate_with_tools", object()):
        return _finish(_null_metrics("provider_without_tools"), [])

    try:
        embed = create_embedding_provider(settings)
    except Exception as exc:  # noqa: BLE001 - 无嵌入器则无法建 read_lore 检索器
        print(f"[warn] 嵌入器不可用：{exc}", flush=True)
        return _finish(_null_metrics("no_embeddings"), [])

    engine = create_engine(settings.test_sqlalchemy_url)
    session_factory = create_session_factory(engine)
    session_id = 0
    all_degraded = False
    try:
        await lore_eval.ensure_agent(session_factory, args.agent)
        imported = await lore_eval.import_fixtures(session_factory, embed, agent_id=args.agent)
        print(f"[info] 导入 lore chunk 数：{imported}", flush=True)
        session_id = await _create_temp_session(session_factory, args.agent)

        tool_retriever = Retriever(
            session_factory=session_factory,
            embedder=embed,
            reranker=None,          # 工具侧不做 rerank（评测只关心触发/参数）
            rewriter=None,
            rewrite_enabled=False,  # D7：工具内不调 LLM
            budget_tokens=settings.prompt_token_budget,
        )
        registry = build_default_registry(
            search_provider=FakeSearchProvider(FAKE_RESULTS),
            session_factory=session_factory,
            lore_retriever=tool_retriever,
        )
        executor = ToolExecutor(registry, max_tool_rounds=settings.skill_max_tool_rounds)
        store = InMemoryStore(
            index={"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]}
        )
        graph = build_graph(
            llm=llm,
            memory_store=MemoryStore(store),
            checkpointer=InMemorySaver(),
            store=store,
            lore_retriever=None,      # 静态 lore 关闭 → read_lore 工具是设定唯一来源
            lore_enabled=False,
            lore_static_mode="off",
            skill_registry=registry,
            tool_executor=executor,
            allowed_skills=list(ALL_SKILLS),
            max_tool_rounds=settings.skill_max_tool_rounds,
        )
        per_case = await _run_cases(
            graph,
            cases,
            agent_id=args.agent,
            session_id=session_id,
            run_id=run_id,
            executor=executor,
        )
    except Exception as exc:  # noqa: BLE001 - 环境异常不得伪造指标：整体置跳过
        print(f"[warn] skill_use 运行失败，整体置 skipped：{exc}", flush=True)
        all_degraded = True
        per_case = []
    finally:
        if session_id:
            await _cleanup_temp_session(session_factory, session_id)
        await engine.dispose()

    if all_degraded:
        return _finish(_null_metrics("runtime_error"), [])
    return _finish(summarize(cases, per_case), per_case)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.2 skill_use（工具触发/参数/失败）评测")
    ap.add_argument("--agent", type=int, default=1, help="评测角色 id，默认 1")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
