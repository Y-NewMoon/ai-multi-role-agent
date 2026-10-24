"""V2.3 记忆评测（契约 §2.11）：40 条套件 + 5 指标 + keep_both 消融 + manifest。

- 套件：`eval/suites/memory.jsonl` 40 条（conflict 12 / profile 8 / recall 20）；
  `extract` 字段是**预置抽取结果**，评测不调真实 LLM（§2.11）。
- 指标：`conflict_accuracy`（≥0.90）、`profile_hit_rate`（=1.00）、
  `recall_hit@5`（≥0.85）、`recall_hit@5_plain`（纯余弦对照）、`weighted_delta`（≥-0.02）、
  `keep_both_delta`（报告项，不设门禁）。
- 跳过（§4.3）：无 `TEST_DATABASE_URL` → 整体 `skipped=true`、指标 null、退出 0；
  无嵌入器 → 两项 recall 指标 null（`skip_reason=no_embeddings`），其余照跑。
- `--latency-probe`：3 条真实抽取记录 p95（≤2000ms，不计入主指标，结果写进报告）。
- 每个 case 使用独立 namespace（`evalmem-{agent}-{run_id}-{case_id}`），不碰真实角色记忆。

用法：
    uv run python scripts/run_memory_eval.py --agent 1 [--latency-probe]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import report as report_mod  # noqa: E402
from app.config import get_settings  # noqa: E402

SUITE_PATH = ROOT / "eval" / "suites" / "memory.jsonl"
OUT_DIR = ROOT / "eval" / "report"

METRIC_KEYS = (
    "conflict_accuracy",
    "profile_hit_rate",
    "recall_hit@5",
    "recall_hit@5_plain",
    "weighted_delta",
    "keep_both_delta",
)

_PREFIXES = ("用户说过：", "你说过：")

# 延迟探针用的 3 条真实抽取输入（不写 Store，只计时）
_LATENCY_SAMPLES = (
    ("我最近搬到苏州了，家里还养了一只柴犬叫阿福", "苏州好，阿福这名字也热闹。"),
    ("我五月八日生日，记得给我留个靠窗的位子", "记住了，五月八日，靠窗。"),
    ("我喝不了浓茶，一喝就睡不着", "那我给您泡淡一些。"),
)


def _null_metrics(reason: str) -> dict:
    row = {key: None for key in METRIC_KEYS}
    row.update({"skipped": True, "skip_reason": reason, "cases": 0})
    return row


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in SUITE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _db_name(url: str) -> str | None:
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0] or None


def _strip_prefix(text: str) -> str:
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _fact_text(fact: dict) -> str:
    text = str(fact.get("text") or "")
    if text:
        return text
    return f"{fact.get('subject', '')}{fact.get('predicate', '')}={fact.get('object', '')}"


def _norm_pool(items: list[str]) -> set[str]:
    return {_strip_prefix(str(item)) for item in items}


async def _purge(store, agent_id: str) -> None:
    from app.memory.namespace import memory_namespace

    ns = memory_namespace(agent_id)
    for item in await store.asearch(ns, limit=1000):
        await store.adelete(item.namespace, item.key)


async def _run_conflict(store, case: dict, agent_id: str, *, mode: str) -> dict:
    """写 setup → 写预置抽取结果 → 读 active，按 expect 判分。"""
    from app.memory.extractor import ExtractedFact, Extraction
    from app.memory.facts import record_facts
    from app.memory.store import MemoryStore

    await _purge(store, agent_id)
    ms = MemoryStore(store)
    for fact in case.get("setup") or []:
        await ms.put_fact(
            agent_id,
            subject=str(fact["subject"]),
            predicate=str(fact["predicate"]),
            obj=str(fact["object"]),
            importance=float(fact.get("importance", 3)) / 5.0,
            turn_id="eval:setup",
            mode=mode,  # type: ignore[arg-type]
        )
    setup_active = [f["text"] for f in await ms.current_facts(agent_id, limit=200)]

    raw_facts = (case.get("extract") or {}).get("facts") or []
    extraction = Extraction(
        facts=[
            ExtractedFact(
                subject=str(f.get("subject", "")),
                predicate=str(f.get("predicate", "")),
                object=str(f.get("object", "")),
                importance=int(f.get("importance", 3)),
            )
            for f in raw_facts
        ]
    )
    await record_facts(
        ms,
        agent_id,
        extraction,
        source_message_id=None,
        turn_id="eval:turn",
        mode=mode,
    )
    active = [_fact_text(f) for f in await ms.current_facts(agent_id, limit=200)]
    active_set = set(active)
    expect = case.get("expect") or {}
    ok = all(text in active_set for text in expect.get("active") or []) and all(
        text not in active_set for text in expect.get("inactive") or []
    )
    return {
        "id": case["id"],
        "mode": mode,
        "ok": ok,
        "setup_active": setup_active,
        "active": active,
        "inactive": [text for text in active if text in (expect.get("inactive") or [])],
        "expect": expect,
    }


async def _run_profile(store, case: dict, agent_id: str) -> dict:
    from app.memory.extractor import Extraction
    from app.memory.profile import render_profile, update_profile
    from app.memory.store import MemoryStore

    await _purge(store, agent_id)
    ms = MemoryStore(store)
    extraction = Extraction(profile=dict((case.get("extract") or {}).get("profile") or {}))
    await update_profile(ms, agent_id, extraction.profile)
    profile = await ms.get_profile(agent_id)
    rendered = await render_profile(profile)
    expect = dict((case.get("expect") or {}).get("profile") or {})
    ok = all(
        str(profile.get(slot) or "") == str(value)
        and str(value) in rendered
        for slot, value in expect.items()
    )
    return {
        "id": case["id"],
        "ok": ok,
        "expect": expect,
        "profile": profile,
        "rendered": rendered,
    }


async def _run_recall(store, case: dict, agent_id: str) -> dict:
    from app.memory.retriever import retrieve_weighted
    from app.memory.store import MemoryStore

    await _purge(store, agent_id)
    ms = MemoryStore(store)
    now = datetime.now(UTC)
    for memory in case.get("memories") or []:
        age_days = float(memory.get("age_days", 0))
        created_at = (now - timedelta(days=age_days)).isoformat()
        await ms.put(
            agent_id,
            str(memory["text"]),
            role=memory.get("role"),
            importance=float(memory.get("importance", 0.5)),
            created_at=created_at,
        )
    for fact in case.get("setup") or []:
        await ms.put_fact(
            agent_id,
            subject=str(fact["subject"]),
            predicate=str(fact["predicate"]),
            obj=str(fact["object"]),
            importance=float(fact.get("importance", 3)) / 5.0,
            turn_id="eval:setup",
        )

    query = str(case.get("query") or "")
    weighted = await retrieve_weighted(ms, agent_id, query=query, limit=5, now=now)
    plain = await retrieve_weighted(
        ms,
        agent_id,
        query=query,
        limit=5,
        w_cosine=1.0,
        w_recency=0.0,
        w_importance=0.0,
        now=now,
    )
    facts = [_fact_text(f) for f in await ms.current_facts(agent_id, limit=8)]
    # R2：严格口径 = 只看加权召回 top-5；事实另有独立注入通道，单独记 `fact_injection_hit@5`
    weighted_pool = _norm_pool(weighted)
    plain_pool = _norm_pool(plain)
    weighted_with_facts = _norm_pool(weighted + facts)
    plain_with_facts = _norm_pool(plain + facts)
    expect = [str(item) for item in (case.get("expect") or [])]
    return {
        "id": case["id"],
        # R2-A：`tags` 含 `fact` 的 5 条只进 `fact_injection_hit@5` 的分母，
        # 其余 15 条 memory 用例才进严格 `recall_hit@5` 的分母。
        "fact_case": "fact" in (case.get("tags") or []),
        "ok": all(item in weighted_pool for item in expect),
        "ok_with_facts": all(item in weighted_with_facts for item in expect),
        "plain_ok": all(item in plain_pool for item in expect),
        "plain_ok_with_facts": all(item in plain_with_facts for item in expect),
        "expect": expect,
        "recalled_weighted": weighted,
        "recalled_plain": plain,
        "facts": facts,
    }


async def _latency_probe(settings, usage: report_mod.UsageTracker) -> dict:
    """3 条真实抽取的 p95（nearest-rank）；无 LLM key 则 skipped。"""
    if not settings.llm_api_key:
        return {"skipped": True, "skip_reason": "no_llm_key", "samples_ms": [], "p95_ms": None}
    from app.llm.factory import create_llm_provider
    from app.memory.extractor import extract

    llm = usage.wrap(create_llm_provider(settings))
    samples: list[float] = []
    for user_text, reply in _LATENCY_SAMPLES:
        started = time.perf_counter()
        await extract(llm, user_text, reply, timeout_s=settings.memory_extract_timeout_s)
        samples.append(round((time.perf_counter() - started) * 1000, 1))
    ordered = sorted(samples)
    rank = max(1, int(round(0.95 * len(ordered) + 0.4999)))
    return {
        "skipped": False,
        "skip_reason": None,
        "samples_ms": samples,
        "p95_ms": ordered[min(rank, len(ordered)) - 1],
        "threshold_ms": 2000,
        "passed": ordered[min(rank, len(ordered)) - 1] <= 2000,
    }


def summarize(per_case: list[dict], *, recall_available: bool) -> dict:
    def _rate(rows: list[dict], key: str = "ok") -> float | None:
        return (sum(1 for row in rows if row.get(key)) / len(rows)) if rows else None

    conflict = [row for row in per_case if row["kind"] == "conflict"]
    conflict_kb = [row for row in per_case if row["kind"] == "conflict_keep_both"]
    profile = [row for row in per_case if row["kind"] == "profile"]
    recall = [row for row in per_case if row["kind"] == "recall"]
    # R2-A（所有者批准）：15 条 memory 用例只算严格召回，5 条 fact 用例只算事实注入
    recall_mem = [row for row in recall if not row.get("fact_case")]
    recall_fact = [row for row in recall if row.get("fact_case")]

    recall_hit = _rate(recall_mem, "ok") if recall_available else None
    recall_plain = _rate(recall_mem, "plain_ok") if recall_available else None
    fact_hit = _rate(recall_fact, "ok_with_facts") if recall_available else None
    fact_plain = _rate(recall_fact, "plain_ok_with_facts") if recall_available else None
    keep_both_delta = None
    if conflict and conflict_kb:
        by_id = {row["id"]: row for row in conflict}
        keep_both_delta = sum(
            len(row["active"]) - len(by_id[row["id"]]["active"]) for row in conflict_kb
        )
    return {
        "conflict_accuracy": _rate(conflict),
        "profile_hit_rate": _rate(profile),
        # R2：严格 top-5（只含加权召回，不含事实注入通道）
        "recall_hit@5": recall_hit,
        "recall_hit@5_plain": recall_plain,
        "weighted_delta": (
            round(recall_hit - recall_plain, 6)
            if recall_hit is not None and recall_plain is not None
            else None
        ),
        # 事实（current_facts）作为独立注入通道的命中率，与上面严格口径并列披露
        "fact_injection_hit@5": fact_hit,
        "fact_injection_hit@5_plain": fact_plain,
        "recall_denominator": {"memory": len(recall_mem), "fact": len(recall_fact)},
        "keep_both_delta": keep_both_delta,
    }


def _fmt(value) -> str:
    if value is None:
        return "null（skipped）"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(metrics: dict, per_case: list[dict], meta: dict) -> str:
    lines = [
        "# V2.3 记忆评测报告",
        "",
        f"- 生成时间：{meta['started_at']} ~ {meta['finished_at']}",
        f"- git sha：`{meta['git_sha']}`（dirty={meta['git_dirty']}）",
        f"- 套件：`eval/suites/memory.jsonl` 40 条（conflict 12 / profile 8 / recall 20）",
        f"- 存储：`{meta['storage']}`；嵌入：`{meta['embedding']}`",
        f"- 跳过：{metrics.get('skipped')}（原因：{metrics.get('skip_reason')}）",
        "- 说明：`extract` 为预置抽取结果，主流程 0 次 LLM 调用。",
        "- **口径（R2-A，所有者批准）**：`recall_hit@5` = 15 条 memory 用例中，期望记忆进入"
        "**加权召回 top-5**（严格，不含事实通道）的比例；`fact_injection_hit@5` = 5 条 fact 用例中，"
        "期望事实出现在「加权召回 top-5 ∪ `current_facts(8)`」的比例（事实在 compose_prompt 里另有独立注入通道）。"
        "两者分母不重叠，门禁只看严格值。",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 值 | 门禁（§4.2） |",
        "|---|---|---|",
        f"| conflict_accuracy | {_fmt(metrics.get('conflict_accuracy'))} | ≥ 0.90 |",
        f"| profile_hit_rate | {_fmt(metrics.get('profile_hit_rate'))} | = 1.00 |",
        f"| recall_hit@5 | {_fmt(metrics.get('recall_hit@5'))} | ≥ 0.85（15 条 memory） |",
        f"| recall_hit@5_plain | {_fmt(metrics.get('recall_hit@5_plain'))} | 对照项（15 条 memory） |",
        f"| weighted_delta | {_fmt(metrics.get('weighted_delta'))} | ≥ -0.02（15 条 memory） |",
        f"| fact_injection_hit@5 | {_fmt(metrics.get('fact_injection_hit@5'))} | = 1.00（5 条 fact，披露项） |",
        f"| fact_injection_hit@5_plain | {_fmt(metrics.get('fact_injection_hit@5_plain'))} | 披露项（5 条 fact） |",
        f"| keep_both_delta | {_fmt(metrics.get('keep_both_delta'))} | 报告项 |",
        "",
    ]
    probe = meta.get("latency_probe") or {}
    lines += [
        "## 抽取延迟探针（--latency-probe）",
        "",
        f"- skipped：{probe.get('skipped', True)}（{probe.get('skip_reason')}）",
        f"- 样本：{probe.get('samples_ms')}",
        f"- p95：{_fmt(probe.get('p95_ms'))} ms（阈值 2000 ms，"
        f"{'通过' if probe.get('passed') else '未通过/未测'}）",
        "",
        "## 逐用例明细",
        "",
        "| id | kind | ok | 期望 | 实际 |",
        "|---|---|---|---|---|",
    ]
    for row in per_case:
        if row["kind"] == "conflict":
            actual = f"active={row['active']}"
        elif row["kind"] == "conflict_keep_both":
            actual = f"active={row['active']}"
        elif row["kind"] == "profile":
            actual = f"profile={row['profile']}"
        else:
            actual = (
                f"weighted={row['recalled_weighted']} / plain={row['recalled_plain']} "
                f"/ with_facts_ok={row.get('ok_with_facts')}"
            )
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['ok']} | "
            f"{json.dumps(row['expect'], ensure_ascii=False)} | {actual} |"
        )
    lines += ["", "> 命令：`uv run python scripts/run_memory_eval.py --agent " + str(meta["agent"]) + "`", ""]
    return "\n".join(lines)


def _write_outputs(
    run_id: str, report: dict, per_case: list[dict], meta: dict
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"memory-{run_id}.json"
    md_path = OUT_DIR / f"memory-{run_id}.md"
    cases_path = OUT_DIR / f"memory-{run_id}.cases.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_case) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(report["metrics"], per_case, {**meta, "latency_probe": report["latency_probe"]}),
        encoding="utf-8",
    )
    manifest = report_mod.build_manifest(
        run_id=run_id,
        command="uv run python scripts/run_memory_eval.py --agent "
        + str(meta["agent"])
        + (" --latency-probe" if meta["latency_probe_enabled"] else ""),
        started_at=meta["started_at"],
        finished_at=meta["finished_at"],
        env=meta["env"],
        inputs={"suite_sha256": _sha256_file(SUITE_PATH), "fixtures_sha256": None},
        outputs={
            "report_sha256": _sha256_file(report_path),
            "cases_sha256": _sha256_file(cases_path),
        },
        usage=report["usage"],
        budget_section="4.6#11" if meta["latency_probe_enabled"] else "4.6#5",
        budget_prompt_tokens=10_000 if meta["latency_probe_enabled"] else 0,
    )
    manifest_path = report_mod.write_manifest(manifest, OUT_DIR / f"memory-{run_id}.manifest.json")
    return {"report": report_path, "markdown": md_path, "cases": cases_path, "manifest": manifest_path}


async def _make_store(settings):
    """返回 (store, storage_label, embed_label, pool, embed_ok)。"""
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    embed = None
    embed_label = "unavailable"
    try:
        from app.llm.factory import create_embedding_provider

        embed = create_embedding_provider(settings)
        embed_label = f"local:{settings.embedding_model}"
    except Exception as exc:  # noqa: BLE001 - 无嵌入器时只跳过 recall 两项
        print(f"[warn] 嵌入器不可用，recall 指标置 null：{exc}", flush=True)

    pool = AsyncConnectionPool(
        settings.test_database_url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    await pool.wait(timeout=5)
    from langgraph.store.postgres.aio import AsyncPostgresStore

    index = (
        {"dims": settings.embedding_dims, "embed": embed, "fields": ["text"]}
        if embed is not None
        else None
    )
    store = AsyncPostgresStore(pool, index=index)
    await store.setup()
    db = _db_name(settings.test_database_url) or "unknown"
    return store, f"postgres:{db}", embed_label, pool, embed is not None


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    usage = report_mod.UsageTracker()
    meta = {
        "agent": args.agent,
        "started_at": started_at,
        "finished_at": started_at,
        "git_sha": report_mod._git_sha(),
        "git_dirty": report_mod._git_dirty(),
        "storage": "skipped",
        "embedding": "skipped",
        "latency_probe_enabled": args.latency_probe,
        "env": {
            "python": sys.version.split()[0],
            "database": _db_name(settings.test_database_url),
            "llm_model": None,
            "embedding_model": settings.embedding_model,
            "rerank_model": None,
        },
    }

    def _finish(metrics: dict, per_case: list[dict], latency: dict, *, storage: str, embedding: str) -> int:
        meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        meta["storage"] = storage
        meta["embedding"] = embedding
        report = {
            "schema_version": "1.0",
            "suite": "memory",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": meta["git_sha"],
            "agent": args.agent,
            "metrics": metrics,
            "latency_probe": latency,
            "usage": usage.snapshot(),
        }
        paths = _write_outputs(run_id, report, per_case, meta)
        print("\n---- memory 汇总 ----")
        for key in METRIC_KEYS:
            print(f"{key:<22}: {_fmt(metrics[key])}")
        print(f"latency p95           : {_fmt(latency.get('p95_ms'))} ms")
        print(f"报告：{paths['markdown']}")
        print(f"manifest：{paths['manifest']}")
        print(
            "usage                 : llm_calls={} prompt_tokens≈{} completion_tokens≈{}".format(
                report["usage"]["llm_calls"],
                report["usage"]["prompt_tokens"],
                report["usage"]["completion_tokens"],
            )
        )
        return 0

    # ---- 跳过条件（§4.3）----
    if not settings.test_database_url:
        return _finish(_null_metrics("no_test_database_url"), [], {"skipped": True, "skip_reason": "no_test_database_url"}, storage="skipped", embedding="skipped")

    try:
        store, storage, embed_label, pool, embed_ok = await _make_store(settings)
    except Exception as exc:  # noqa: BLE001 - 库不可用不得伪造指标
        print(f"[warn] 测试库不可用，整体 skipped：{exc}", flush=True)
        return _finish(_null_metrics("database_unavailable"), [], {"skipped": True, "skip_reason": "database_unavailable"}, storage="unavailable", embedding="unavailable")

    meta["env"]["embedding_model"] = settings.embedding_model if embed_ok else None
    cases = load_cases()
    per_case: list[dict] = []
    try:
        for case in cases:
            kind = case["kind"]
            base_agent = f"evalmem-{args.agent}-{run_id}-{case['id']}"
            try:
                if kind == "conflict":
                    per_case.append({**await _run_conflict(store, case, base_agent, mode="overwrite"), "kind": kind})
                    per_case.append(
                        {
                            **await _run_conflict(store, case, base_agent + "-kb", mode="keep_both_with_note"),
                            "kind": "conflict_keep_both",
                        }
                    )
                elif kind == "profile":
                    per_case.append({**await _run_profile(store, case, base_agent), "kind": kind})
                else:
                    per_case.append({**await _run_recall(store, case, base_agent), "kind": kind})
            except Exception as exc:  # noqa: BLE001 - 单条失败计入分母（ok=False）
                per_case.append(
                    {
                        "id": case["id"],
                        "kind": kind if kind != "conflict" else "conflict",
                        "ok": False,
                        "plain_ok": False,
                        "expect": case.get("expect", {}),
                        "error": str(exc),
                        "active": [],
                        "recalled_weighted": [],
                        "recalled_plain": [],
                    }
                )
                print(f"[case] {case['id']} 失败：{exc}", flush=True)

        latency = (
            await _latency_probe(settings, usage)
            if args.latency_probe
            else {"skipped": True, "skip_reason": "not_requested", "samples_ms": [], "p95_ms": None}
        )
        metrics = summarize(per_case, recall_available=embed_ok)
        metrics.update(
            {
                "skipped": False,
                "skip_reason": None if embed_ok else "no_embeddings",
                "cases": len(cases),
            }
        )
    finally:
        await pool.close()

    return _finish(metrics, per_case, latency, storage=storage, embedding=embed_label)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2.3 记忆评测（40 条 + 5 指标 + keep_both 消融）")
    ap.add_argument("--agent", type=int, default=1, help="评测基准角色 id（每个 case 仍用独立 namespace）")
    ap.add_argument("--latency-probe", action="store_true", help="额外跑 3 条真实抽取记录 p95")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
