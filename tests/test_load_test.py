"""压测脚本与 fake LLM 单测（V2.4 §2.10）：解析 / 统计 / 不跑网络。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import load_test  # noqa: E402

from app.llm.fake import FakeLLMProvider  # noqa: E402


def test_percentile_nearest_rank():
    assert load_test.percentile([], 50) is None
    assert load_test.percentile([10], 95) == 10
    assert load_test.percentile([1, 2, 3, 4], 50) == 2
    assert load_test.percentile([1, 2, 3, 4], 95) == 4


def test_sse_parser_frames():
    parser = load_test.SseParser()
    lines = [
        "event: token", 'data: {"text": "你"}', "",
        "event: token", 'data: {"text": "好"}', "",
        "event: done", 'data: {"reply": "你好"}', "",
    ]
    frames = [f for line in lines if (f := parser.feed(line)) is not None]
    assert frames == [
        ("token", {"text": "你"}),
        ("token", {"text": "好"}),
        ("done", {"reply": "你好"}),
    ]


def test_sse_parser_ignores_blank_without_event():
    parser = load_test.SseParser()
    assert parser.feed("") is None


def test_sse_parser_bad_json():
    parser = load_test.SseParser()
    assert parser.feed("event: error") is None
    assert parser.feed("data: not-json") is None
    assert parser.feed("") == ("error", {"raw": "not-json"})


def test_summarize_metrics_and_zero_usage():
    results = [
        {"ok": True, "status": 200, "first_token_ms": 10, "total_ms": 20, "error": None},
        {"ok": True, "status": 200, "first_token_ms": 30, "total_ms": 40, "error": None},
        {"ok": False, "status": 500, "first_token_ms": None, "total_ms": 50,
         "error": "HTTP 500"},
        {"ok": False, "status": 200, "first_token_ms": 99, "total_ms": 100,
         "error": "timeout"},
    ]
    report = load_test.summarize(
        results, run_id="r", base_url="http://x", llm_provider="fake",
        concurrency=2, rounds=2, wall_ms=1000, fake_calls=2,
    )
    assert report["requests"] == 4 and report["errors"] == 2
    assert report["error_rate"] == 0.5
    assert report["p50_first_token_ms"] == 30
    assert report["p95_first_token_ms"] == 99
    assert report["p50_total_ms"] == 40
    assert report["p95_total_ms"] == 100
    assert report["throughput_req_s"] == 4.0
    # R9：fake_llm_calls 是 /health 差值，不是“成功请求数”（成功请求为 2 只是巧合）
    assert report["fake_llm_calls"] == 2
    assert report["fake_llm_calls_source"] == "health_delta"
    # C4：真实 usage 必须为 0
    assert report["usage"] == {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    assert report["budget_section"] == "4.6#4"


def test_parse_session_ids_dedupes():
    assert load_test._parse_session_ids(" 1, 2,2,3 ") == [1, 2, 3]


async def test_fake_llm_provider_no_network():
    provider = FakeLLMProvider(delay_ms=0, reply="abcd", chunk_chars=2)
    assert await provider.generate([]) == "abcd"
    chunks = [chunk async for chunk in provider.astream([])]
    assert chunks == ["ab", "cd"]
    reply = await provider.generate_with_tools([], [])
    assert reply.content == "abcd" and reply.tool_calls == []
    assert provider.calls == 3


def _manifest(**overrides) -> dict:
    base = load_test._build_manifest(
        run_id="r", command="cmd", started_at="t0", finished_at="t1",
        report_sha=None, llm_provider="fake", skipped=False,
        health={
            "database": "roleplay_test",
            "embedding_provider": "fake",
            "lore_enabled": False,
            "trace_enabled": True,
            "trace_dir": "data/traces/loadtest-r",
            "trace_sample_rate": 0.05,
        },
        session_ids=[1, 2],
    )
    base.update(overrides)
    return base


def test_manifest_records_server_db_and_sessions():
    manifest = _manifest()
    assert manifest["env"]["database"] == "roleplay_test"
    assert manifest["env"]["embedding_provider"] == "fake"
    assert manifest["env"]["lore_enabled"] is False
    assert manifest["inputs"]["sessions"] == [1, 2]
    assert manifest["inputs"]["seed_script"] == "scripts/seed_loadtest.py"
    assert manifest["outputs"]["cases_sha256"] is None


def test_write_outputs_manifest_hash(tmp_path):
    out = tmp_path / "loadtest-r.json"
    report = load_test.summarize(
        [], run_id="r", base_url="http://x", llm_provider="fake",
        concurrency=1, rounds=1, wall_ms=None, fake_calls=None,
    )
    assert report["fake_llm_calls"] is None
    assert report["fake_llm_calls_source"] == "unavailable"
    manifest = _manifest()
    cases = [{"session_id": 1, "round_no": 1, "ok": True,
              "status": 200, "first_token_ms": 5, "total_ms": 9, "error": None}]
    md_path, manifest_path = load_test._write_outputs(
        out_path=out, report=report, manifest=manifest, skipped=False, cases=cases
    )
    assert out.exists() and md_path.exists() and manifest_path.exists()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["outputs"]["report_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    cases_path = out.with_suffix(".cases.jsonl")
    assert cases_path.exists()
    assert stored["outputs"]["cases_sha256"] == hashlib.sha256(cases_path.read_bytes()).hexdigest()
    assert json.loads(cases_path.read_text(encoding="utf-8").strip())["session_id"] == 1
    assert stored["usage"]["llm_calls"] == 0 and stored["usage"]["prompt_tokens"] == 0
    assert stored["budget_section"] == "4.6#4"
