"""报告渲染与基线对比（契约 4.5）：JSON + Markdown 双格式，键名逐字固定。

产物：`eval/report/eval-YYYYmmdd-HHMMSS.json` + 同名 `.md`。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 单一真相源（契约 §1.4）：rubric 版本号只由 `judge.RUBRIC_VERSION` 定义，
# `report` 不得再硬编码，避免 r1/r2 两处漂移。
from judge import RUBRIC_VERSION as _RUBRIC_VERSION

SCHEMA_VERSION = "1.1"
SUITE_VERSION = "v2.0.0"

# 契约 4.5 的顶层键（顺序即文档顺序，审查脚本按名比对）
TOP_LEVEL_KEYS = (
    "schema_version",
    "generated_at",
    "git_sha",
    "suite_version",
    "split",
    "storage",
    "llm_enabled",
    "judge",
    "metrics",
    "baseline",
    "failures",
    # V2.3（§1.4）：只新增 usage 统计，指标定义/门禁/加载逻辑不变
    "usage",
)

METRIC_KEYS = {
    "golden": ("isolation_leaks", "memory_hit_rate", "forbid_violations"),
    "scenario": ("persona_pass", "boundary_pass"),
    "adversarial": (
        "refusal_rate",
        "persona_intact_rate",
        # V2.2（契约 §2.11）：新增 expect 消费元信息；旧两键保留不动
        "expect_consumed",
        "source_counts",
    ),
    "retrieval": ("hit@1", "hit@3", "mrr", "ndcg@10"),
    # V2.1 新增（契约 4.10）：前四项为数值指标，其余四项为运行元信息
    "retrieval_lore": (
        "hit@1",
        "hit@3",
        "mrr",
        "ndcg@10",
        "queries",
        "rerank",
        "skipped",
        "skip_reason",
    ),
    "system": (
        "p50_first_token_ms",
        "p95_first_token_ms",
        "tokens_per_turn",
        "cost_cny_per_turn",
    ),
}


def empty_metrics() -> dict:
    """全 null 的指标树（--no-llm 时未跑的套件保持 null，不得写 0/1）。"""
    return {
        suite: {key: None for key in keys} for suite, keys in METRIC_KEYS.items()
    }


def new_report(*, split: str, storage: str, llm_enabled: bool, judge_model: str | None) -> dict:
    """构造契约 4.5 结构的空报告。"""
    report = {key: None for key in TOP_LEVEL_KEYS}
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "suite_version": SUITE_VERSION,
            "split": split,
            "storage": storage,
            "llm_enabled": llm_enabled,
            "judge": {
                # judge 固定配置：temperature=0（契约 4.3 硬要求 1），是否真正跑 judge 看 llm_enabled
                "model": judge_model,
                "rubric_version": _RUBRIC_VERSION,
                "temperature": 0,
            },
            "metrics": empty_metrics(),
            "baseline": {"path": None, "delta": {}},
            "failures": [],
            "usage": {},
        }
    )
    return report


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 非 git 环境也要能出报告
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# V2.3 usage（§1.4 / §4.6 / §8.3）：只统计用量，不改指标与门禁
# ---------------------------------------------------------------------------
def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def estimate_tokens(text: str) -> int:
    """与 compose_prompt 同口径的保守估算（CJK 1 字/token，其余 4 字/token）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + (len(text) - cjk + 3) // 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


class UsageTracker:
    """统计 LLM 调用与 token 估算；`wrap(llm)` 返回透明代理。

    口径（§4.6）：只统计 deepseek-chat 输入 token；completion 单列，不并入预算。
    第三方 tokenizer 不可用，故用字符启发式估算（§5 记录偏差）。
    """

    def __init__(self) -> None:
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.judge_cache_hits = 0
        self.judge_cache_misses = 0

    def add_prompt(self, messages: list[dict]) -> None:
        self.prompt_tokens += estimate_messages_tokens(messages)

    def add_completion(self, text: str) -> None:
        self.completion_tokens += estimate_tokens(text or "")

    def wrap(self, llm):
        return _TrackedLLM(llm, self)

    def merge_cache_stats(self, stats: dict) -> None:
        self.judge_cache_hits += int(stats.get("hits", 0))
        self.judge_cache_misses += int(stats.get("misses", 0))

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "judge_cache_hits": self.judge_cache_hits,
            "judge_cache_misses": self.judge_cache_misses,
            # 估算口径（不引第三方 tokenizer，见交接文档 §0.3 / §5 偏差）
            "estimator": "cjk1_other4",
        }


class _TrackedLLM:
    """透明代理：只统计，不改变 provider 的任何语义。"""

    def __init__(self, llm, usage: UsageTracker) -> None:
        self._llm = llm
        self._usage = usage

    def __getattr__(self, item):
        return getattr(self._llm, item)

    async def generate(self, messages: list[dict]) -> str:
        self._usage.llm_calls += 1
        self._usage.add_prompt(messages)
        text = await self._llm.generate(messages)
        self._usage.add_completion(text)
        return text

    async def astream(self, messages: list[dict]):
        self._usage.llm_calls += 1
        self._usage.add_prompt(messages)
        parts: list[str] = []
        async for chunk in self._llm.astream(messages):
            parts.append(str(chunk))
            yield chunk
        self._usage.add_completion("".join(parts))

    async def generate_with_tools(self, messages: list[dict], tools: list[dict], **kwargs):
        self._usage.llm_calls += 1
        self._usage.add_prompt(messages)
        reply = await self._llm.generate_with_tools(messages, tools, **kwargs)
        self._usage.add_completion(getattr(reply, "content", "") or "")
        return reply


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_manifest(
    *,
    run_id: str,
    command: str,
    started_at: str,
    finished_at: str,
    env: dict,
    inputs: dict,
    outputs: dict,
    usage: dict,
    budget_section: str,
    budget_prompt_tokens: int,
) -> dict:
    """§8.3 manifest：usage 的 within_budget 由脚本自身比较得出，不得手写。"""
    usage = dict(usage)
    usage["budget_prompt_tokens"] = int(budget_prompt_tokens)
    usage["within_budget"] = int(usage.get("prompt_tokens") or 0) <= int(budget_prompt_tokens)
    return {
        "run_id": run_id,
        "command": command,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "started_at": started_at,
        "finished_at": finished_at,
        "env": env,
        "inputs": inputs,
        "outputs": outputs,
        "usage": usage,
        "budget_section": budget_section,
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_baseline(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"基线报告不存在：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def compute_delta(metrics: dict, baseline_metrics: dict) -> dict:
    """逐键算 差值 = 本次 - 基线；任一侧为 null / 布尔 / 非数字则为 null。"""
    delta: dict = {}
    for suite, keys in METRIC_KEYS.items():
        row: dict = {}
        for key in keys:
            new_v = metrics.get(suite, {}).get(key)
            base_v = baseline_metrics.get(suite, {}).get(key)
            if isinstance(new_v, bool) or isinstance(base_v, bool):
                row[key] = None  # rerank/skipped 等布尔元信息不作差
            elif isinstance(new_v, (int, float)) and isinstance(base_v, (int, float)):
                row[key] = round(new_v - base_v, 6)
            else:
                row[key] = None
        delta[suite] = row
    return delta


def apply_baseline(report: dict, baseline_path: str | Path) -> dict:
    """把 baseline.path / delta 写进报告（契约 4.5 的 baseline 键）。"""
    baseline = load_baseline(baseline_path)
    report["baseline"] = {
        "path": str(baseline_path),
        "delta": compute_delta(report["metrics"], baseline.get("metrics", {})),
    }
    return report


def _fmt(value) -> str:
    if value is None:
        return "null（skipped）"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(report: dict, run_env: dict | None = None) -> str:
    """Markdown 报告：指标汇总表（含基线差值列）+ 失败清单 + 运行环境。

    P0 可观测性（审查 §13.2）：judge 降级计数写进「运行环境」段；降级占比 > 5%
    时在顶部打显式警告（只告警，不进 §4.7 门禁），不新增/改动 §4.5 的报告键。
    """
    run_env = run_env or {}
    degraded = run_env.get("judge_degraded") or 0
    judged = run_env.get("judge_judged") or 0

    lines = ["# V2.0 评测报告", ""]
    if judged > 0 and degraded / judged > 0.05:
        lines += [
            f"> **警告**：judge 降级 {degraded}/{judged}（{degraded / judged:.0%}）超过 5%，"
            "本轮 scenario/adversarial 结论偏保守（降级一律计 fail），建议复核。",
            "",
        ]
    lines += [
        f"- 生成时间：{report['generated_at']}",
        f"- git sha：`{report['git_sha']}`",
        f"- suite 版本：{report['suite_version']}",
        "",
        "## 运行环境",
        "",
        f"- storage：`{report['storage']}`",
        f"- llm_enabled：{report['llm_enabled']}",
        f"- split：`{report['split']}`（dev:test = {run_env.get('split_counts', 'n/a')}）",
    ]
    if degraded > 0:
        lines.append(
            f"- judge 降级：{degraded}/{judged}"
            "（解析失败按保守取向计 fail，已计入分母；来源见失败清单的 `[词表兜底]` / `[judge 降级-未判定]`）"
        )
    if run_env.get("embedding"):
        lines.append(f"- embedding：`{run_env['embedding']}`")
    usage = report.get("usage") or {}
    if usage:
        lines.append(
            "- usage：llm_calls={} / prompt_tokens≈{} / completion_tokens≈{} / "
            "judge 缓存命中 {} / 未命中 {}".format(
                usage.get("llm_calls"),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("judge_cache_hits"),
                usage.get("judge_cache_misses"),
            )
        )
    if report["llm_enabled"]:
        judge = report["judge"]
        lines.append(
            f"- judge：model=`{judge['model']}` rubric=`{judge['rubric_version']}` "
            f"temperature={judge['temperature']}"
        )
    else:
        lines.append("- judge：skipped（--no-llm，scenario/adversarial 指标为 null）")

    has_delta = bool(report.get("baseline", {}).get("path"))
    delta = report.get("baseline", {}).get("delta", {})

    lines += ["", "## 指标汇总", ""]
    if has_delta:
        lines += [
            f"基线：`{report['baseline']['path']}`",
            "",
            "| 套件 | 指标 | 本次 | 基线差值 |",
            "|---|---|---|---|",
        ]
    else:
        lines += ["| 套件 | 指标 | 本次 | 基线差值 |", "|---|---|---|---|"]
    for suite, keys in METRIC_KEYS.items():
        for key in keys:
            value = report["metrics"].get(suite, {}).get(key)
            d = delta.get(suite, {}).get(key)
            d_text = "-" if not has_delta else _fmt(d)
            lines.append(f"| {suite} | {key} | {_fmt(value)} | {d_text} |")

    failures = report.get("failures", [])
    lines += ["", "## 失败用例清单", ""]
    if not failures:
        lines.append("（无）")
    else:
        shown = failures[:20]
        for f in shown:
            lines.append(f"- [{f['suite']}] `{f['id']}`：{f['reason']}")
        if len(failures) > len(shown):
            lines.append(f"- …… 其余 {len(failures) - len(shown)} 条见 JSON 报告")
        lines += ["", f"失败总数：{len(failures)}"]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path, run_env: dict | None = None) -> tuple[Path, Path]:
    """写出 eval-YYYYmmdd-HHMMSS.json + 同名 .md，返回两个路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out / f"eval-{ts}.json"
    md_path = out / f"eval-{ts}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report, run_env), encoding="utf-8")
    return json_path, md_path
