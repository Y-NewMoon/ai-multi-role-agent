"""冻结一份 `compose_prompt` 的完整输出，作为 V3 M1（注入 now）之前的对照证据。

纯离线：只构造固定的 state 并调用 `compose_prompt`，不读库、不连库、不调 LLM、不出网。
产物写到 `eval/baseline/v2.5-prompt-sample/`：
    prompt.json  返回值原样（含 prompt 数组）
    prompt.txt   可读版（system 在前，逐条标 role）
    meta.json    日期、git sha/dirty、prompt_version、关键配置、token 估算、两个产物的 sha256
    README.md    用途与重新生成说明

用法（仓库根目录）：
    uv run python scripts/snapshot_prompt.py

V3 M1 之后输出会永久改变；除非明确要更新对照，否则不要覆盖本目录。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "eval" / "baseline" / "v2.5-prompt-sample"
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app.agents.nodes.compose_prompt import _estimate_tokens, compose_prompt  # noqa: E402
from app.core import clock  # noqa: E402

SYSTEM_PROMPT = (
    "你是「清和」，一位在城南经营旧书店的年轻店主，今年二十九岁。"
    "你说话温和、语速偏慢，习惯先听对方把话说完，再轻轻接一句。"
    "你喜欢用天气、茶和旧书作比喻，从不催促客人，也不替人做决定。"
    "你记得每位熟客的偏好，但不会主动打探隐私；对方不想说的事，你只当没听见。"
    "遇到不懂的话题，你会坦率承认「这个我不太懂」，然后把话头带回书、茶或街坊的日常。"
    "你偶尔引用一句旧诗，但只引你确实记得的句子，记不清就说记不清。"
    "你不喜欢热闹的推销，也不评价客人的选择；在你看来，人愿意坐下来喝杯茶，就已经很好了。"
)

MESSAGES = [
    HumanMessage(content="今天下班晚了，路过你店门口看见灯还亮着。"),
    AIMessage(content="灯是给你留的。\n进来坐会儿吧，水刚烧开。"),
    HumanMessage(content="那我要一杯上次说的陈皮普洱，还加两块桂花糕。"),
    AIMessage(content="记得。\n糕是今天新做的，还温着。"),
    HumanMessage(content="最近总睡不好，脑子里像有好多书页在翻。"),
    AIMessage(content="那就先别急着合上。\n我陪你坐一会儿，等它自己慢下来。"),
    HumanMessage(content="你这家店开了多久？感觉每条街你都认识。"),
    AIMessage(content="第七年了。\n街口修伞的老周、卖花的阿婆，都算老邻居。"),
    HumanMessage(content="下次能不能帮我留一本讲老城区的旧书？不着急的。"),
    AIMessage(content="可以。\n我周三去旧货市场，看到合适的就替你收着。"),
    HumanMessage(content="那就说定了，周三之后我再来，顺便还你这把伞。"),
    AIMessage(content="好。\n伞不急，你人来了就行。"),
]

STATE = {
    "system_prompt": SYSTEM_PROMPT,
    "messages": MESSAGES,
    "summary": (
        "客人最近工作忙、睡眠浅，已来过店里三次；喜欢陈皮普洱和桂花糕；"
        "托店主周三去旧货市场找一本讲老城区的旧书，并说会顺便还伞。"
    ),
    "recalled_memories": [
        "客人说过睡前翻书反而更清醒，后来改成听雨声。",
        "客人不喜欢太甜的茶点，桂花糕只吃两块。",
    ],
    "recalled_facts": [
        {
            "subject": "客人",
            "predicate": "喜欢的茶",
            "object": "陈皮普洱",
            "text": "客人喜欢陈皮普洱",
            "importance": 0.9,
            "updated_at": "2026-10-20T21:00:00",
        },
        {
            "subject": "客人",
            "predicate": "睡眠",
            "object": "容易失眠",
            "text": "客人最近容易失眠",
            "importance": 0.7,
            "updated_at": "2026-10-22T22:10:00",
        },
    ],
    "user_profile": {"称呼": "阿棠", "关系阶段": "熟客"},
    "l2_portrait": (
        "你们认识两年多。阿棠第一次来是为了躲雨，后来成了每周三的常客。"
        "她知道你从不问私事，你也知道她加班后会绕路来看看灯。"
    ),
    "recalled_lore": [
        {
            "source_ref": "old-town:01",
            "header_path": "老城区/街巷",
            "content": "清和的书店在城南青石巷尽头，门口有一盏常年不灭的暖黄灯。",
            "score": 0.91,
            "routes": ["always_on"],
        },
        {
            "source_ref": "old-town:02",
            "header_path": "老城区/旧货市场",
            "content": "周三清晨的旧货市场在东门桥下，摊位多卖旧书、旧钟表和缺角的瓷碗。",
            "score": 0.78,
            "routes": ["keyword", "vector"],
        },
    ],
    "allowed_skills": [],
}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings():
    """只读配置（含 .env）；不建连接、不调 LLM。失败则回落到环境变量/默认值。"""
    try:
        from app.config import get_settings

        return get_settings()
    except Exception:  # noqa: BLE001 - 离线快照不应因缺 .env 失败
        return None


def _render_txt(prompt: list[dict]) -> str:
    blocks = []
    for index, message in enumerate(prompt):
        role = message.get("role", "")
        content = str(message.get("content") or "")
        label = "system" if index == 0 and role == "system" else f"#{index} {role}"
        blocks.append(f"[{label}]\n{content}")
    return "\n\n".join(blocks) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="冻结一份 compose_prompt 输出（不连库、不调 LLM）")
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="产物目录（默认 eval/baseline/v2.5-prompt-sample）",
    )
    ap.add_argument(
        "--fixed-now",
        default=None,
        help="冻结时钟（ISO 8601；等价设置 EVAL_FIXED_NOW）",
    )
    return ap.parse_args(argv)


async def main() -> None:
    args = _parse_args(sys.argv[1:])
    if args.fixed_now:
        # 校验后写进进程环境变量；compose_prompt 的时间块走 clock.now()
        clock.parse_fixed_now(args.fixed_now)
        os.environ[clock.ENV_KEY] = args.fixed_now
    out_dir = Path(args.out_dir)

    result = await compose_prompt(
        STATE, token_budget=8000, recent_rounds=10, skill_registry=None
    )
    prompt = result["prompt"]

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_json = out_dir / "prompt.json"
    prompt_txt = out_dir / "prompt.txt"
    meta_json = out_dir / "meta.json"
    readme = out_dir / "README.md"

    prompt_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prompt_txt.write_text(_render_txt(prompt), encoding="utf-8")

    system_tokens = _estimate_tokens(str(prompt[0].get("content") or ""))
    total_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in prompt)
    dirty = bool(_git("status", "--porcelain"))
    settings = _settings()

    def _cfg(name: str, default):
        return getattr(settings, name, default) if settings is not None else default

    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD") or None,
        "git_dirty": dirty,
        "prompt_version_env": os.environ.get("PROMPT_VERSION"),
        "prompt_version": os.environ.get("PROMPT_VERSION") or _cfg("prompt_version", "v2.6"),
        "eval_fixed_now": os.environ.get(clock.ENV_KEY),
        "config": {
            "prompt_token_budget": _cfg("prompt_token_budget", 8000),
            "prompt_recent_rounds": _cfg("prompt_recent_rounds", 10),
            "compose_token_budget": 8000,
            "compose_recent_rounds": 10,
            "summary_trigger_rounds": _cfg("summary_trigger_rounds", 10),
            "llm_model": _cfg("llm_model", os.environ.get("LLM_MODEL") or "deepseek-chat"),
            "embedding_model": _cfg(
                "embedding_model", os.environ.get("EMBEDDING_MODEL") or "bge-m3"
            ),
            "embedding_dims": _cfg(
                "embedding_dims", int(os.environ.get("EMBEDDING_DIMS") or 1024)
            ),
            "allowed_skills": [],
        },
        "tokens": {
            "estimator": "_estimate_tokens（CJK 1 字/token，其余 4 字/token）",
            "estimated_tokens": total_tokens,
            "system_tokens": system_tokens,
            "messages_tokens": total_tokens - system_tokens,
            "prompt_messages": len(prompt),
        },
        "files": {
            "prompt.json": _sha256(prompt_json),
            "prompt.txt": _sha256(prompt_txt),
        },
    }
    meta_json.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not readme.exists():
        readme.write_text(_README_V3 if args.fixed_now else _README, encoding="utf-8")

    print(f"prompt.json / prompt.txt / meta.json 已写入 {out_dir}")
    print(
        f"estimated_tokens={total_tokens} "
        f"(system={system_tokens}, messages={total_tokens - system_tokens}, "
        f"messages_count={len(prompt) - 1})"
    )


_README_V3 = """# V3 M1 prompt 样本（after）

## 这是什么

一份**固定 state** + **固定时钟**（`EVAL_FIXED_NOW=2026-10-25T10:35:00+08:00`）经
`src/app/agents/nodes/compose_prompt.py::compose_prompt` 拼装后的完整输出。
state 完全离线构造，不读数据库、不调用 LLM、不出网。

- `prompt.json`：`compose_prompt` 返回值原样（`{"prompt": [...]}`）。
- `prompt.txt`：同一内容的可读版，system 在前，逐条 message 标 role。
- `meta.json`：日期、git sha/dirty、`prompt_version=v3`、`eval_fixed_now`、关键配置、
  token 估算与两个产物的 sha256。

## before / after

- **before**：`eval/baseline/v2.5-prompt-sample/`（V3 之前的对照，原样保留，不得覆盖）。
- **after**：本目录。与 before 的差异只有两类：system 中 `base` 之后多出【当前时间】三行；
  消息 content 多出 `[MM-DD HH:mm]` / `[HH:mm]` 前缀（本样本的固定 state 消息**没有** `at`，
  因此消息行没有前缀——这是有意的，用来只体现时间块差异）。

## 如何重新生成

```powershell
uv run python scripts/snapshot_prompt.py --out-dir eval/baseline/v3-prompt-sample `
  --fixed-now 2026-10-25T10:35:00+08:00
```

固定 `now` + 固定 state 下输出逐字稳定；换 `now` 必然改变 `prompt.txt` 与哈希。
"""

_README = """# V2.5 prompt 样本（冻结）

## 这是什么

一份**固定 state** 经 `src/app/agents/nodes/compose_prompt.py::compose_prompt` 拼装后的完整输出，
用于日后回答「V3 到底改了什么」。state 完全离线构造，不读数据库、不调用 LLM、不出网。

- `prompt.json`：`compose_prompt` 返回值原样（`{"prompt": [...]}`）。
- `prompt.txt`：同一内容的可读版，system 在前，逐条 message 标 role。
- `meta.json`：日期、git sha/dirty、`PROMPT_VERSION`、关键配置、token 估算与两个产物的 sha256。

## 为什么冻结

V3 M1 会给 prompt 注入当前时间（`now`），注入点一旦上线，这份输出就**无法再逐字复现**。
因此必须在 V3 改动前落盘一份基线样本；它只作对照证据，**不是评测基线**，不参与任何门禁。

## V3 M1 之后会变

V3 M1 之后 `prompt.txt` 会出现时间相关段落（具体位置/措辞以 V3 实现为准），
`meta.json.files` 的 sha256 也会随之改变。**变化本身是预期的**，不要为了让哈希一致而改脚本。

## 如何重新生成

```powershell
uv run python scripts/snapshot_prompt.py
```

脚本幂等：`prompt.json` / `prompt.txt` 的字节内容只由固定 state 和当前 `compose_prompt` 决定；
重复运行只有 `meta.json` 的 `generated_at` / `git_dirty` 等元信息会变化。

## 不要随手覆盖

- 本目录是 **V2.5 时点的历史证据**。V3 开发期间不要重新生成后提交覆盖。
- 如需新基线，另建 `eval/baseline/v3-prompt-sample/`，保留本目录。
- 不要改 `compose_prompt.py` 来「修正」本样本；脚本只读调用，不 patch 业务代码。
"""

if __name__ == "__main__":
    asyncio.run(main())
