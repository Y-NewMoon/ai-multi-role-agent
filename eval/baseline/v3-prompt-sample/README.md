# V3 M1 prompt 样本（after）

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
