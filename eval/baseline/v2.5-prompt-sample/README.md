# V2.5 prompt 样本（冻结）

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
