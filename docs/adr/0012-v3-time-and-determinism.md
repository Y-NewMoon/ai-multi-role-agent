# ADR-0012：时间感全量化，以及“同 now 逐字稳定”的确定性口径

- 状态：**已接受**（V3 M1 · 时间感全量化）
- 时点：2026-10-25
- 关联：V3 M1 交接文档 §2.1–§2.6、§2.8；`doc/待修清单.md` F-01；
  ADR-0003（消息真相源与 SSE 时序）、ADR-0006（事实记忆存 Store）
- 影响范围：`src/app/core/clock.py`（新增）、`src/app/agents/nodes/compose_prompt.py`、
  `src/app/agents/state.py`、`src/app/services/chat_service.py`、`src/app/services/group_service.py`、
  `src/app/agents/supervisor.py`、`src/app/memory/summarizer.py`、`src/app/prompts/summarize.py`、
  `src/app/config.py`、`src/app/observability/trace.py`、`src/app/main.py`、
  `scripts/snapshot_prompt.py`、`scripts/run_long_session.py`、`eval/run_eval.py`

---

## 1. Context

V2.x 的 prompt 不变量是“**逐字一致**”：给定同一 state，`compose_prompt` 的输出必须与历史快照
完全相同。它让 V2.0–V2.6 的回归测试非常硬，也把“prompt 改动”变成可审查的 diff。

但这条不变量有一个前提：**prompt 里没有“现在”**。V3 M1 的 F-01 要求所有角色都知道
“现在几点”“你们上次说话是多久前”，历史消息还要带时间戳。只要注入真实时钟：

- 每次调用的 `now` 都在变，**逐字一致在数学上不再成立**；
- 快照测试如果依赖真实时间，会在分钟边界偶发失败（9:59 → 10:00 就换了文案）；
- 旧基线（V2.x 的 prompt 样本、token 曲线）与新版本不可直接比较。

因此必须显式打破旧不变量，并给出一个**可执行的替代口径**，而不是悄悄放宽断言。

## 2. Decision

### 2.1 “现在”收敛到唯一入口

新增 `src/app/core/clock.py`：

```text
now() / now_iso() / parse_fixed_now(raw) / validate_fixed_now()
```

优先级：进程环境变量 `EVAL_FIXED_NOW` → `settings.eval_fixed_now`（`.env` 可选）→ 真实时间。
业务代码（服务层 `at`、`last_user_at`、`compose_prompt`）只允许经此入口取时间；
`main.py` 启动时校验一次，非法值**启动即失败**，不把坏配置带进请求。

### 2.2 新不变量：同 `now` + 同 state + 同配置 → prompt 逐字稳定

旧不变量被替换为：

> 给定同一 `now`（显式传入或由 `EVAL_FIXED_NOW` 冻结）、同一 state、同一配置，
> `compose_prompt` 的输出逐字相等；换 `now` 必须变。

评测/测试固定时钟统一用 `EVAL_FIXED_NOW`（M1 证据全部取 `2026-10-25T10:35:00+08:00`），
所有精确快照断言必须显式传 `now=`，不得依赖真实时钟。

### 2.3 注入范围有边界

M1 只注入两样东西：`now` 与 `last_user_at`（上一条 **user** 消息时间）。

- 时间块对所有角色常驻（`base → time → lore → …`），**不参与预算裁剪**；
- 消息前缀只给带 `at` 的 `HumanMessage`；`AIMessage` 不带（留到确有需要时再立验收）；
- 不引入随机数、不打乱其余段顺序、不改时区（固定 `Asia/Shanghai`）；
- `EVAL_FIXED_NOW` 只是评测可复现的手段，不是业务配置。

## 3. Consequences

### 3.1 代价

- **旧基线不可直接比**：V2.x 的 prompt 快照全部失效；旧 token 曲线只能“记录增量 + 解释”，
  不能拿来宣称行为回归。跨版本评测必须标 `prompt_version`（`manifest.env`）。
- **快照测试全部要带 `now`**：任何断言 system 全文或比较两次输出的用例，必须显式传固定 `now`；
  否则分钟边界会偶发失败。
- **每轮 token 增加**：时间块约 70–90 token；消息前缀约 50–100 token（11 条消息量级）。
  长会话会更早触顶，按现有顺序先裁 lore/摘要，不裁时间。
- **每轮多一次 SELECT**：取 `last_user_at`；查询失败 fail-soft（warning + `None`），不阻断本轮。

### 3.2 收益

- 修复 F-01：所有角色（含无工具、空闲轮）都有时间锚点与上次说话间隔；
- 评测可复现：同一 `EVAL_FIXED_NOW` 下 prompt 与基线可逐字核对；
- 旧数据兼容：没有 `at` 的旧消息不加前缀，不迁移、不回填。

### 3.3 基线口径

`eval/baseline/v3-m1.json` 是**口径基线**（golden CI + 固定时钟 + 版本号），
不是模型行为基线；`v3-m1-longsession.json` 只记录 token 曲线。
M1 默认 0 真实 LLM，scenario/adversarial 的 sample-only 口径沿用 V2.5，不重跑。

## 4. 什么时候重审

- M2/M3 改动 prompt 组成或召回策略时；
- 真实使用发现“时间说法仍错”（例如跨时区、相对时间换算、上次说话语义偏旧）时；
- 需要给 `AIMessage` 也打时间戳（当前明确不做，需另立验收）；
- 需要多时区支持（当前固定 `Asia/Shanghai`，不做配置化）。
