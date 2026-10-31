# AI Agent Windows 安装 Runbook（给 DSH 执行）

> **这份文档写给本地 agent（DSH / Codex / Claude Code 等），不是普通 README。**
> 人类是审批者：涉及 UAC、重启、BIOS、API Key、付款的动作，agent 必须停下来让人操作。
>
> **目标版本**：V3 M2（记忆接口）——`TARGET_COMMIT: TBD`（截至 2026-10-28）
> **发送前必须把 `TBD` 换成 M2 push 后的实际 commit SHA，并用 `git rev-parse HEAD` 校验。**
> 公开仓库当前 `main` 还是 V2.6；如果 clone 到的版本没有 `src/app/api/v1/memory.py`，说明版本不对，立即停止。
>
> 适用系统：Windows 10 22H2+ / Windows 11，x64，有管理员权限。
> 预计耗时：45–120 分钟（取决于网络；模型约 2.2GB，依赖约 2.5GB）。
> 磁盘：至少 15GB 空闲；内存：建议 16GB（8GB 可跑，但加载模型/rerank 会吃力）。

**目录**：§0 规则 · §1 标准 · §2 预检 · §3 依赖 · §4 代码 · §5 `.env` · §6 数据库 · §7 Python 依赖 · §8 迁移 · §9 模型 · §10 启动 · §11 验收 · §12 故障 · §13 停启 · §14 安全 · §15 报告 · 附录 A/B/C

---

## 0. 给 DSH 的硬规则（先读，违反即停止）

1. **先只读检查，再给人类一份计划**；人类确认后再开始安装。
2. **系统级安装、UAC、重启、BIOS、驱动、Docker Desktop 登录**：停下来让人类操作。不要尝试绕过权限。
3. **API Key 只在 `.env` 里**：不要打印、不要回显、不要写进日志、不要发到聊天。优先让人类自己编辑 `.env`。
4. **禁止**：关闭 Defender/防火墙、改 hosts、安装来历不明的“修复工具”、使用 `--force`/`--no-verify`、删除 `data/`、`docker compose down -v`、`git reset --hard`。
5. **同一条命令失败两次就停止**，报告：命令、完整错误关键行、已尝试的方案、下一步建议。不要随机换方案。
6. 每步只做一件事，报告“命令 + 结果 + 下一步”；长日志只贴关键 20 行。
7. 任何“模型名 / 版本 / 端口 / 数据库”与本文不一致时，先停下核对，不要猜。
8. 需要人类操作时，明确说：“请你在普通 PowerShell 里执行下面这条命令，然后把输出贴回来。”

---

## 1. 元信息与成功标准

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/Y-NewMoon/ai-multi-role-agent` |
| 目标版本 | **M2**，`TARGET_COMMIT = TBD` |
| 默认分支 | `main`（M2 push 后） |
| 后端 | FastAPI + LangGraph，`127.0.0.1:8000` |
| 前端 | Vite + React，`localhost:5173` |
| 数据库 | PostgreSQL 16 + pgvector，Docker 容器 `roleplay-db` |
| 嵌入模型 | `BAAI/bge-m3`，约 2.2GB |
| LLM | DeepSeek OpenAI 兼容接口，模型名 `deepseek-flash` |
| 记忆接口（M2） | `/api/v1/memory/` 下的 `turn` / `context` / `forget` / `state` |

**安装成功的标准（全部满足才算通过）**：

- [ ] `git rev-parse HEAD` 等于 `TARGET_COMMIT`
- [ ] `docker compose ps` 中 `roleplay-db` 正在运行
- [ ] `uv run alembic current` 显示 `0006_skills_reminders (head)`
- [ ] bge-m3 缓存存在且约 2.2GB
- [ ] `GET /health` 返回 `status=ok`、`llm_provider=deepseek`、`database=roleplay`
- [ ] 真实发一条消息有回复（不是 401/402/400）
- [ ] 重启后端后，会话消息还在
- [ ] M2：`POST /api/v1/memory/turn` 返回 `ok=true`，`GET /api/v1/memory/state` 能读到 buffer
- [ ] 前端 `http://localhost:5173` 能打开并收到流式回复
- [ ] 后端窗口没有 traceback

---

## 2. 前置检查（只读）

在**普通 PowerShell**（不需要管理员）执行。
如果 DSH 的沙箱报 `permission denied` / `only core types` / 无法访问 Docker 命名管道，停下来让人类在普通 PowerShell 执行同一段命令，再把输出贴回来；不要尝试提权或改安全设置。

```powershell
$ErrorActionPreference = 'Continue'

Write-Host '--- OS ---'
[Environment]::Is64BitOperatingSystem
(Get-CimInstance Win32_OperatingSystem).Caption
(Get-CimInstance Win32_OperatingSystem).Version

Write-Host '--- CPU / RAM ---'
(Get-CimInstance Win32_Processor).Name
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

Write-Host '--- Disk C ---'
Get-PSDrive C | Select-Object @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}

Write-Host '--- Virtualization ---'
(Get-CimInstance Win32_ComputerSystem).HypervisorPresent
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled

Write-Host '--- Admin ---'
([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host '--- Tools ---'
foreach ($cmd in 'git','docker','uv','node','npm','wsl') {
  $c = Get-Command $cmd -ErrorAction SilentlyContinue
  if ($c) { "${cmd}: $($c.Source)" } else { "${cmd}: MISSING" }
}

Write-Host '--- Versions ---'
git --version
docker --version
docker compose version
uv --version
node -v
npm -v
wsl --status

Write-Host '--- Ports ---'
Get-NetTCPConnection -LocalPort 5432,8000,5173 -State Listen -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,OwningProcess

Write-Host '--- Network ---'
Test-NetConnection api.deepseek.com -Port 443 | Select-Object ComputerName,TcpTestSucceeded
Test-NetConnection github.com -Port 443 | Select-Object ComputerName,TcpTestSucceeded
```

**判定规则**：

| 检查 | 通过 | 不通过怎么办 |
|---|---|---|
| OS | Windows 10 22H2+ / 11，x64 | 停止；旧系统不保证 Docker/torch 可用 |
| RAM | ≥16GB 推荐 | 8GB 可继续，但记录风险；关闭浏览器/其他大程序 |
| 磁盘 C: | ≥15GB 空闲 | 停止，清理磁盘后再来 |
| Hypervisor / 虚拟化 | 任一为 True | 若都为 False：进 BIOS 开 VT-x/AMD-V；agent 不能代做 |
| 管理员 | True | Docker 安装需要管理员；让人类开“以管理员身份运行”的 PowerShell |
| Docker | 已安装且 `docker info` 成功 | 见 §3 |
| uv / Node / git | 已安装且版本达标 | 见 §3 |
| Node | 20.19+ 或 22.12+（建议 22/24 LTS） | 升级；**Node 18 不够**（Vite 8 要求 Node 20.19+ 或 22.12+） |
| 端口 5432/8000/5173 | 未被占用 | 见 §6 / §12 |
| `api.deepseek.com:443` | True | 检查代理/网络；不能访问就不要继续真实聊天 |
| `github.com:443` | True | 可用 ZIP 下载替代；或配置代理 |

---

## 3. 安装系统依赖

> 以下命令默认使用 `winget`。如果 `winget` 不存在，先装“应用安装程序”（App Installer），或改用每节给出的官方安装包。
> `winget` 返回 `already installed` 不算失败，继续验证版本即可。
> 安装 Docker Desktop 可能需要管理员 + 重启；**不要静默硬装**，让人类确认。

### 3.1 Git

```powershell
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
```

装完**重开 PowerShell**，确认：

```powershell
git --version
```

没有 winget：<https://git-scm.com/download/win>

### 3.2 Docker Desktop + WSL2

```powershell
winget install --id Docker.DockerDesktop -e --source winget --accept-package-agreements --accept-source-agreements
```

没有 winget：<https://www.docker.com/products/docker-desktop/>

安装后：

1. 让人类启动 **Docker Desktop**，接受许可，等鲸鱼图标稳定。
2. 如果提示 WSL2 不完整：

```powershell
wsl --status
wsl --update
wsl --set-default-version 2
```

3. 如果 `wsl --install` 要求重启，**停止并让人类重启**；重启后重开 PowerShell，继续。
4. 验证：

```powershell
docker info
docker compose version
```

期望：`docker info` 能打印 `Server Version`，不报 `permission denied`。
如果报 `permission denied ... dockerDesktopLinuxEngine`：先确认 Docker Desktop 已启动；仍失败则让人类在普通 PowerShell 跑 `docker info`，把输出贴回来（agent 沙箱可能拦截 Docker 命名管道）。少数机器需要注销/重启，让 `docker-users` 组生效。

### 3.3 uv（Python 包管理器）

```powershell
winget install --id astral-sh.uv -e --source winget --accept-package-agreements --accept-source-agreements
```

装完重开 PowerShell：

```powershell
uv --version
```

没有 winget：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3.4 Node.js（前端）

```powershell
winget install --id OpenJS.NodeJS.LTS -e --source winget --accept-package-agreements --accept-source-agreements
```

装完重开 PowerShell，确认：

```powershell
node -v
npm -v
```

**必须满足 Vite 8：Node 20.19+ 或 22.12+；建议直接用 22 LTS/24 LTS。** 如果仍是 v18 或低于 20.19：

- 先 `winget upgrade --id OpenJS.NodeJS.LTS`；
- 仍不行就让人类卸载旧 Node 或调整 PATH；
- 不要用 `npm install --force` 绕过 Vite 的 engine 检查。

没有 winget：<https://nodejs.org/>

### 3.5 Python（不装系统 Python）

```powershell
uv python install 3.12
```

项目要求 `>=3.12`；用 3.12 是为了避开不必要的 wheel 兼容问题。**不要用系统 Python 直接跑项目。**

### 3.6 可选：VC++ 运行库

如果后面 torch/PostgreSQL 报 `MSVCP140.dll` / `VCRUNTIME140.dll` 缺失：

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget --accept-package-agreements --accept-source-agreements
```

---

## 4. 获取代码并校验版本

**路径要求**：`C:\ai-agent`，不要放桌面、OneDrive、中文/带空格路径。
如果当前用户不能在 `C:\` 根目录创建文件夹，改用 `$env:USERPROFILE\ai-agent`（例如 `C:\Users\你\ai-agent`），后续所有 `C:\ai-agent` 同步替换。

```powershell
if (Test-Path C:\ai-agent) {
  Write-Host 'C:\ai-agent 已存在。不要覆盖/删除；让人类确认是否换目录。'
  throw 'APP_DIR_EXISTS'
}

$TargetCommit = 'TBD'   # FILL: 把 TBD 换成 M2 push 后的 commit SHA（7 位或 40 位）
if ($TargetCommit -eq 'TBD') { throw 'TARGET_COMMIT_NOT_SET' }

git clone https://github.com/Y-NewMoon/ai-multi-role-agent.git C:\ai-agent
Set-Location C:\ai-agent
git fetch --all --tags
git checkout $TargetCommit

$targetFull = (git rev-parse --verify $TargetCommit).Trim()
$head = (git rev-parse HEAD).Trim()
if ($head -ne $targetFull) { throw "HEAD_MISMATCH: $head != $targetFull" }
git status -sb
```

把 `TBD` 换成文档顶部的实际 SHA。期望：

- `git rev-parse HEAD` 与目标完全一致；
- `git status -sb` 显示 clean（没有未提交改动）；
- `Test-Path src\app\api\v1\memory.py` 为 `True`（M2 才有）。

如果 `git checkout` 报 `did not match any file(s) known to git`，说明 SHA 写错或没 fetch 到；停止，不要改用 `main` 猜。

```powershell
if (-not (Test-Path src\app\api\v1\memory.py)) {
  throw 'NOT_M2：当前代码没有 memory 接口，版本不对，停止。'
}
```

如果 GitHub 访问失败：让人类用浏览器下载 ZIP，解压到 `C:\ai-agent`，再继续；但**无法用 git 校验 commit**，必须在报告里注明“ZIP 安装，未校验 SHA”。
ZIP 路径下跳过 `git clone` / `git checkout` / `git rev-parse`，解压后直接做 `Test-Path src\app\api\v1\memory.py` 检查，并在报告里写明。

---

## 5. 创建 `.env`

```powershell
if (Test-Path .env) {
  Write-Host '.env 已存在：不要覆盖，直接编辑；确认 LLM_API_KEY 与 LLM_MODEL 正确。'
} else {
  Copy-Item .env.example .env
}
notepad .env
```

让人类编辑，至少保证以下键存在（其余保持默认）：

```env
DATABASE_URL=postgresql://roleplay:roleplay@localhost:5432/roleplay
TEST_DATABASE_URL=postgresql://roleplay:roleplay@localhost:5432/roleplay_test

LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-flash

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMS=1024

MEMORY_API_AGENTS=ext
```

**关键点**：

1. **`LLM_MODEL` 不要用 `deepseek-chat`。** DeepSeek 官方更新日志写明旧模型名 `deepseek-chat` / `deepseek-reasoner` 于 **2026-07-24 停止使用**；当前模型名是 `deepseek-flash`（或更贵的 `deepseek-v4-pro`）。
   - 如果 `.env.example` 里还是 `deepseek-chat`，手动改掉。
   - 如果平台报 `Model Not Exist` / 400，先核对 `.env`，不要怀疑网络。
   - `deepseek-flash` 默认开启思考模式，首 token 可能比旧 `deepseek-chat` 慢，属正常现象；不要因此认为安装失败。
2. `LLM_API_KEY` 为空时 [start.ps1](scripts/start.ps1) 会直接退出；不要用 `LLM_PROVIDER=fake` 冒充真实聊天。
3. `MEMORY_API_AGENTS=ext` 是 M2 白名单；不配或配空，`/api/v1/memory/*` 会 403。
4. `.env` 已在 `.gitignore`；不要让 agent 把内容打印出来。

让 agent 做“只检查不泄露”的校验：

```powershell
$line = Select-String -Path .env -Pattern '^LLM_API_KEY=(.*)$' | Select-Object -First 1
$key = if ($line) { $line.Matches[0].Groups[1].Value.Trim() } else { '' }
if ($key.Length -lt 20) { Write-Host 'KEY_MISSING_OR_TOO_SHORT' } else { Write-Host 'KEY_PRESENT' }
```

**如何获取 Key（给人类看）**：

1. 打开 <https://platform.deepseek.com/>，注册/登录（按平台要求实名）。
2. 充值 10–20 元即可试很久（支付宝/微信；以平台页面为准）。
3. 打开 <https://platform.deepseek.com/api_keys> 创建 key，复制一次。
4. 粘贴到 `.env` 的 `LLM_API_KEY`，不要在聊天里贴。
5. 当前 `deepseek-flash` 价格约：输入 1–2 元/百万 token（闲时/峰时，缓存未命中），输出 4–8 元/百万；本项目一轮约 0.005–0.008 元，1000 轮约 5–8 元。以 [官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) 为准。

如果人类没有国内手机号/支付方式：DeepSeek 可能不方便；可改用其他 OpenAI 兼容服务（硅基流动、Kimi、智谱、DashScope、OpenRouter 等），但模型名/价格/Base URL 不同，需要人类提供。此时 `.env` 用：

```env
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://厂商的/v1
LLM_MODEL=厂商支持的模型名
```

**没有 key 时只能做 UI 演示（回复是假的，不是真实聊天）**：

```env
LLM_PROVIDER=fake
EMBEDDING_PROVIDER=fake
LORE_ENABLED=false
LLM_API_KEY=fake-placeholder
```

这样不需要 §9 的 bge-m3 下载，也不需要真实 key；但不要把它当成“装好了”。
注意：[start.ps1](scripts/start.ps1) 会检查 `LLM_API_KEY` 非空，所以 fake 模式也要填一个占位值（`fake-placeholder` 即可，fake provider 不会用它）。
想再省 2.5GB，可在 `uv sync` 前设 `$env:UV_NO_GROUP='embeddings'`（只适合 fake UI 演示；真实聊天不要这样做）。

---

## 6. 启动数据库

```powershell
docker compose up -d
docker compose ps
```

等待数据库就绪（最多 60 秒）：

```powershell
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
  docker compose exec -T db pg_isready -U roleplay -d roleplay *> $null
  if ($LASTEXITCODE -eq 0) { $ready = $true; break }
  Start-Sleep -Seconds 1
}
if ($ready) { Write-Host 'DB_READY' } else { Write-Host 'DB_NOT_READY'; docker compose logs --tail 50 db }
```

期望：`roleplay-db` 正在运行，`DB_READY`。

**失败分支**：

| 现象 | 处理 |
|---|---|
| `port is already allocated` / 5432 被占用 | 改 `docker-compose.yml` 的 `ports` 为 `"5433:5432"`，同时把 `.env` 两个 URL 的端口改成 5433；重新 `docker compose up -d` |
| `pull access denied` / timeout | Docker Hub 被墙或代理问题；让人类在 Docker Desktop 配 registry mirror 或代理；不要连续重试 |
| `permission denied ... dockerDesktopLinuxEngine` | Docker Desktop 没启动；启动后重试；agent 沙箱拦管道就让人类手动跑 |
| `no configuration file provided` | cwd 不对；`Set-Location C:\ai-agent` |
| `pg_isready` 一直失败 | `docker compose logs db`；确认镜像 `pgvector/pgvector:pg16` 拉取成功 |

---

## 7. 安装 Python 依赖

```powershell
uv venv --python 3.12
uv sync
```

期望：生成 `.venv\Scripts\python.exe`，安装 FastAPI、LangGraph、torch、sentence-transformers 等。约 2.5GB，10–30 分钟。

**注意**：

- 如果环境变量里有 `UV_NO_GROUP=embeddings`，先移除；否则 torch/sentence-transformers 不会安装，§9 会失败。
- 如果 `.venv` 已存在且 Python 不是 3.12：先问人类，再删除 `.venv` 重来；不要直接删。
- 如果 PyPI 超时/SSL：

```powershell
$env:UV_DEFAULT_INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple'
uv sync
```

- 校验：

```powershell
.venv\Scripts\python.exe -c "import torch, sentence_transformers, fastapi; print('DEPS_OK', torch.__version__)"
```

期望：`DEPS_OK ...`。

---

## 8. 数据库迁移 + 种子角色

```powershell
uv run alembic upgrade head
uv run alembic current
uv run python scripts/seed_personas.py
```

期望：

- `alembic current` 最后一行是 `0006_skills_reminders (head)`；
- seed 输出 `新增 6：...`（重复执行会变成 `更新 6`，正常）。

**失败分支**：

| 现象 | 处理 |
|---|---|
| `connection refused` | 数据库没起或 URL 端口不对；回 §6 |
| `extension "vector" is not available` / `CREATE EXTENSION vector` 失败 | 镜像不对，不是 `pgvector/pgvector:pg16`；停止，不要用原生 PostgreSQL 硬顶 |
| `password authentication failed` | `.env` URL 与 compose 的 user/password 不一致；默认 `roleplay/roleplay` |
| `Target database is not up to date` | 先 `uv run alembic upgrade head`，再 current |

---

## 9. 首次下载 bge-m3（最容易漏的一步）

项目 [src/app/llm/embedding.py](src/app/llm/embedding.py) 默认设置 `HF_HUB_OFFLINE=1`。**干净机器没有缓存时，直接启动后端会失败**；README 里“首次启动会下载”在默认配置下不成立。

先在同一 PowerShell 会话里：

```powershell
$env:HF_HUB_OFFLINE = '0'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HUB_DISABLE_XET = '1'

uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3'); print('BGE_M3_READY')"
```

期望：下载约 2.2GB，最后打印 `BGE_M3_READY`。10–40 分钟，取决于网络。

**校验缓存**：

```powershell
$hfHome = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $env:USERPROFILE '.cache\huggingface' }
$cache = Join-Path $hfHome 'hub\models--BAAI--bge-m3'
if (Test-Path $cache) {
  $gb = (Get-ChildItem $cache -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
  Write-Host ("CACHE_GB={0:N2}" -f $gb)
} else {
  Write-Host 'CACHE_MISSING'
}
```

期望 `CACHE_GB≈2.2`（HF 缓存可能因 blobs/snapshots 重复计数显示更大，属正常）。

**失败分支**：

| 现象 | 处理 |
|---|---|
| `LocalEntryNotFoundError` / `OSError: We couldn't connect` | 还是离线模式或模型未缓存；确认 `$env:HF_HUB_OFFLINE='0'`，重跑 |
| 下载卡住/超时 | 保留已下载部分，重跑同一条命令会续传；换 `HF_ENDPOINT` 或让人类配代理 |
| `ModuleNotFoundError: sentence_transformers` | `uv sync` 没装 embeddings 组；回 §7 重跑 `uv sync` |
| 磁盘不足 | 停止，清理到至少 15GB；不要删 HF 缓存里的其他模型 |

下载完成后，可以保持 `HF_HUB_OFFLINE=1`（更快、不联网校验）；但要在**同一个会话**里设置，或让人类设用户环境变量。

---

## 10. 启动后端与前端

推荐用项目自带脚本（它会起 Docker、跑迁移、起后端、起前端、开浏览器）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

或双击 `start.bat`（如果模型缓存已存在，双击也能用；但首次下载必须按 §9 先做）。

启动后确认：

```powershell
$health = Invoke-RestMethod http://127.0.0.1:8000/health
$health | ConvertTo-Json -Depth 5
```

期望：

```json
{
  "status": "ok",
  "llm_provider": "deepseek",
  "database": "roleplay",
  "embedding_provider": "local",
  "embedding_model": "bge-m3"
}
```

如果 `start.ps1` 失败，改用手动方式（两个 PowerShell 窗口）：

```powershell
# 窗口 1：后端
Set-Location C:\ai-agent
uv run python scripts/run_server.py

# 窗口 2：前端
Set-Location C:\ai-agent\frontend
npm ci
npm run dev
```

**不要**直接用 `uvicorn app.main:app`：Windows 上 psycopg 异步需要 `SelectorEventLoop`，项目的 `scripts/run_server.py` 已处理。

前端：浏览器打开 <http://localhost:5173>。如果空白/报错：

- 看窗口 2 的 Vite 日志；
- `node -v` 必须满足 20.19+ 或 22.12+（建议 22 LTS）；
- 后端必须先在 8000 端口健康。

---

## 11. 验收测试

### 11.1 角色列表

```powershell
$agents = Invoke-RestMethod http://127.0.0.1:8000/api/v1/agents
$agents | Select-Object id,name | Format-Table
```

期望：6 个种子角色。

### 11.2 真实聊天（非流式）

```powershell
$agentId = @($agents)[0].id

$sessionBody = @{ agent_id = $agentId; title = 'setup-test' } | ConvertTo-Json
$session = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/sessions `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($sessionBody))

$chatBody = @{ session_id = $session.id; message = '你好，用一句话介绍你自己。' } | ConvertTo-Json
$chat = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/chat `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($chatBody))

Write-Host "SESSION_ID=$($session.id) AGENT_ID=$($chat.agent_id)"
$chat.reply
```

期望：`reply` 非空，中文回复，`SESSION_ID` 有值。

**错误码**：

| 错误 | 原因 |
|---|---|
| 401 | key 无效/复制错/带空格 |
| 402 | 余额不足 |
| 400 `Model Not Exist` | `LLM_MODEL` 还是 `deepseek-chat`；改 `deepseek-flash` |
| 500 + `UpstreamError` | 上游网络/模型问题；看后端窗口日志 |
| 404 | `session_id` 不存在；用上面创建的 `$session.id` |

### 11.3 前端流式

打开 <http://localhost:5173>，选角色，发“你好”，确认逐字出现。若前端能开但发消息报错，看浏览器 Network/Console 和后端窗口。

### 11.4 重启持久化

1. 关闭后端窗口（或 `.\scripts\stop.ps1`）。
2. 重新 `.\scripts\start.ps1`。
3. 验证：

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sessions/$($session.id)/messages" | ConvertTo-Json -Depth 5
```

期望：能看到刚才的用户消息和角色回复。

> 已知限制：V2.6 前端刷新后不加载历史（页面内存态），不要误判为“数据库没存”。用上面的 API 验证。

### 11.5 M2 记忆接口（仅当 `memory.py` 存在）

```powershell
$at = (Get-Date).ToString('o')
$turnBody = @{
  agent = 'ext'
  session = 'main'
  user_text = '我叫阿明，喜欢喝普洱。'
  replies = @('记住了，阿明喜欢普洱。')
  at = $at
  idempotency_key = 'setup-test-1'
} | ConvertTo-Json -Depth 5

$turn = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/memory/turn `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($turnBody))
$turn | ConvertTo-Json -Depth 5
```

期望：`ok=true`，`rounds>=1`，`buffer_size>=2`。

```powershell
$ctxBody = @{ agent='ext'; session='main'; query='阿明喜欢喝什么' } | ConvertTo-Json
$ctx = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/memory/context `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($ctxBody))
$ctx | ConvertTo-Json -Depth 5
```

期望：`ok=true`。`facts` / `memories` / `summary` / `profile` 至少一个非空是期望；全空不一定代表接口坏（抽取/召回可能未命中），但要在报告里记录。

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/memory/state?agent=ext&session=main' | ConvertTo-Json
```

期望：`buffer_size`、`entries`、`facts`、`l2_merges` 等字段存在；`buffer.json` 在 `data/memory_api/` 下。

**注意**：

- `MEMORY_API_AGENTS` 不包含 `ext` → 403；
- `/memory/*` 只绑定 `127.0.0.1`，无鉴权，不要暴露公网；
- 11 轮摘要测试会额外消耗真实 token；M2 的 pytest 已覆盖，不要为验收强行跑 11 轮。

---

## 12. 故障树（按现象查）

| 现象 | 先查 | 处理 |
|---|---|---|
| `docker: command not found` | PATH / 安装 | 重开 PowerShell；仍无则装 Docker Desktop |
| `permission denied ... dockerDesktopLinuxEngine` | Docker Desktop 是否启动 | 启动并等 `docker info`；agent 沙箱拦截就让人类手动跑 |
| `wsl --status` 报错 / WSL2 缺失 | Windows 功能 | `wsl --install`，重启，`wsl --update` |
| Docker 提示虚拟化未开启 | BIOS | 开 VT-x/AMD-V；停止，让人类操作 |
| `docker compose up` 端口占用 | 5432 | 改 5433 并同步 `.env` |
| Docker 拉取超时 | 网络/镜像 | 配 registry mirror/代理；不要死循环 |
| `uv sync` SSL/超时 | PyPI | `UV_DEFAULT_INDEX` 换清华镜像；重试 |
| `uv sync` 磁盘不足 | C: 空间 | 清理；至少 15GB |
| `.venv` Python 版本不对 | `python --version` | 问人类后删 `.venv`；`uv venv --python 3.12`；`uv sync` |
| `alembic` 连接失败 | DB/端口/密码 | 回 §6；确认 `.env` URL |
| `CREATE EXTENSION vector` 失败 | 镜像 | 必须是 `pgvector/pgvector:pg16`；停止 |
| 后端启动即退出 | 模型缓存 | 回 §9；看 `LocalEntryNotFoundError` |
| 后端 500 / `UpstreamError` | key/余额/模型名 | 查 401/402/400；看后端窗口 |
| `Model Not Exist` | `.env` | `LLM_MODEL=deepseek-flash` |
| 前端 Vite 报 Node engine | Node | 升到 22.12+/24；不要 `--force` |
| 前端 404/空白 | 后端/代理 | 后端 8000 是否健康；`vite.config.ts` 代理是否 8000 |
| 8000/5173 被占用 | 端口 | 停占用进程；改后端端口需同步改 Vite 代理 |
| `psycopg` 事件循环错误 | 启动方式 | 用 `scripts/run_server.py`，不要 `uvicorn` CLI |
| 首 token 很慢 | CPU 嵌入 | bge-m3 CPU 上数秒属正常；不要杀进程 |
| 重启后数据没了 | 卷 | `docker compose down -v` 会删数据；确认没执行；`pgdata` 卷是否还在 |
| M2 memory 403 | 白名单 | `.env` 的 `MEMORY_API_AGENTS=ext`，重启后端 |
| M2 memory 502 | 上游/摘要 | 看后端日志；M2 有已知稳定性问题，记录版本和复现步骤 |

**任何情况下都不要**：关闭 Defender、改 hosts、删 `data/`、`down -v`、`git reset --hard`、用管理员权限跑所有命令。

---

## 13. 停止、重启、清理

```powershell
# 停止后端/前端 + 数据库容器（数据保留）
.\scripts\stop.ps1

# 只停数据库
docker compose stop

# 停止并移除容器（数据卷仍保留）
docker compose down

# 危险：会删除数据库数据，必须人类明确同意
# docker compose down -v
```

- `data/` 存头像、trace、M2 buffer；不要删。
- 升级：`git fetch` → `git checkout <新 SHA>` → `uv sync` → `uv run alembic upgrade head` → 重启；`pgdata` 卷保留。
- 日志：后端/前端窗口；`data/traces/`。当前版本日志只进控制台，关闭窗口会丢；事故时保留窗口。

---

## 14. 安全与隐私

- `.env` 含 key，已在 `.gitignore`；不要提交、截图、发聊天。
- 不要让 DSH 把 key 写入日志/报告；需要检查时只输出 `KEY_PRESENT`。
- trace 会记录截断后的对话片段，按隐私数据处理。
- 服务只绑定 `127.0.0.1`，无鉴权；不要端口转发到公网。
- 如果 key 泄露：去平台删除/重建；不要只改 `.env` 留着旧 key。
- 不要为了“让它跑”关闭杀毒/防火墙；只允许人类手动加例外。

---

## 15. 安装报告模板（让朋友回传）

```text
Windows:
CPU / RAM / 磁盘:
git / docker / uv / node 版本:
目标 commit（git rev-parse HEAD）:
安装耗时:
/health:
真实聊天是否成功（模型名）:
重启后数据是否还在:
M2 memory turn/state 是否成功:
前端 5173 是否正常:
遇到的错误 + 处理:
未解决项:
```

---

## 附录 A：给 DSH 的启动提示（朋友复制给 DSH）

```text
请阅读并严格执行 SETUP-AGENT.md。

规则：
1. 先做只读检查，把结果和计划给我；不要直接开始装系统。
2. 遇到 UAC、重启、BIOS、Docker 登录、API Key、付款，停下来让我操作。
3. 不要打印/回显我的 API Key；只在 .env 里使用。
4. 同一条命令失败两次就停止，报告命令、错误关键行、已尝试方案。
5. 不要关闭 Defender/防火墙，不要改 hosts，不要删 data/，不要 docker compose down -v。
6. 目标 commit 必须是 SETUP-AGENT.md 里的 TARGET_COMMIT；不是就停止。
7. 每步给我“命令 + 结果 + 下一步”，最后按验收清单逐项确认。
```

---

## 附录 B：M2 发送前检查（项目所有者看，DSH 可跳过）

- [ ] `git fetch origin` 后按 H-01 rebase 到 `origin/main`，跑全量 pytest。
- [ ] push M2，打 tag（如 `v3-m2`），记录 `git rev-parse HEAD`。
- [ ] 修 `.env.example`、`src/app/config.py`、README 配置表里的 `LLM_MODEL`：`deepseek-chat` → `deepseek-flash`。
- [ ] 修 README：Node ≥22.12；首次 bge-m3 下载需 `HF_HUB_OFFLINE=0` + `HF_ENDPOINT`。
- [ ] 更新本文 `TARGET_COMMIT` 的 3 处：头部、§1 表格、§4 代码块；同时更新“公开 main 当前是 V2.6”的说明。
- [ ] 用干净 Windows VM + 全新 DSH，只给这份文档从零跑一遍。
- [ ] 确认文档/仓库里没有真实 API Key、没有 `data/`、没有未提交改动。

---

## 附录 C：版本事实（2026-10 核对）

- DeepSeek 当前模型：`deepseek-flash`、`deepseek-v4-pro`；旧名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停止使用（[更新日志](https://api-docs.deepseek.com/zh-cn/updates)）。
- 本项目默认配置（截至本稿）仍是 `deepseek-chat`，**新装必须改**。
- Vite 8.3.4 engines：`^20.19.0 || >=22.12.0`；README 的 Node ≥18 已过时。
- bge-m3 默认离线加载；首次必须显式允许下载，否则启动失败。
- 公开仓库 `main` 当前是 V2.6（`4417bf1`），M2 尚未 push；M2 的 `/memory/*` 只在目标 commit 中存在。
