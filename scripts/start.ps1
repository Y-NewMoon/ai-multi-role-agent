# 一键启动（Windows）：双击项目根目录的 start.bat 即可。
#
# 流程：确保 Docker 引擎 → docker compose up -d → 等 PostgreSQL 就绪
#       → alembic upgrade head → 起后端（独立窗口）→ 起前端（独立窗口）→ 打开浏览器。
#
# 约束提醒：
# - 后端必须走 scripts/run_server.py（Windows 需 SelectorEventLoop，见交接文档 5.3），不用 uvicorn CLI。
# - 本脚本不改动任何业务数据；不自动播种演示角色。

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot   # 项目根目录（scripts 的上一级）
Set-Location $root

$backendPort = 8000
$frontendPort = 5173
$healthUrl = "http://127.0.0.1:$backendPort/health"
$frontendDir = Join-Path $root 'frontend'

function Info($m) { Write-Host "[启动] $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [警告] $m" -ForegroundColor Yellow }
function Die($m) {
    Write-Host "  [失败] $m" -ForegroundColor Red
    exit 1
}

function Test-PortOpen([int]$port) {
    # IPv4 / IPv6 都算已监听：Vite dev server 默认只绑 localhost(::1)，后端绑 127.0.0.1
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return [bool]$conn
}

function Wait-PortOpen([int]$port, [int]$seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-PortOpen $port) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Test-BackendHealthy {
    try {
        $r = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return ($r.status -eq 'ok')
    } catch {
        return $false
    }
}

function Test-DockerEngine {
    docker info > $null 2>&1
    return ($LASTEXITCODE -eq 0)
}

Write-Host ''
Write-Host '===== AI 多角色扮演智能体 · 一键启动 =====' -ForegroundColor White

# ---------- 1. Docker 引擎 ----------
Info '检查 Docker 引擎 ...'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Die '未找到 docker 命令，请先安装 Docker Desktop。'
}
if (-not (Test-DockerEngine)) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'),
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Docker\Docker\Docker Desktop.exe')
    )
    $exe = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $exe) {
        Die 'Docker 引擎未运行，且未找到 Docker Desktop。请手动启动 Docker Desktop 后重试。'
    }
    Warn 'Docker 引擎未运行，正在启动 Docker Desktop（首次可能要 1-2 分钟）...'
    Start-Process -FilePath $exe | Out-Null
    for ($i = 0; $i -lt 90; $i++) {
        if (Test-DockerEngine) { break }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-DockerEngine)) {
        Die '等待 Docker 引擎超时（3 分钟），请确认 Docker Desktop 已正常启动后重试。'
    }
}
Ok 'Docker 引擎已就绪'

# ---------- 2. .env ----------
Info '检查 .env ...'
$envPath = Join-Path $root '.env'
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $root '.env.example') $envPath
    Die '.env 不存在，已从 .env.example 复制生成。请在 .env 填入 LLM_API_KEY 后重新双击 start.bat。'
}
$keyLine = Select-String -Path $envPath -Pattern '^\s*LLM_API_KEY\s*=\s*(.*)$' | Select-Object -First 1
$keyVal = if ($keyLine) { $keyLine.Matches[0].Groups[1].Value.Trim() } else { '' }
if ([string]::IsNullOrWhiteSpace($keyVal)) {
    Die '.env 中 LLM_API_KEY 为空，无法调用对话模型。请填写后重试。'
}
Ok '.env 就绪（LLM_API_KEY 已配置）'

# ---------- 3. 数据库容器 ----------
Info '启动数据库容器（docker compose up -d）...'
docker compose up -d
if ($LASTEXITCODE -ne 0) { Die 'docker compose up -d 执行失败，请查看上方输出。' }

Info '等待 PostgreSQL 就绪 ...'
$dbReady = $false
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T db pg_isready -U roleplay -d roleplay > $null 2>&1
    if ($LASTEXITCODE -eq 0) { $dbReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $dbReady) { Die 'PostgreSQL 60 秒内未就绪，请查看容器状态：docker compose ps' }
Ok 'PostgreSQL 已就绪'

# ---------- 4. 迁移 ----------
Info '执行数据库迁移（alembic upgrade head）...'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$py = Join-Path $root '.venv\Scripts\python.exe'
$alembic = Join-Path $root '.venv\Scripts\alembic.exe'
if (-not (Test-Path $py)) {
    Die '未找到 .venv，请先在项目根目录执行 uv sync。'
}
if (Test-Path $alembic) { & $alembic upgrade head } else { & $py -m alembic upgrade head }
if ($LASTEXITCODE -ne 0) { Die 'alembic upgrade head 失败，请查看上方输出。' }
Ok '迁移完成'

# ---------- 5. 后端 ----------
Info '启动后端 ...'
if (Test-BackendHealthy) {
    Ok "后端已在运行（127.0.0.1:$backendPort），跳过启动"
} elseif (Test-PortOpen $backendPort) {
    Die "端口 $backendPort 已被其它进程占用，请先释放后重试。"
} else {
    # 相对路径 + 工作目录 = 命令行无空格，避免 Start-Process 引号转义问题
    $backendCmd = 'chcp 65001>nul & title ai-agent-backend & .venv\Scripts\python.exe scripts\run_server.py'
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', $backendCmd -WorkingDirectory $root | Out-Null
    Info "等待后端就绪（首次需加载本地嵌入模型 bge-m3，约 15-30 秒）..."
    if (-not (Wait-PortOpen $backendPort 120)) {
        Die "后端 120 秒内未监听 $backendPort，请查看 'ai-agent-backend' 窗口的报错。"
    }
    Ok "后端已启动（http://127.0.0.1:$backendPort）"
}

# ---------- 6. 前端 ----------
Info '启动前端 ...'
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Die '未找到 npm，请先安装 Node.js。'
}
if (Test-PortOpen $frontendPort) {
    Ok "端口 $frontendPort 已被占用，跳过启动（若前端已在运行则直接使用）"
} else {
    $nodeModules = Join-Path $frontendDir 'node_modules'
    if (-not (Test-Path $nodeModules)) {
        Info '首次运行：安装前端依赖（npm install）...'
        Push-Location $frontendDir
        npm install
        $installCode = $LASTEXITCODE
        Pop-Location
        if ($installCode -ne 0) { Die 'npm install 失败，请查看上方输出。' }
    }
    $frontendCmd = 'chcp 65001>nul & title ai-agent-frontend & npm run dev'
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', $frontendCmd -WorkingDirectory $frontendDir | Out-Null
    Info '等待前端 dev server 就绪 ...'
    if (-not (Wait-PortOpen $frontendPort 120)) {
        Die "前端 120 秒内未监听 $frontendPort，请查看 'ai-agent-frontend' 窗口的报错。"
    }
    Ok "前端已启动（http://localhost:$frontendPort）"
}

# ---------- 7. 打开浏览器 ----------
$url = "http://localhost:$frontendPort"
Start-Process $url | Out-Null
Ok "已在浏览器打开 $url"

Write-Host ''
Write-Host '----------------- 启动完成 -----------------' -ForegroundColor Green
Write-Host "  前端（在浏览器使用）: $url"
Write-Host "  后端接口文档        : http://127.0.0.1:$backendPort/docs"
Write-Host "  健康检查            : $healthUrl"
Write-Host ''
Write-Host '  停止：关闭 "ai-agent-backend" 与 "ai-agent-frontend" 两个窗口即可。'
Write-Host '       数据库容器如需停止：docker compose stop'
Write-Host '-------------------------------------------' -ForegroundColor Green
Write-Host ''
