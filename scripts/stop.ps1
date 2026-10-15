# 一键停止（Windows）：双击项目根目录的 stop.bat 即可。
#
# 顺序：后端窗口 → 前端窗口 → 数据库容器（docker compose stop，数据保留）。
# 数据安全：本脚本只 stop 容器，不使用 down -v，数据卷 pgdata 不会被删除。

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot   # 项目根目录（scripts 的上一级）
Set-Location $root

function Info($m) { Write-Host "[停止] $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [警告] $m" -ForegroundColor Yellow }

function Get-ChildProcId([int]$procId) {
    return @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$procId" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty ProcessId
    )
}

function Stop-ProcessTree([int]$procId) {
    # 先杀子再杀父，避免留下孤儿进程
    foreach ($child in Get-ChildProcId $procId) { Stop-ProcessTree $child }
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '===== AI 多角色扮演智能体 · 一键停止 =====' -ForegroundColor White

# ---------- 1. 后端 / 前端 ----------
Info '停止后端 / 前端 ...'
# 按命令行匹配本项目启动的进程：两个带标题的窗口 + 后端脚本 + npm/vite
$patterns = @(
    'ai-agent-backend',
    'ai-agent-frontend',
    'run_server\.py',
    'npm-cli\.js\s+run\s+dev',
    'vite\\bin\\vite\.js'
)
$targets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $line = $_.CommandLine
    $line -and ($patterns | Where-Object { $line -match $_ })
}

if ($targets) {
    foreach ($t in $targets) { Stop-ProcessTree ([int]$t.ProcessId) }
    Ok "已停止 $($targets.Count) 个进程（含其子进程）"
} else {
    Ok '未发现运行中的后端 / 前端'
}

Start-Sleep -Seconds 2
foreach ($port in 8000, 5173) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Warn "端口 $port 仍被占用（PID $($conn[0].OwningProcess)），请手动处理"
    } else {
        Ok "端口 $port 已释放"
    }
}

# ---------- 2. 数据库容器 ----------
Info '停止数据库容器（docker compose stop）...'
$dockerReady = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker info > $null 2>&1
    $dockerReady = ($LASTEXITCODE -eq 0)
}
if (-not $dockerReady) {
    Ok 'Docker 引擎未运行，数据库容器本来就没启动'
} else {
    docker compose stop
    if ($LASTEXITCODE -ne 0) {
        Warn 'docker compose stop 失败，请查看上方输出'
    } else {
        Ok '数据库容器已停止（数据保留）'
    }
}

Write-Host ''
Write-Host '----------------- 停止完成 -----------------' -ForegroundColor Green
Write-Host '  后端 / 前端窗口已关闭，数据库容器已停止。'
Write-Host '  数据卷 pgdata 保留，下次双击 start.bat 可直接复用。'
Write-Host '  若启动器窗口（结尾停在 Press any key）还开着，直接关掉即可。'
Write-Host '-------------------------------------------' -ForegroundColor Green
Write-Host ''
