import { useEffect, useState } from 'react'

import { getHealth, type HealthInfo } from '../api/client'

function Flag({ on, label }: { on: boolean; label: string }) {
  return <span className={on ? 'chip on' : 'chip off'}>{label} {on ? '开' : '关'}</span>
}

/**
 * 顶栏运行状态：读取后端 `/health`，展示 llm_provider / 数据库 / 嵌入 / 世界书 / trace。
 * 用于 demo 与排障，避免“为什么是假回复 / 为什么没有工具”的误解。
 */
export default function HealthBadge() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)

  async function load() {
    setBusy(true)
    setError(null)
    try {
      const data = await getHealth()
      setHealth(data)
      setUpdatedAt(new Date().toLocaleTimeString('zh-CN', { hour12: false }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  useEffect(() => {
    if (!open) return
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open])

  const healthy = health !== null && health.status === 'ok'

  return (
    <div className="health">
      <button
        type="button"
        className="health-pill"
        aria-expanded={open}
        title="查看后端运行状态"
        onClick={() => {
          setOpen((v) => !v)
          if (!open) void load()
        }}
      >
        <span className={healthy ? 'health-dot ok' : error !== null ? 'health-dot bad' : 'health-dot'} />
        <span className="health-label">{health?.llm_provider ?? '状态'}</span>
      </button>

      {open && (
        <div className="health-popover" role="dialog" aria-label="运行状态">
          <div className="health-head">
            <h2>运行状态</h2>
            <div className="drawer-actions">
              <button
                type="button"
                className="icon-btn"
                disabled={busy}
                title="刷新"
                onClick={() => void load()}
              >
                {busy ? '…' : '刷新'}
              </button>
              <button type="button" className="icon-btn" title="关闭" onClick={() => setOpen(false)}>
                ✕
              </button>
            </div>
          </div>

          {error !== null && <div className="error-bar">{error}</div>}

          {health !== null && (
            <>
              <dl className="health-grid">
                <dt>状态</dt>
                <dd>{health.status}</dd>
                <dt>LLM</dt>
                <dd>
                  {health.llm_provider}
                  {health.fake_llm_calls !== null ? ` · fake 调用 ${health.fake_llm_calls}` : ''}
                </dd>
                <dt>数据库</dt>
                <dd>{health.database ?? '—'}</dd>
                <dt>嵌入</dt>
                <dd>
                  {health.embedding_provider}
                  {health.embedding_model !== null ? ` · ${health.embedding_model}` : ''}
                </dd>
                <dt>trace</dt>
                <dd>
                  {health.trace_enabled ? `开启 · 采样 ${health.trace_sample_rate}` : '关闭'}
                </dd>
                <dt>目录</dt>
                <dd>{health.trace_dir}</dd>
              </dl>

              <div className="health-chips">
                <Flag on={health.lore_enabled} label="世界书" />
                <Flag on={health.lore_static_mode} label="静态注入" />
                <Flag on={health.memory_extract_enabled} label="记忆抽取" />
                <Flag on={health.memory_profile_enabled} label="记忆画像" />
                <Flag on={health.memory_l2_enabled} label="L2 摘要" />
                <Flag on={health.memory_weighted_recall} label="加权召回" />
                <Flag on={health.trace_enabled} label="trace" />
              </div>
            </>
          )}

          <span className="health-time">
            {updatedAt !== null ? `更新于 ${updatedAt}` : '尚未获取到状态'}
          </span>
        </div>
      )}
    </div>
  )
}
