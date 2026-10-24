import { useEffect, useMemo, useState } from 'react'

import {
  deleteLoreDoc,
  importLoreDoc,
  listAgents,
  listLoreDocs,
  reindexLore,
  type Agent,
  type LoreDoc,
} from '../api/client'
import { formatDateTime } from '../lib/format'

const SLUG_RE = /^[a-z0-9_]{1,40}$/

function isImageAvatar(avatar: string | null): avatar is string {
  return avatar !== null && (avatar.startsWith('/') || avatar.startsWith('http'))
}

/**
 * 世界书页：文档列表 / 导入（同 slug 版本递增）/ 删除 / 索引检查与重建。
 *
 * 只暴露面向使用者的两项操作：检查（是否导入成功）与重建索引（检查失败后的修复）。
 * 三路召回、RRF、重排、预算等调试细节属于开发者能力，不在前端展示。
 * 后端 `LoreDocRead` 不返回正文，因此「编辑」只能用同 slug 重新导入。
 */
export default function LorePage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [activeAgentId, setActiveAgentId] = useState<number | null>(null)
  const [docs, setDocs] = useState<LoreDoc[]>([])
  const [agentsLoading, setAgentsLoading] = useState(true)
  const [docsLoading, setDocsLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  // 检查与重建共用一个“忙”标记：两者都基于同一份索引，互斥更安全
  const [indexBusy, setIndexBusy] = useState<'check' | 'rebuild' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // 导入表单
  const [showImport, setShowImport] = useState(false)
  const [slug, setSlug] = useState('')
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [alwaysOn, setAlwaysOn] = useState(false)

  useEffect(() => {
    listAgents()
      .then((list) => {
        setAgents(list)
        setActiveAgentId(list.length > 0 ? list[0].id : null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setAgentsLoading(false))
  }, [])

  const activeAgent = useMemo(
    () => agents.find((a) => a.id === activeAgentId) ?? null,
    [agents, activeAgentId],
  )

  useEffect(() => {
    if (activeAgentId === null) {
      setDocs([])
      return
    }
    let cancelled = false
    setDocsLoading(true)
    setError(null)
    ;(async () => {
      try {
        const list = await listLoreDocs(activeAgentId)
        if (!cancelled) setDocs(list)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setDocsLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeAgentId])

  async function refreshDocs(agentId: number) {
    const list = await listLoreDocs(agentId)
    setDocs(list)
    return list
  }

  async function handleFile(file: File) {
    try {
      const text = await file.text()
      setContent(text)
      if (title.trim() === '') {
        const base = file.name.replace(/\.(md|txt|markdown)$/i, '')
        setTitle(base)
      }
      if (slug.trim() === '') {
        const guess = file.name
          .replace(/\.(md|txt|markdown)$/i, '')
          .toLowerCase()
          .replace(/[^a-z0-9_]+/g, '_')
          .replace(/^_+|_+$/g, '')
          .slice(0, 40)
        if (SLUG_RE.test(guess)) setSlug(guess)
      }
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function handleImport(e: React.FormEvent) {
    e.preventDefault()
    if (activeAgentId === null || busy) return
    const cleanSlug = slug.trim()
    const cleanTitle = title.trim()
    if (!SLUG_RE.test(cleanSlug)) {
      setError('slug 只能包含小写字母、数字、下划线，长度 1~40')
      return
    }
    if (cleanTitle === '') {
      setError('标题不能为空')
      return
    }
    if (content.trim() === '') {
      setError('内容不能为空：请粘贴文本或选择文件')
      return
    }
    setError(null)
    setNotice(null)
    setBusy(true)
    try {
      const result = await importLoreDoc({
        agent_id: activeAgentId,
        slug: cleanSlug,
        title: cleanTitle,
        content,
        always_on: alwaysOn,
      })
      await refreshDocs(activeAgentId)
      setNotice(
        result.changed
          ? `已导入 v${result.version}：${result.chunks} 块 · ${result.tokens} tokens`
          : `内容未变化（仍为 v${result.version}）：${result.chunks} 块 · ${result.tokens} tokens`,
      )
      setContent('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete(doc: LoreDoc) {
    if (busy) return
    if (!window.confirm(`删除设定「${doc.title}」（${doc.slug}）？\n该文档的索引会被清除，不可恢复。`)) return
    setError(null)
    setNotice(null)
    setBusy(true)
    try {
      await deleteLoreDoc(doc.id)
      if (activeAgentId !== null) await refreshDocs(activeAgentId)
      setNotice(`已删除「${doc.title}」`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  /** 检查：只看“有没有导入、索引是否完整”，结论只给成功/失败。 */
  async function handleCheck() {
    if (activeAgentId === null || indexBusy !== null) return
    setError(null)
    setNotice(null)
    if (docs.length === 0) {
      setError('检查失败：该角色还没有导入世界书设定')
      return
    }
    setIndexBusy('check')
    try {
      // dry_run 只统计缺索引的片段，不写库；为 0 即视为导入成功
      const result = await reindexLore(activeAgentId, true)
      if (result.chunks_rebuilt > 0) {
        setError('检查失败：索引不完整，请点击「重建索引」')
      } else {
        setNotice('检查通过：世界书已正确导入')
      }
    } catch (e) {
      setError(`检查失败：${(e as Error).message}`)
    } finally {
      setIndexBusy(null)
    }
  }

  /** 重建：补齐缺失的索引；结果同样只给成功/失败。 */
  async function handleReindex() {
    if (activeAgentId === null || indexBusy !== null) return
    setError(null)
    setNotice(null)
    setIndexBusy('rebuild')
    try {
      await reindexLore(activeAgentId, false)
      await refreshDocs(activeAgentId)
      setNotice('重建成功')
    } catch (e) {
      setError(`重建失败：${(e as Error).message}`)
    } finally {
      setIndexBusy(null)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="side-title">角色</div>
        {agentsLoading ? (
          <p className="hint">加载角色中…</p>
        ) : agents.length === 0 ? (
          <p className="hint">暂无角色，请到「角色管理」新建。</p>
        ) : (
          <ul className="agent-list">
            {agents.map((agent) => (
              <li key={agent.id}>
                <button
                  type="button"
                  className={agent.id === activeAgentId ? 'agent-item active' : 'agent-item'}
                  onClick={() => setActiveAgentId(agent.id)}
                >
                  {isImageAvatar(agent.avatar) ? (
                    <img className="agent-avatar" src={agent.avatar} alt="" />
                  ) : (
                    <span className="agent-avatar">{agent.name.slice(0, 1)}</span>
                  )}
                  <span className="agent-text">
                    <span className="agent-name">{agent.name}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="manage lore">
        <header className="chat-header">
          <span className="chat-title">世界书</span>
          <span className="chat-sub">
            {activeAgent === null
              ? '先在左侧选择角色'
              : `${activeAgent.name} · ${docs.length} 篇设定`}
          </span>
          {activeAgent !== null && (
            <div className="chat-actions">
              <button
                type="button"
                className="chat-idle-btn"
                disabled={indexBusy !== null}
                title="检查世界书是否已正确导入（只检查，不修改数据）"
                onClick={() => void handleCheck()}
              >
                {indexBusy === 'check' ? '检查中…' : '检查'}
              </button>
              <button
                type="button"
                className="chat-idle-btn"
                disabled={indexBusy !== null}
                title="检查失败时点击：补齐缺失的索引（可能较慢，请勿重复点击）"
                onClick={() => void handleReindex()}
              >
                {indexBusy === 'rebuild' ? '重建中…' : '重建索引'}
              </button>
              <button
                type="button"
                className="chat-clear-btn"
                disabled={docsLoading || busy}
                title="刷新文档列表"
                onClick={() => activeAgentId !== null && void refreshDocs(activeAgentId)}
              >
                刷新
              </button>
            </div>
          )}
        </header>

        <div className="lore-body">
          {error !== null && <div className="error-bar">{error}</div>}
          {notice !== null && <div className="notice-bar">{notice}</div>}

          <section className="panel">
            <div className="panel-head">
              <h2>设定文档</h2>
              <div className="panel-head-actions">
                <span className="count">{docs.length} 篇</span>
                <button
                  type="button"
                  className="mini-btn"
                  disabled={activeAgentId === null || busy}
                  onClick={() => {
                    setShowImport((v) => !v)
                    setError(null)
                    setNotice(null)
                  }}
                >
                  {showImport ? '收起导入' : '＋ 导入设定'}
                </button>
              </div>
            </div>

            {showImport && (
              <form className="lore-form" onSubmit={handleImport}>
                <div className="lore-form-row">
                  <div className="field">
                    <span className="field-label">slug（唯一标识）</span>
                    <input
                      className="text-input"
                      value={slug}
                      maxLength={40}
                      placeholder="例如：four_seasons_notes"
                      disabled={busy}
                      onChange={(e) => setSlug(e.target.value)}
                    />
                    <span className="field-hint">
                      小写字母 / 数字 / 下划线，1~40 位；同 slug 再次导入会生成新版本。
                    </span>
                  </div>
                  <div className="field">
                    <span className="field-label">标题</span>
                    <input
                      className="text-input"
                      value={title}
                      placeholder="例如：四季夏目的读书笔记"
                      disabled={busy}
                      onChange={(e) => setTitle(e.target.value)}
                    />
                  </div>
                </div>

                <div className="field">
                  <div className="tool-field-head">
                    <span className="field-label">正文</span>
                    <label className="file-btn small">
                      选择 .md / .txt
                      <input
                        type="file"
                        accept=".md,.markdown,.txt,text/markdown,text/plain"
                        disabled={busy}
                        onChange={(e) => {
                          const file = e.target.files?.[0]
                          if (file !== undefined) void handleFile(file)
                          e.target.value = ''
                        }}
                      />
                    </label>
                  </div>
                  <textarea
                    className="text-input area lore-content"
                    value={content}
                    placeholder="粘贴 Markdown / 纯文本，或用上面的按钮从文件载入。"
                    disabled={busy}
                    onChange={(e) => setContent(e.target.value)}
                  />
                </div>

                <div className="lore-form-actions">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={alwaysOn}
                      disabled={busy}
                      onChange={(e) => setAlwaysOn(e.target.checked)}
                    />
                    常驻（always_on，每轮都注入）
                  </label>
                  <button type="submit" className="btn-primary small" disabled={busy}>
                    {busy ? '导入中…' : '导入 / 更新'}
                  </button>
                  <span className="field-hint">
                    后端不返回正文，因此「编辑」只能用同 slug 重新导入。
                  </span>
                </div>
              </form>
            )}

            {docsLoading ? (
              <p className="hint">加载文档中…</p>
            ) : docs.length === 0 ? (
              <div className="empty-panel">
                该角色还没有世界书设定。导入一篇 Markdown，角色就能在对话中引用它。
              </div>
            ) : (
              <ul className="lore-docs">
                {docs.map((doc) => (
                  <li key={doc.id} className="lore-doc">
                    <div className="lore-doc-main">
                      <span className="lore-doc-title">{doc.title}</span>
                      <span className="lore-doc-meta">
                        <code>{doc.slug}</code>
                        <span>v{doc.version}</span>
                        <span>{doc.chunks} 段</span>
                        {doc.always_on && <span className="chip on">常驻</span>}
                        <span>更新 {formatDateTime(doc.updated_at)}</span>
                      </span>
                    </div>
                    <button
                      type="button"
                      className="mini-btn danger"
                      disabled={busy}
                      title="删除该设定（连同其索引）"
                      onClick={() => void handleDelete(doc)}
                    >
                      删除
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}
