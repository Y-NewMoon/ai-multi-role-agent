import { useEffect, useState } from 'react'

import {
  clearAgentMemory,
  createAgent,
  deleteAgent,
  listAgents,
  updateAgent,
  uploadAvatar,
  type Agent,
} from '../api/client'
import SkillBadges from '../components/SkillBadges'
import { SKILLS } from '../constants/skills'

type Mode = 'edit' | 'create'

interface FormState {
  name: string
  persona: string
  background: string
  avatar: string | null
  allowed_skills: string[]
}

const EMPTY_FORM: FormState = {
  name: '',
  persona: '',
  background: '',
  avatar: null,
  allowed_skills: [],
}

function isImageAvatar(avatar: string | null): avatar is string {
  return avatar !== null && (avatar.startsWith('/') || avatar.startsWith('http'))
}

/** 角色管理页：新建 / 编辑（姓名·人设·背景·头像）/ 删除 / 清除记忆。 */
export default function AgentManagePage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [mode, setMode] = useState<Mode>('create')
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  useEffect(() => {
    listAgents()
      .then((list) => setAgents(list))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  function beginCreate() {
    setMode('create')
    setSelectedId(null)
    setForm(EMPTY_FORM)
    setError(null)
    setNotice(null)
  }

  function selectAgent(agent: Agent) {
    setMode('edit')
    setSelectedId(agent.id)
    setForm({
      name: agent.name,
      persona: agent.persona_prompt,
      background: agent.background ?? '',
      avatar: agent.avatar,
      allowed_skills: [...agent.allowed_skills],
    })
    setError(null)
    setNotice(null)
  }

  function toggleTool(name: string) {
    setForm((f) => ({
      ...f,
      allowed_skills: f.allowed_skills.includes(name)
        ? f.allowed_skills.filter((x) => x !== name)
        : [...f.allowed_skills, name],
    }))
  }

  async function refresh() {
    const list = await listAgents()
    setAgents(list)
    return list
  }

  async function handleUpload(file: File) {
    setError(null)
    setBusy(true)
    try {
      const url = await uploadAvatar(file)
      setForm((f) => ({ ...f, avatar: url }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault()
    if (form.name.trim() === '' || form.persona.trim() === '') {
      setError('姓名与人设为必填')
      return
    }
    setError(null)
    setBusy(true)
    const payload = {
      name: form.name.trim(),
      persona_prompt: form.persona,
      background: form.background.trim() === '' ? null : form.background,
      avatar: form.avatar,
      allowed_skills: form.allowed_skills,
    }
    try {
      if (mode === 'create') {
        const created = await createAgent(payload)
        await refresh()
        selectAgent(created)
        setNotice('已创建角色')
      } else if (selectedId !== null) {
        const updated = await updateAgent(selectedId, payload)
        await refresh()
        selectAgent(updated)
        setNotice('已保存修改')
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete() {
    if (selectedId === null) return
    const target = agents.find((a) => a.id === selectedId)
    if (!window.confirm(`删除角色「${target?.name ?? selectedId}」？\n其会话、消息与全部记忆会一并清除，且不可恢复。`))
      return
    setError(null)
    setBusy(true)
    try {
      await deleteAgent(selectedId)
      const list = await refresh()
      if (list.length > 0) selectAgent(list[0])
      else beginCreate()
      setNotice('已删除角色')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function handleClearMemory() {
    if (selectedId === null) return
    const target = agents.find((a) => a.id === selectedId)
    if (
      !window.confirm(
        `清除「${target?.name ?? selectedId}」的记忆？\n将清空长期记忆与全部对话历史（角色档案保留），不可恢复。`,
      )
    )
      return
    setError(null)
    setBusy(true)
    try {
      await clearAgentMemory(selectedId)
      setNotice('已清除该角色的记忆与对话历史')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const avatarPreview = isImageAvatar(form.avatar) ? form.avatar : null
  const avatarFallback = form.name.trim().slice(0, 1) || '新'

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="side-title">角色</div>
        <button type="button" className="new-group-btn" onClick={beginCreate} disabled={busy}>
          ＋ 新建角色
        </button>
        {loading ? (
          <p className="hint">加载角色中…</p>
        ) : agents.length === 0 ? (
          <p className="hint">暂无角色，点击上方新建。</p>
        ) : (
          <ul className="agent-list">
            {agents.map((agent) => (
              <li key={agent.id}>
                <button
                  type="button"
                  className={agent.id === selectedId ? 'agent-item active' : 'agent-item'}
                  disabled={busy}
                  onClick={() => selectAgent(agent)}
                >
                  {isImageAvatar(agent.avatar) ? (
                    <img className="agent-avatar" src={agent.avatar} alt="" />
                  ) : (
                    <span className="agent-avatar">{agent.name.slice(0, 1)}</span>
                  )}
                  <span className="agent-text">
                    <span className="agent-name">{agent.name}</span>
                    <SkillBadges skills={agent.allowed_skills} />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="manage">
        <header className="chat-header">
          <span className="chat-title">{mode === 'create' ? '新建角色' : '编辑角色'}</span>
          {mode === 'edit' && selectedId !== null && (
            <span className="chat-sub">ID #{selectedId}</span>
          )}
        </header>

        <form className="manage-body" onSubmit={handleSave}>
          <div className="field">
            <span className="field-label">姓名</span>
            <input
              className="text-input"
              value={form.name}
              maxLength={100}
              placeholder="例如：四季夏目"
              disabled={busy}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
            />
          </div>

          <div className="field">
            <span className="field-label">头像</span>
            <div className="avatar-row">
              {avatarPreview ? (
                <img className="avatar-preview" src={avatarPreview} alt="头像预览" />
              ) : (
                <span className="avatar-preview placeholder">{avatarFallback}</span>
              )}
              <div className="avatar-actions">
                <label className="file-btn">
                  选择图片
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp,image/gif"
                    disabled={busy}
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) void handleUpload(file)
                      e.target.value = ''
                    }}
                  />
                </label>
                {form.avatar !== null && (
                  <button
                    type="button"
                    className="mini-btn"
                    disabled={busy}
                    onClick={() => setForm((f) => ({ ...f, avatar: null }))}
                  >
                    移除
                  </button>
                )}
                <span className="field-hint">PNG/JPG/WebP/GIF，≤2MB</span>
              </div>
            </div>
          </div>

          <div className="field">
            <span className="field-label">人设（system prompt）</span>
            <textarea
              className="text-input area"
              value={form.persona}
              placeholder="描述该角色的性格、说话方式、行为准则…"
              disabled={busy}
              onChange={(e) => setForm((f) => ({ ...f, persona: e.target.value }))}
            />
          </div>

          <div className="field">
            <span className="field-label">背景</span>
            <textarea
              className="text-input area"
              value={form.background}
              placeholder="世界观 / 身世 / 与用户的关系等（可留空）"
              disabled={busy}
              onChange={(e) => setForm((f) => ({ ...f, background: e.target.value }))}
            />
          </div>

          <div className="field">
            <div className="tool-field-head">
              <span className="field-label">可用工具</span>
              <span className="tool-count">已选 {form.allowed_skills.length} / {SKILLS.length}</span>
            </div>
            <p className="field-hint">
              不勾选任何工具时，角色只会纯文本回复；工具名由后端 registry 校验，未注册的名字会返回 422。
            </p>
            <div className="tool-grid">
              {SKILLS.map((skill) => {
                const checked = form.allowed_skills.includes(skill.name)
                return (
                  <label
                    key={skill.name}
                    className={checked ? 'tool-option checked' : 'tool-option'}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={busy}
                      onChange={() => toggleTool(skill.name)}
                    />
                    <span className="tool-text">
                      <span className="tool-name">
                        {skill.label}
                        <code className="tool-code">{skill.name}</code>
                      </span>
                      <span className="tool-desc">{skill.description}</span>
                      {skill.note !== undefined && (
                        <span className="tool-note">{skill.note}</span>
                      )}
                    </span>
                  </label>
                )
              })}
            </div>
          </div>

          {notice !== null && <div className="notice-bar">{notice}</div>}
          {error !== null && <div className="error-bar">{error}</div>}

          <div className="manage-actions">
            <button type="submit" className="btn-primary" disabled={busy}>
              {mode === 'create' ? '创建角色' : '保存修改'}
            </button>
            {mode === 'edit' && (
              <button type="button" className="btn-ghost" disabled={busy} onClick={handleClearMemory}>
                清除记忆
              </button>
            )}
            {mode === 'edit' && (
              <button type="button" className="btn-danger" disabled={busy} onClick={handleDelete}>
                删除角色
              </button>
            )}
          </div>
        </form>
      </main>
    </div>
  )
}
