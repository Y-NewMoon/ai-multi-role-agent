import { useEffect, useMemo, useRef, useState } from 'react'

import {
  cancelReminder,
  clearIdle,
  confirmReminder,
  createSession,
  deleteSession,
  listAgents,
  listMessages,
  listReminders,
  listSessions,
  recallMessage,
  streamChat,
  streamChatIdle,
  updateSession,
  type Agent,
  type Reminder,
  type Session,
} from '../api/client'
import AgentList from '../components/AgentList'
import InputBox from '../components/InputBox'
import MessageList from '../components/MessageList'
import SkillBadges from '../components/SkillBadges'
import { formatDate, formatDateTime } from '../lib/format'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /** 库中消息 id；0 表示尚未落库（发送中的占位消息），此时不提供撤回。 */
  id: number
  /** 是否为角色自主闲聊（按钮触发的主动发言）。 */
  isIdle?: boolean
  /** 角色该条发言的「内心独白」（思考链）；无则不显示「思考」按钮。 */
  thinking?: string | null
}

const REMINDER_STATUS_LABEL: Record<string, string> = {
  pending: '待确认',
  active: '已确认',
  cancelled: '已取消',
  fired: '已触发',
}

/**
 * 单人对话页（V2.6 前端补齐）：
 * - 角色 → 会话两级导航，会话可新建/切换/重命名/删除（不再合并多会话历史）；
 * - 工具白名单徽标；
 * - 私聊提醒 HITL 抽屉（pending → 用户确认 → active）。
 */
export default function ChatPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [activeAgentId, setActiveAgentId] = useState<number | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [loading, setLoading] = useState(true)
  const [threadLoading, setThreadLoading] = useState(false)
  const [sessionBusy, setSessionBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 会话重命名（行内编辑）
  const [renamingId, setRenamingId] = useState<number | null>(null)
  const [renameText, setRenameText] = useState('')
  const cancelRenameRef = useRef(false)
  /** 防止「Enter 提交」与「失焦提交」重复发请求。 */
  const commitGuardRef = useRef(false)

  // 提醒面板
  const [reminderOpen, setReminderOpen] = useState(false)
  const [reminders, setReminders] = useState<Reminder[]>([])
  const [reminderLoading, setReminderLoading] = useState(false)
  const [reminderBusy, setReminderBusy] = useState(false)

  /**
   * 当前 `messages` 对应的会话 id；用于区分「刚在本地创建、无需回读」与「切换后需从后端拉取」。
   * 不能用 Set：StrictMode 下 effect 会执行两次，消费型标记会被第一次调用吃掉。
   */
  const loadedSessionRef = useRef<number | null>(null)

  useEffect(() => {
    listAgents()
      .then((list) => {
        setAgents(list)
        setActiveAgentId(list.length > 0 ? list[0].id : null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const activeAgent = useMemo(
    () => agents.find((a) => a.id === activeAgentId) ?? null,
    [agents, activeAgentId],
  )
  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? null,
    [sessions, activeSessionId],
  )
  const pendingCount = useMemo(
    () => reminders.filter((r) => r.status === 'pending').length,
    [reminders],
  )

  /** 切换角色：清空当前会话上下文，拉取该角色的会话列表并默认选中最近一个。 */
  useEffect(() => {
    if (activeAgentId === null) {
      setSessions([])
      setActiveSessionId(null)
      setMessages([])
      loadedSessionRef.current = null
      return
    }
    let cancelled = false
    setThreadLoading(true)
    setError(null)
    ;(async () => {
      try {
        const list = await listSessions(activeAgentId)
        if (cancelled) return
        setSessions(list)
        const target = list.length > 0 ? list[list.length - 1].id : null
        loadedSessionRef.current = null
        setActiveSessionId(target)
        if (target === null) setMessages([])
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setThreadLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeAgentId])

  /** 切换会话：从后端只加载该会话的历史（不再合并同一角色的所有会话）。 */
  useEffect(() => {
    if (activeSessionId === null) {
      setMessages([])
      loadedSessionRef.current = null
      return
    }
    if (loadedSessionRef.current === activeSessionId) return

    let cancelled = false
    setThreadLoading(true)
    ;(async () => {
      try {
        const records = await listMessages(activeSessionId)
        if (cancelled) return
        setMessages(
          records
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map((m) => ({
              role: m.role as 'user' | 'assistant',
              content: m.content,
              id: m.id,
              isIdle: m.is_idle,
              thinking: m.thinking,
            })),
        )
        loadedSessionRef.current = activeSessionId
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setThreadLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeSessionId])

  /** 打开提醒面板时加载当前会话的提醒（cancelled 不返回时刷新）。 */
  useEffect(() => {
    if (!reminderOpen || activeAgentId === null || activeSessionId === null) return
    let cancelled = false
    setReminderLoading(true)
    ;(async () => {
      try {
        const list = await listReminders(activeAgentId, activeSessionId)
        if (!cancelled) setReminders(list)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setReminderLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [reminderOpen, activeAgentId, activeSessionId])

  function selectAgent(id: number) {
    if (streaming || id === activeAgentId) return
    setActiveAgentId(id)
    setActiveSessionId(null)
    setSessions([])
    setMessages([])
    setReminders([])
    setRenamingId(null)
    setError(null)
    loadedSessionRef.current = null
  }

  function selectSession(id: number) {
    if (streaming || id === activeSessionId) return
    loadedSessionRef.current = null
    setMessages([])
    setReminders([])
    setActiveSessionId(id)
    setError(null)
  }

  async function handleNewSession() {
    if (activeAgentId === null || streaming || sessionBusy) return
    setError(null)
    setSessionBusy(true)
    try {
      const created = await createSession(activeAgentId)
      loadedSessionRef.current = created.id
      setSessions((prev) => [...prev, created])
      setActiveSessionId(created.id)
      setMessages([])
      setReminders([])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSessionBusy(false)
    }
  }

  function startRename(session: Session) {
    if (streaming) return
    cancelRenameRef.current = false
    setRenamingId(session.id)
    setRenameText(session.title ?? '')
  }

  function cancelRename() {
    cancelRenameRef.current = false
    commitGuardRef.current = false
    setRenamingId(null)
  }

  async function commitRename() {
    if (renamingId === null || commitGuardRef.current) return
    const title = renameText.trim()
    if (title.length > 255) {
      setError('会话标题不能超过 255 个字符')
      return
    }
    commitGuardRef.current = true
    setError(null)
    setSessionBusy(true)
    try {
      const updated = await updateSession(renamingId, title === '' ? null : title)
      setSessions((prev) => prev.map((s) => (s.id === updated.id ? updated : s)))
      setRenamingId(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      commitGuardRef.current = false
      setSessionBusy(false)
    }
  }

  async function handleDeleteSession(session: Session) {
    if (streaming || sessionBusy) return
    const name = session.title?.trim() ? session.title : '未命名会话'
    if (
      !window.confirm(
        `删除会话「${name}」？\n该会话的消息、摘要与上下文会被清除；角色与其他会话保留。`,
      )
    )
      return
    setError(null)
    setSessionBusy(true)
    try {
      await deleteSession(session.id)
      const remaining = sessions.filter((s) => s.id !== session.id)
      setSessions(remaining)
      if (activeSessionId === session.id) {
        loadedSessionRef.current = null
        const next = remaining.length > 0 ? remaining[remaining.length - 1] : null
        setActiveSessionId(next?.id ?? null)
        setMessages([])
        setReminders([])
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSessionBusy(false)
    }
  }

  /** 追加一段文本到最后一条 assistant 气泡。 */
  function appendAssistant(text: string) {
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (last === undefined || last.role !== 'assistant') return prev
      const next = prev.slice()
      next[next.length - 1] = { ...last, content: last.content + text }
      return next
    })
  }

  /** 在末尾追加一个空的 assistant 占位气泡（角色连发多条时用）。 */
  function pushAssistant(isIdle = false) {
    setMessages((prev) => [...prev, { role: 'assistant', content: '', id: 0, isIdle }])
  }

  /** 确保有会话可用：本地已有则复用，否则新建并写入会话列表。 */
  async function ensureSession(agentId: number): Promise<number> {
    if (activeSessionId !== null) return activeSessionId
    const created = await createSession(agentId)
    loadedSessionRef.current = created.id
    setSessions((prev) => [...prev, created])
    setActiveSessionId(created.id)
    setReminders([])
    return created.id
  }

  async function loadReminders(agentId: number, sessionId: number, silent = false) {
    if (!silent) setReminderLoading(true)
    try {
      const list = await listReminders(agentId, sessionId)
      setReminders(list)
    } catch (e) {
      if (!silent) setError((e as Error).message)
    } finally {
      if (!silent) setReminderLoading(false)
    }
  }

  async function handleSend(text: string) {
    if (activeAgentId === null || streaming) return
    const agentId = activeAgentId
    setError(null)
    setStreaming(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text, id: 0 },
      { role: 'assistant', content: '', id: 0 },
    ])

    let sid: number | null = null
    try {
      sid = await ensureSession(agentId)
      await streamChat(sid, text, {
        onToken: (token) => appendAssistant(token),
        onBreak: () => pushAssistant(),
        onDone: (payload) => {
          setMessages((prev) => {
            // 回填本轮各条消息的真实 id：先定位刚发出的用户占位，再顺序回填其后各 assistant 气泡
            const next = prev.map((m) => ({ ...m }))
            let uid = -1
            for (let i = next.length - 1; i >= 0; i--) {
              if (next[i].role === 'user' && next[i].id === 0) {
                uid = i
                break
              }
            }
            if (uid < 0) return prev
            next[uid] = { ...next[uid], id: payload.user_message_id }
            let k = 0
            for (let i = uid + 1; i < next.length && k < payload.replies.length; i++) {
              if (next[i].role === 'assistant') {
                next[i] = {
                  ...next[i],
                  content: payload.replies[k].content,
                  id: payload.replies[k].message_id,
                  thinking: payload.replies[k].thinking,
                }
                k++
              }
            }
            // 清理未被回填的空白助手占位（模型多输出分隔符导致的幽灵气泡）
            return next.filter((m) => !(m.role === 'assistant' && m.id === 0 && m.content === ''))
          })
          if (sid !== null) void loadReminders(agentId, sid, true)
        },
      })
    } catch (e) {
      setError((e as Error).message)
      // 失败时移除本轮遗留的空助手气泡，避免留下空白回复
      setMessages((prev) =>
        prev.filter((m) => !(m.role === 'assistant' && m.id === 0 && m.content === '')),
      )
    } finally {
      setStreaming(false)
    }
  }

  /** 触发一次自主闲聊：角色主动发消息（无用户消息）。可反复点击。 */
  async function handleIdle() {
    if (activeAgentId === null || streaming) return
    const agentId = activeAgentId
    setError(null)
    setStreaming(true)

    try {
      const sid = await ensureSession(agentId)
      pushAssistant(true) // 起始占位气泡（标记为闲聊）
      await streamChatIdle(sid, {
        onToken: (token) => appendAssistant(token),
        onBreak: () => pushAssistant(true),
        onDone: (payload) => {
          setMessages((prev) => {
            // 闲聊无用户消息：把末尾连续的「未落库 assistant 占位」直接替换为落库后的真实回复
            let start = prev.length
            while (start > 0 && prev[start - 1].role === 'assistant' && prev[start - 1].id === 0) {
              start--
            }
            const head = prev.slice(0, start)
            const tail: ChatMessage[] = payload.replies.map((r) => ({
              role: 'assistant',
              content: r.content,
              id: r.message_id,
              isIdle: true,
              thinking: r.thinking,
            }))
            return [...head, ...tail]
          })
          void loadReminders(agentId, sid, true)
        },
      })
    } catch (e) {
      setError((e as Error).message)
      setMessages((prev) =>
        prev.filter((m) => !(m.role === 'assistant' && m.id === 0 && m.content === '')),
      )
    } finally {
      setStreaming(false)
    }
  }

  /** 清空当前会话的全部自主闲聊消息（含其写入的角色记忆）。 */
  async function handleClearIdle() {
    if (activeSessionId === null || streaming) return
    const sid = activeSessionId
    setError(null)
    try {
      const removed = await clearIdle(sid)
      const removedSet = new Set(removed)
      setMessages((prev) =>
        prev.filter(
          (m) => !removedSet.has(m.id) && !(m.role === 'assistant' && m.id === 0 && m.isIdle === true),
        ),
      )
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 撤回一条用户消息：后端连带删除角色回复与该轮产生的记忆，前端同步移除气泡。 */
  async function handleRecall(message: ChatMessage) {
    if (activeSessionId === null || streaming) return
    if (message.role !== 'user' || message.id <= 0) return
    setError(null)
    try {
      const removed = await recallMessage(activeSessionId, message.id)
      const removedSet = new Set(removed)
      setMessages((prev) => prev.filter((m) => !removedSet.has(m.id)))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function toggleReminder() {
    if (reminderOpen) {
      setReminderOpen(false)
      return
    }
    setReminderOpen(true)
  }

  async function handleConfirmReminder(reminder: Reminder) {
    if (reminderBusy) return
    setReminderBusy(true)
    setError(null)
    try {
      await confirmReminder(reminder.id)
      if (activeAgentId !== null && activeSessionId !== null) {
        await loadReminders(activeAgentId, activeSessionId)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setReminderBusy(false)
    }
  }

  async function handleCancelReminder(reminder: Reminder) {
    if (reminderBusy) return
    setReminderBusy(true)
    setError(null)
    try {
      await cancelReminder(reminder.id)
      if (activeAgentId !== null && activeSessionId !== null) {
        await loadReminders(activeAgentId, activeSessionId)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setReminderBusy(false)
    }
  }

  function handleReminderRefresh() {
    if (activeAgentId !== null && activeSessionId !== null) {
      void loadReminders(activeAgentId, activeSessionId)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="side-title">角色</div>
        {loading ? (
          <p className="hint">加载角色中…</p>
        ) : (
          <AgentList
            agents={agents}
            activeId={activeAgentId}
            disabled={streaming}
            onSelect={selectAgent}
          />
        )}

        {activeAgent !== null && (
          <>
            <div className="side-divider" />
            <div className="side-title-row">
              <span className="side-title">会话</span>
              <button
                type="button"
                className="side-add"
                title="新建会话"
                disabled={streaming || sessionBusy}
                onClick={() => void handleNewSession()}
              >
                ＋
              </button>
            </div>

            {sessions.length === 0 ? (
              <div className="session-empty">
                <p className="hint">该角色还没有会话；直接发消息会自动创建。</p>
                <button
                  type="button"
                  className="mini-btn"
                  disabled={streaming || sessionBusy}
                  onClick={() => void handleNewSession()}
                >
                  新建会话
                </button>
              </div>
            ) : (
              <ul className="session-list">
                {sessions.map((session) => (
                  <li
                    key={session.id}
                    className={session.id === activeSessionId ? 'session-row active' : 'session-row'}
                  >
                    {renamingId === session.id ? (
                      <div className="session-rename-row">
                        <input
                          className="session-rename"
                          autoFocus
                          maxLength={255}
                          aria-label="会话名称"
                          value={renameText}
                          disabled={sessionBusy}
                          onChange={(e) => setRenameText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault()
                              void commitRename()
                            } else if (e.key === 'Escape') {
                              e.preventDefault()
                              cancelRenameRef.current = true
                              e.currentTarget.blur()
                            }
                          }}
                          onBlur={() => {
                            if (cancelRenameRef.current) {
                              cancelRename()
                              return
                            }
                            void commitRename()
                          }}
                        />
                        <span className="session-actions visible">
                          <button
                            type="button"
                            className="icon-btn"
                            title="保存名称"
                            disabled={sessionBusy}
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => void commitRename()}
                          >
                            ✓
                          </button>
                          <button
                            type="button"
                            className="icon-btn danger"
                            title="取消重命名"
                            disabled={sessionBusy}
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={cancelRename}
                          >
                            ✕
                          </button>
                        </span>
                      </div>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="session-item"
                          disabled={streaming}
                          onClick={() => selectSession(session.id)}
                        >
                          <span className="session-title">
                            {session.title?.trim() ? session.title : '未命名会话'}
                          </span>
                          <span className="session-meta">
                            #{session.id} · {formatDate(session.created_at)}
                          </span>
                        </button>
                        <span className="session-actions">
                          <button
                            type="button"
                            className="icon-btn"
                            title="重命名会话"
                            disabled={streaming || sessionBusy}
                            onClick={() => startRename(session)}
                          >
                            ✎
                          </button>
                          <button
                            type="button"
                            className="icon-btn danger"
                            title="删除会话"
                            disabled={streaming || sessionBusy}
                            onClick={() => void handleDeleteSession(session)}
                          >
                            ✕
                          </button>
                        </span>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </aside>

      <main className="chat">
        <header className="chat-header">
          <span className="chat-title">{activeAgent?.name ?? '请选择角色'}</span>
          {activeAgent && (
            <span className="chat-sub">
              {activeSession !== null && (
                <span className="chat-session-tag">
                  {activeSession.title?.trim() ? activeSession.title : `会话 #${activeSession.id}`}
                </span>
              )}
              {activeAgent.background ?? '（无背景设定）'}
            </span>
          )}
          {activeAgent && <SkillBadges skills={activeAgent.allowed_skills} />}
          {activeAgent && (
            <div className="chat-actions">
              <button
                type="button"
                className={reminderOpen ? 'chat-idle-btn active' : 'chat-idle-btn'}
                disabled={streaming || activeSessionId === null}
                title={
                  activeSessionId === null
                    ? '先创建会话后才能查看提醒'
                    : '查看当前会话的提醒（pending 需确认后才生效）'
                }
                onClick={toggleReminder}
              >
                提醒
                {pendingCount > 0 && <span className="count-badge">{pendingCount}</span>}
              </button>
              <button
                type="button"
                className="chat-idle-btn"
                disabled={streaming}
                title="让角色主动找你聊几句（可反复点击）"
                onClick={() => void handleIdle()}
              >
                开始闲聊
              </button>
              <button
                type="button"
                className="chat-clear-btn"
                disabled={streaming || activeSessionId === null}
                title="清空本会话的全部自主闲聊消息与相关记忆"
                onClick={() => void handleClearIdle()}
              >
                清空闲聊
              </button>
            </div>
          )}
        </header>

        {threadLoading ? (
          <div className="message-list empty">加载对话中…</div>
        ) : activeSessionId === null ? (
          <div className="message-list empty">
            {activeAgent === null ? '请选择左侧角色' : '该角色还没有会话，发条消息即可开始'}
          </div>
        ) : (
          <MessageList
            messages={messages}
            agentName={activeAgent?.name ?? '角色'}
            avatar={activeAgent?.avatar ?? null}
            onRecall={handleRecall}
          />
        )}

        {error !== null && <div className="error-bar">{error}</div>}

        <InputBox
          disabled={activeAgentId === null || streaming}
          streaming={streaming}
          onSend={(text) => void handleSend(text)}
        />

        {reminderOpen && (
          <aside className="reminder-drawer" aria-label="提醒">
            <div className="drawer-head">
              <span className="drawer-title">提醒</span>
              <div className="drawer-actions">
                <button
                  type="button"
                  className="icon-btn"
                  title="刷新提醒列表"
                  disabled={reminderLoading || activeSessionId === null}
                  onClick={handleReminderRefresh}
                >
                  刷新
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  title="关闭提醒面板"
                  onClick={() => setReminderOpen(false)}
                >
                  ✕
                </button>
              </div>
            </div>
            <p className="drawer-sub">
              {activeSession === null
                ? '当前没有会话'
                : `会话 #${activeSession.id} · 待确认 ${pendingCount} 条`}
            </p>

            {reminderLoading ? (
              <p className="hint">加载提醒中…</p>
            ) : reminders.length === 0 ? (
              <p className="hint">
                暂无提醒。角色启用「提醒」工具后，创建的事项会出现在这里等待你确认。
              </p>
            ) : (
              <ul className="reminder-list">
                {reminders.map((reminder) => (
                  <li key={reminder.id} className={`reminder-card ${reminder.status}`}>
                    <div className="reminder-content">{reminder.content}</div>
                    <div className="reminder-meta">
                      <span className={`reminder-status ${reminder.status}`}>
                        {REMINDER_STATUS_LABEL[reminder.status] ?? reminder.status}
                      </span>
                      <span title="提醒时间">{formatDateTime(reminder.remind_at)}</span>
                    </div>
                    {reminder.status === 'pending' && (
                      <div className="reminder-actions">
                        <button
                          type="button"
                          className="btn-primary small"
                          disabled={reminderBusy}
                          onClick={() => void handleConfirmReminder(reminder)}
                        >
                          确认
                        </button>
                        <button
                          type="button"
                          className="btn-ghost small"
                          disabled={reminderBusy}
                          onClick={() => void handleCancelReminder(reminder)}
                        >
                          取消
                        </button>
                      </div>
                    )}
                    {reminder.status === 'active' && (
                      <div className="reminder-actions">
                        <button
                          type="button"
                          className="btn-ghost small"
                          disabled={reminderBusy}
                          onClick={() => void handleCancelReminder(reminder)}
                        >
                          取消提醒
                        </button>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </aside>
        )}
      </main>
    </div>
  )
}
