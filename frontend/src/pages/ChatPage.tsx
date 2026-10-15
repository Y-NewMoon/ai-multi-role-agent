import { useEffect, useMemo, useRef, useState } from 'react'

import {
  clearIdle,
  createSession,
  listAgents,
  listMessages,
  listSessions,
  recallMessage,
  streamChat,
  streamChatIdle,
  type Agent,
} from '../api/client'
import AgentList from '../components/AgentList'
import InputBox from '../components/InputBox'
import MessageList from '../components/MessageList'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /** 库中消息 id；0 表示尚未落库（发送中的占位消息），此时不提供撤回。 */
  id: number
  /** 该消息所属会话 id；用于撤回时定位对应 thread。 */
  sessionId: number | null
  /** 是否为角色自主闲聊（按钮触发的主动发言）。 */
  isIdle?: boolean
  /** 角色该条发言的「内心独白」（思考链）；无则不显示「思考」按钮。 */
  thinking?: string | null
}

/** 单个角色的会话状态：惰性创建 session，消息只存在前端内存。 */
interface AgentChat {
  sessionId: number | null
  messages: ChatMessage[]
}

const EMPTY_CHAT: AgentChat = { sessionId: null, messages: [] }

export default function ChatPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [chats, setChats] = useState<Record<number, AgentChat>>({})
  const [streaming, setStreaming] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listAgents()
      .then((list) => {
        setAgents(list)
        setActiveId(list.length > 0 ? list[0].id : null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const activeAgent = useMemo(
    () => agents.find((a) => a.id === activeId) ?? null,
    [agents, activeId],
  )
  const activeChat = activeId === null ? EMPTY_CHAT : (chats[activeId] ?? EMPTY_CHAT)

  function patchChat(agentId: number, patch: (prev: AgentChat) => AgentChat) {
    setChats((prev) => ({ ...prev, [agentId]: patch(prev[agentId] ?? EMPTY_CHAT) }))
  }

  /** 切换角色时从后端拉取历史消息（仅一次；已加载的会话不会被覆盖）。 */
  const loadedRef = useRef<Set<number>>(new Set())
  useEffect(() => {
    if (activeId === null || loadedRef.current.has(activeId)) return
    loadedRef.current.add(activeId)
    const agentId = activeId
    let cancelled = false
    ;(async () => {
      try {
        const sessions = await listSessions(agentId)
        if (cancelled || sessions.length === 0) return
        // 历史可能分散在多个会话（早期每次刷新都会新建会话）：合并按时间序展示；
        // 后续新消息仍写入最近一个会话，会话数不再增长。
        const perSession = await Promise.all(sessions.map((s) => listMessages(s.id)))
        if (cancelled) return
        const messages: ChatMessage[] = sessions.flatMap((s, i) =>
          perSession[i]
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map((m) => ({
              role: m.role as 'user' | 'assistant',
              content: m.content,
              id: m.id,
              sessionId: s.id,
              isIdle: m.is_idle,
              thinking: m.thinking,
            })),
        )
        patchChat(agentId, () => ({ sessionId: sessions[sessions.length - 1].id, messages }))
      } catch (e) {
        loadedRef.current.delete(agentId)
        if (!cancelled) setError((e as Error).message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeId])

  /** 追加一段文本到该角色最后一条 assistant 消息（首次为空串占位）。 */
  function appendAssistant(agentId: number, text: string) {
    patchChat(agentId, (chat) => {
      const messages = chat.messages.slice()
      const last = messages[messages.length - 1]
      messages[messages.length - 1] = { ...last, content: last.content + text }
      return { ...chat, messages }
    })
  }

  /** 在末尾追加一个空的 assistant 占位气泡（角色连发多条时用）。 */
  function pushAssistant(agentId: number, isIdle = false) {
    patchChat(agentId, (chat) => ({
      ...chat,
      messages: [
        ...chat.messages,
        { role: 'assistant', content: '', id: 0, sessionId: chat.sessionId, isIdle },
      ],
    }))
  }

  async function handleSend(text: string) {
    if (activeId === null || streaming) return
    const agentId = activeId
    const sessionId = activeChat.sessionId

    setError(null)
    setStreaming(true)
    patchChat(agentId, (chat) => ({
      ...chat,
      messages: [
        ...chat.messages,
        { role: 'user', content: text, id: 0, sessionId: chat.sessionId },
        { role: 'assistant', content: '', id: 0, sessionId: chat.sessionId },
      ],
    }))

    try {
      // 首次发言才建会话（每个角色一条；多会话并存能力由后端提供）
      const sid = sessionId ?? (await createSession(agentId)).id
      if (sessionId === null) {
        patchChat(agentId, (chat) => ({ ...chat, sessionId: sid }))
      }

      await streamChat(sid, text, {
        onToken: (token) => appendAssistant(agentId, token),
        onBreak: () => pushAssistant(agentId),
        onDone: (payload) =>
          patchChat(agentId, (chat) => {
            // 回填本轮各条消息的真实 id：先定位刚发出的用户占位，再顺序回填其后各 assistant 气泡
            const messages = chat.messages.map((m) => ({ ...m }))
            let uid = -1
            for (let i = messages.length - 1; i >= 0; i--) {
              if (messages[i].role === 'user' && messages[i].id === 0) {
                uid = i
                break
              }
            }
            if (uid < 0) return chat
            messages[uid] = { ...messages[uid], id: payload.user_message_id, sessionId: payload.session_id }
            let k = 0
            for (let i = uid + 1; i < messages.length && k < payload.replies.length; i++) {
              if (messages[i].role === 'assistant') {
                messages[i] = {
                  ...messages[i],
                  content: payload.replies[k].content,
                  id: payload.replies[k].message_id,
                  sessionId: payload.session_id,
                  thinking: payload.replies[k].thinking,
                }
                k++
              }
            }
            // 清理未被回填的空白助手占位（如模型多输出分隔符导致的幽灵气泡），
            // 否则会永久停留在「正在输入…」。
            return {
              ...chat,
              messages: messages.filter(
                (m) => !(m.role === 'assistant' && m.id === 0 && m.content === ''),
              ),
            }
          }),
      })
    } catch (e) {
      setError((e as Error).message)
      // 失败时移除本轮遗留的空助手气泡，避免留下空白回复
      patchChat(agentId, (chat) => ({
        ...chat,
        messages: chat.messages.filter((m) => !(m.role === 'assistant' && m.id === 0 && m.content === '')),
      }))
    } finally {
      setStreaming(false)
    }
  }

  /** 触发一次自主闲聊：角色主动发消息（无用户消息）。可反复点击。 */
  async function handleIdle() {
    if (activeId === null || streaming) return
    const agentId = activeId
    const sessionId = activeChat.sessionId
    setError(null)
    setStreaming(true)

    try {
      // 闲聊也需要一个会话承载；首次触发时惰性创建
      const sid = sessionId ?? (await createSession(agentId)).id
      if (sessionId === null) {
        patchChat(agentId, (chat) => ({ ...chat, sessionId: sid }))
      }
      pushAssistant(agentId, true) // 起始占位气泡（标记为闲聊）
      await streamChatIdle(sid, {
        onToken: (token) => appendAssistant(agentId, token),
        onBreak: () => pushAssistant(agentId, true),
        onDone: (payload) =>
          patchChat(agentId, (chat) => {
            // 闲聊无用户消息：把末尾连续的「未落库 assistant 占位」直接替换为落库后的真实回复。
            // 用 replies 重建这段气泡可同时：① 消除模型多输出分隔符产生的空占位（幽灵气泡）；
            // ② 保证每个气泡都带上真实 id，使「清空闲聊」能精确定位删除。
            let start = chat.messages.length
            while (
              start > 0 &&
              chat.messages[start - 1].role === 'assistant' &&
              chat.messages[start - 1].id === 0
            ) {
              start--
            }
            const head = chat.messages.slice(0, start)
            const tail: ChatMessage[] = payload.replies.map((r) => ({
              role: 'assistant',
              content: r.content,
              id: r.message_id,
              sessionId: payload.session_id,
              isIdle: true,
              thinking: r.thinking,
            }))
            return { ...chat, messages: [...head, ...tail] }
          }),
      })
    } catch (e) {
      setError((e as Error).message)
      patchChat(agentId, (chat) => ({
        ...chat,
        messages: chat.messages.filter(
          (m) => !(m.role === 'assistant' && m.id === 0 && m.content === ''),
        ),
      }))
    } finally {
      setStreaming(false)
    }
  }

  /** 清空当前会话的全部自主闲聊消息（含其写入的角色记忆）。 */
  async function handleClearIdle() {
    if (activeId === null || streaming) return
    const agentId = activeId
    const sessionId = activeChat.sessionId
    if (sessionId === null) return
    setError(null)
    try {
      const removed = await clearIdle(sessionId)
      const removedSet = new Set(removed)
      patchChat(agentId, (chat) => ({
        ...chat,
        // 删除后端返回的闲聊消息；同时清除未落库的闲聊占位（id===0），避免残留
        messages: chat.messages.filter(
          (m) => !removedSet.has(m.id) && !(m.role === 'assistant' && m.id === 0 && m.isIdle === true),
        ),
      }))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 撤回一条用户消息：后端连带删除角色回复与该轮产生的记忆，前端同步移除气泡。 */
  async function handleRecall(message: ChatMessage) {
    if (activeId === null || streaming) return
    if (message.role !== 'user' || message.id <= 0 || message.sessionId === null) return
    const agentId = activeId
    setError(null)
    try {
      const removed = await recallMessage(message.sessionId, message.id)
      const removedSet = new Set(removed)
      patchChat(agentId, (chat) => ({
        ...chat,
        messages: chat.messages.filter((m) => !removedSet.has(m.id)),
      }))
    } catch (e) {
      setError((e as Error).message)
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
            activeId={activeId}
            disabled={streaming}
            onSelect={setActiveId}
          />
        )}
      </aside>

      <main className="chat">
        <header className="chat-header">
          <span className="chat-title">{activeAgent?.name ?? '请选择角色'}</span>
          {activeAgent && (
            <span className="chat-sub">{activeAgent.background ?? '（无背景设定）'}</span>
          )}
          {activeAgent && (
            <div className="chat-actions">
              <button
                type="button"
                className="chat-idle-btn"
                disabled={streaming}
                title="让角色主动找你聊几句（可反复点击）"
                onClick={handleIdle}
              >
                开始闲聊
              </button>
              <button
                type="button"
                className="chat-clear-btn"
                disabled={streaming}
                title="清空本会话的全部自主闲聊消息与相关记忆"
                onClick={handleClearIdle}
              >
                清空闲聊
              </button>
            </div>
          )}
        </header>

        <MessageList
          messages={activeChat.messages}
          agentName={activeAgent?.name ?? '角色'}
          avatar={activeAgent?.avatar ?? null}
          onRecall={handleRecall}
        />

        {error !== null && <div className="error-bar">{error}</div>}

        <InputBox
          disabled={activeId === null || streaming}
          streaming={streaming}
          onSend={handleSend}
        />
      </main>
    </div>
  )
}
