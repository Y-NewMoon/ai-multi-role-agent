import { useEffect, useMemo, useRef, useState } from 'react'

import {
  clearGroupIdle,
  createGroup,
  deleteGroup,
  listAgents,
  listGroupMessages,
  listGroups,
  recallGroupMessage,
  removeGroupMember,
  streamGroupChat,
  streamGroupIdle,
  type Agent,
  type Group,
} from '../api/client'
import InputBox from '../components/InputBox'

/** 群聊消息：assistant 需带上发言人（同一轮不同角色轮流发言）。 */
interface GroupMessage {
  /** 前端本地键（React key + 流式定位）；与库中 id 无关。 */
  id: number
  /** 库中消息 id；null 表示尚未落库（流式中），此时不提供撤回。 */
  dbId: number | null
  role: 'user' | 'assistant'
  speakerId: number | null
  speakerName: string
  content: string
  /** 是否为角色自主闲聊（按钮触发的主动发言）。 */
  isIdle: boolean
  /** 该发言角色的「内心独白」（思考链）；无则不显示「思考」按钮。 */
  thinking?: string | null
}

const MAX_MEMBERS = 8 // 与后端 group_max_members 一致

/** 头像为上传的图片 URL（/static/... 或 http...）时用图片渲染，否则回退首字。 */
function isImageAvatar(avatar: string | null): avatar is string {
  return avatar !== null && (avatar.startsWith('/') || avatar.startsWith('http'))
}

export default function GroupChatPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [groups, setGroups] = useState<Group[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [messages, setMessages] = useState<GroupMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // 「思考」展开状态：按本地消息 key 记录（默认全部收起，点按钮才出现）。
  const [openThinks, setOpenThinks] = useState<Set<number>>(new Set())

  // 新建群聊表单
  const [creating, setCreating] = useState(false)
  const [groupName, setGroupName] = useState('')
  const [picked, setPicked] = useState<number[]>([])

  const seqRef = useRef(0)
  const currentIdRef = useRef(-1)
  const currentSpeakerRef = useRef<{ id: number; name: string } | null>(null)
  /** 当前气泡已累积的文本，用于识别「空段 break」（模型在开头多输出的分隔符）。 */
  const currentTextRef = useRef('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    Promise.all([listAgents(), listGroups()])
      .then(([agentList, groupList]) => {
        setAgents(agentList)
        setGroups(groupList)
        setActiveId(groupList.length > 0 ? groupList[0].id : null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  /** 切换群聊时从后端拉取历史消息（群聊消息数组不按群缓存，切换即重取）。 */
  useEffect(() => {
    if (activeId === null) {
      setMessages([])
      return
    }
    const groupId = activeId
    let cancelled = false
    setOpenThinks(new Set()) // 切换群聊时重置「思考」展开状态
    ;(async () => {
      try {
        const records = await listGroupMessages(groupId)
        if (cancelled) return
        const loaded: GroupMessage[] = records.map((m) => ({
          id: m.id,
          dbId: m.id,
          role: m.role === 'user' ? 'user' : 'assistant',
          speakerId: m.agent_id,
          speakerName: m.agent_id === null ? '我' : nameOf(m.agent_id),
          content: m.content,
          isIdle: m.is_idle,
          thinking: m.thinking,
        }))
        // seqRef 抬到已加载最大 id 之上，避免新增消息的本地 id 与库中 id 撞键
        seqRef.current = loaded.reduce((mx, m) => Math.max(mx, m.id), 0)
        setMessages(loaded)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [activeId])

  const activeGroup = useMemo(
    () => groups.find((g) => g.id === activeId) ?? null,
    [groups, activeId],
  )

  function nameOf(agentId: number): string {
    return agents.find((a) => a.id === agentId)?.name ?? `角色#${agentId}`
  }

  function avatarOf(agentId: number | null): string | null {
    if (agentId === null) return null
    return agents.find((a) => a.id === agentId)?.avatar ?? null
  }

  function selectGroup(id: number) {
    if (streaming || id === activeId) return
    setActiveId(id)
    setMessages([]) // 立即清空，历史随后由 effect 从后端拉取
    setError(null)
  }

  function togglePicked(id: number) {
    setPicked((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      return prev.length >= MAX_MEMBERS ? prev : [...prev, id]
    })
  }

  /** 展开/收起某条消息的思考过程（按本地消息 key）。 */
  function toggleThink(id: number) {
    setOpenThinks((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  async function handleCreate() {
    const name = groupName.trim()
    if (name === '' || picked.length === 0) return
    setError(null)
    try {
      const group = await createGroup(name, picked)
      setGroups((prev) => [...prev, group])
      setActiveId(group.id)
      setMessages([])
      setCreating(false)
      setGroupName('')
      setPicked([])
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 构造群聊 SSE 回调：正常轮（idle=false）与闲聊轮（idle=true）共用。 */
  function streamHandlers({ idle, uid }: { idle: boolean; uid: number }) {
    return {
      onSpeaker: (p: { agent_id: number; name: string }) => {
        const id = ++seqRef.current
        currentIdRef.current = id
        currentSpeakerRef.current = { id: p.agent_id, name: p.name }
        currentTextRef.current = ''
        setMessages((prev) => [
          ...prev,
          {
            id,
            dbId: null,
            role: 'assistant' as const,
            speakerId: p.agent_id,
            speakerName: p.name,
            content: '',
            isIdle: idle,
          },
        ])
      },
      onToken: (p: { text: string }) => {
        const id = currentIdRef.current
        if (id < 0) return
        currentTextRef.current += p.text
        setMessages((prev) =>
          prev.map((m) => (m.id === id ? { ...m, content: m.content + p.text } : m)),
        )
      },
      onBreak: () => {
        // 同一角色连发多条：为它新起一个气泡。
        // 若当前气泡尚无内容，说明模型在开头多输出了分隔符（服务端会过滤掉该空段），
        // 此时忽略这次 break，否则会多出一个空气泡并与后续发言错位。
        const speaker = currentSpeakerRef.current
        if (speaker === null || currentTextRef.current === '') return
        const id = ++seqRef.current
        currentIdRef.current = id
        currentTextRef.current = ''
        setMessages((prev) => [
          ...prev,
          {
            id,
            dbId: null,
            role: 'assistant' as const,
            speakerId: speaker.id,
            speakerName: speaker.name,
            content: '',
            isIdle: idle,
          },
        ])
      },
      onTurnDone: (p: { replies: string[] }) => {
        // 兜底：若本轮未收到任何 token，则以 turn_done 的完整回复收尾
        const id = currentIdRef.current
        if (id < 0) return
        setMessages((prev) =>
          prev.map((m) =>
            m.id === id && m.content === '' ? { ...m, content: p.replies.join('\n') } : m,
          ),
        )
      },
      onDone: (p: {
        message_id: number
        replies: {
          agent_id: number
          message_id: number
          content: string
          thinking?: string | null
        }[]
      }) => {
        // 按落库顺序逐条回填本轮各气泡的真实 id 与最终内容，供撤回使用。
        // 注意：计数器必须声明在 updater 内部 —— StrictMode 会重复调用 updater，
        // 若把 k 放在外部会被第一次调用耗尽，导致第二次（生效的那次）不回填。
        setMessages((prev) => {
          let k = 0
          const mapped = prev.map((m) => {
            if (!idle && m.role === 'user' && m.id === uid) return { ...m, dbId: p.message_id }
            if (m.role === 'assistant' && m.dbId === null && k < p.replies.length) {
              const r = p.replies[k]
              k++
              // 以落库结果为准回填「发言人 + 内容 + 思考 + id」，避免流式 break 造成的错位
              return {
                ...m,
                dbId: r.message_id,
                content: r.content,
                speakerId: r.agent_id,
                speakerName: nameOf(r.agent_id),
                thinking: r.thinking,
              }
            }
            return m
          })
          // 落库条数少于气泡数时（模型多输出分隔符产生的空占位），移除未被回填的气泡，
          // 否则会残留「幽灵空气泡」。
          return mapped.filter((m) => !(m.role === 'assistant' && m.dbId === null))
        })
      },
    }
  }

  async function handleSend(text: string) {
    if (activeId === null || streaming) return
    const groupId = activeId
    setError(null)
    setStreaming(true)

    const uid = ++seqRef.current
    setMessages((prev) => [
      ...prev,
      {
        id: uid,
        dbId: null,
        role: 'user',
        speakerId: null,
        speakerName: '我',
        content: text,
        isIdle: false,
      },
    ])
    currentIdRef.current = -1

    try {
      await streamGroupChat(groupId, text, streamHandlers({ idle: false, uid }))
    } catch (e) {
      setError((e as Error).message)
      // 失败时移除本轮遗留的空助手气泡
      setMessages((prev) => prev.filter((m) => !(m.role === 'assistant' && m.content === '')))
    } finally {
      setStreaming(false)
    }
  }

  /** 触发一次群聊自主闲聊：角色之间互相搭话（无用户消息）。可反复点击。 */
  async function handleIdle() {
    if (activeId === null || streaming) return
    const groupId = activeId
    setError(null)
    setStreaming(true)
    currentIdRef.current = -1
    try {
      await streamGroupIdle(groupId, streamHandlers({ idle: true, uid: -1 }))
    } catch (e) {
      setError((e as Error).message)
      setMessages((prev) => prev.filter((m) => !(m.role === 'assistant' && m.content === '')))
    } finally {
      setStreaming(false)
    }
  }

  /** 清空该群的全部自主闲聊消息（含写入各角色记忆的部分）。 */
  async function handleClearIdle() {
    if (activeId === null || streaming) return
    const groupId = activeId
    setError(null)
    try {
      const removed = await clearGroupIdle(groupId)
      const removedSet = new Set(removed)
      setMessages((prev) =>
        prev.filter((m) => {
          if (m.dbId !== null) return !removedSet.has(m.dbId)
          // 未落库的本地气泡：闲聊占位（如流式中断/幽灵气泡）也应一并清除
          return !m.isIdle
        }),
      )
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 撤回一条群聊用户消息：后端连带删除本轮全部角色回复与相关记忆，前端同步移除气泡。 */
  async function handleRecall(message: GroupMessage) {
    if (activeId === null || streaming) return
    if (message.role !== 'user' || message.dbId === null) return
    const groupId = activeId
    setError(null)
    try {
      const removed = await recallGroupMessage(groupId, message.dbId)
      const removedSet = new Set(removed)
      setMessages((prev) => prev.filter((m) => m.dbId === null || !removedSet.has(m.dbId)))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 删除当前群聊：连同该群全部消息与各角色在群里的记忆一并清理。 */
  async function handleDeleteGroup() {
    if (activeId === null || streaming) return
    const groupId = activeId
    if (!window.confirm(`确定删除群聊「${activeGroup?.name ?? ''}」吗？该群消息与角色在群里的记忆都会被清除。`))
      return
    setError(null)
    try {
      await deleteGroup(groupId)
      const left = groups.filter((g) => g.id !== groupId)
      setGroups(left)
      setActiveId(left.length > 0 ? left[0].id : null)
      setMessages([])
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** 把一个角色移出群聊；若移出后群内已无成员，后端会连同该群一并删除。 */
  async function handleRemoveMember(agentId: number) {
    if (activeId === null || streaming) return
    const groupId = activeId
    if (!window.confirm(`确定把「${nameOf(agentId)}」移出群聊吗？`)) return
    setError(null)
    try {
      await removeGroupMember(groupId, agentId)
      const nextIds = activeGroup?.agent_ids.filter((id) => id !== agentId) ?? []
      if (nextIds.length === 0) {
        const left = groups.filter((g) => g.id !== groupId)
        setGroups(left)
        setActiveId(left.length > 0 ? left[0].id : null)
        setMessages([])
      } else {
        setGroups((prev) =>
          prev.map((g) => (g.id === groupId ? { ...g, agent_ids: nextIds } : g)),
        )
      }
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="side-title">会话</div>

        {creating ? (
          <div className="group-create">
            <input
              className="group-name-input"
              placeholder="群名称"
              value={groupName}
              onChange={(e) => setGroupName(e.target.value)}
            />
            <p className="hint">
              选择成员（最多 {MAX_MEMBERS} 个，已选 {picked.length}）
            </p>
            <ul className="agent-list">
              {agents.map((a) => (
                <li key={a.id}>
                  <label className="agent-item">
                    <input
                      type="checkbox"
                      checked={picked.includes(a.id)}
                      onChange={() => togglePicked(a.id)}
                    />
                    <span className="agent-name">{a.name}</span>
                  </label>
                </li>
              ))}
            </ul>
            <div className="group-create-actions">
              <button
                type="button"
                className="primary"
                disabled={groupName.trim() === '' || picked.length === 0}
                onClick={handleCreate}
              >
                创建
              </button>
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  setCreating(false)
                  setGroupName('')
                  setPicked([])
                }}
              >
                取消
              </button>
            </div>
          </div>
        ) : (
          <>
            <button
              type="button"
              className="new-group-btn"
              disabled={streaming}
              onClick={() => setCreating(true)}
            >
              ＋ 新建群聊
            </button>
            {loading ? (
              <p className="hint">加载中…</p>
            ) : groups.length === 0 ? (
              <p className="hint">暂无群聊</p>
            ) : (
              <ul className="agent-list">
                {groups.map((g) => (
                  <li key={g.id}>
                    <button
                      type="button"
                      className={g.id === activeId ? 'agent-item active' : 'agent-item'}
                      disabled={streaming}
                      onClick={() => selectGroup(g.id)}
                    >
                      <span className="agent-avatar">群</span>
                      <span className="agent-name">{g.name}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </aside>

      <main className="chat">
        <header className="chat-header">
          <span className="chat-title">{activeGroup?.name ?? '请选择群聊'}</span>
          {activeGroup && (
            <span className="chat-sub">
              {activeGroup.agent_ids.map((aid) => (
                <span key={aid} className="chat-member">
                  {nameOf(aid)}
                  <button
                    type="button"
                    className="chat-member-remove"
                    disabled={streaming}
                    title={`把「${nameOf(aid)}」移出群聊`}
                    onClick={() => handleRemoveMember(aid)}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </span>
          )}
          {activeGroup && (
            <div className="chat-actions">
              <button
                type="button"
                className="chat-idle-btn"
                disabled={streaming}
                title="让群里的角色们自己聊几句（可反复点击）"
                onClick={handleIdle}
              >
                开始闲聊
              </button>
              <button
                type="button"
                className="chat-clear-btn"
                disabled={streaming}
                title="清空本群的全部自主闲聊消息与相关记忆"
                onClick={handleClearIdle}
              >
                清空闲聊
              </button>
              <button
                type="button"
                className="chat-clear-btn"
                disabled={streaming}
                title="删除本群聊（连同群内消息与角色在群里的记忆）"
                onClick={handleDeleteGroup}
              >
                删除群聊
              </button>
            </div>
          )}
        </header>

        {messages.length === 0 ? (
          <div className="message-list empty">
            {activeGroup ? '发条消息，让群里聊起来' : '先在左侧新建一个群聊'}
          </div>
        ) : (
          <div className="message-list">
            {messages.map((m) => {
              const avatar = avatarOf(m.speakerId)
              return (
                <div key={m.id} className={m.role === 'user' ? 'msg user' : 'msg assistant'}>
                  <div className="msg-role">
                    {m.speakerId !== null &&
                      (isImageAvatar(avatar) ? (
                        <img className="msg-avatar" src={avatar} alt="" />
                      ) : (
                        <span className="msg-avatar">{m.speakerName.slice(0, 1)}</span>
                      ))}
                    <span>{m.speakerName}</span>
                    {m.isIdle && (
                      <span className="msg-idle-tag" title="角色自主闲聊（可用「清空闲聊」清除）">
                        闲聊
                      </span>
                    )}
                    {m.role === 'assistant' && (m.thinking ?? '') !== '' && (
                      <button
                        type="button"
                        className="msg-think-btn"
                        title="查看这条发言的思考过程"
                        onClick={() => toggleThink(m.id)}
                      >
                        {openThinks.has(m.id) ? '收起思考' : '思考'}
                      </button>
                    )}
                    {m.role === 'user' && m.dbId !== null && (
                      <button
                        type="button"
                        className="msg-recall"
                        title="撤回这条消息（连同本轮角色回复与该轮产生的记忆）"
                        onClick={() => handleRecall(m)}
                      >
                        撤回
                      </button>
                    )}
                  </div>
                  <div className="msg-bubble">
                    {m.content !== '' ? m.content : <span className="typing">正在输入…</span>}
                  </div>
                  {m.role === 'assistant' && (m.thinking ?? '') !== '' && openThinks.has(m.id) && (
                    <div className="msg-think" title="角色这条发言的内心独白">
                      {m.thinking}
                    </div>
                  )}
                </div>
              )
            })}
            <div ref={bottomRef} />
          </div>
        )}

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
