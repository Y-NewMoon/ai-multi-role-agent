/** 后端接口封装：REST + SSE 流式（M3.5 最小前端）。 */

export interface Agent {
  id: number
  name: string
  persona_prompt: string
  background: string | null
  avatar: string | null
  /** V2.2 工具白名单；空数组表示该角色不绑定任何工具。 */
  allowed_skills: string[]
  created_at: string
}

export interface Session {
  id: number
  agent_id: number
  title: string | null
  created_at: string
}

/** 展示用消息快照（历史回显）。 */
export interface MessageRecord {
  id: number
  role: string
  content: string
  /** 是否为角色自主闲聊（按钮触发的主动发言）。 */
  is_idle: boolean
  /** 角色该条发言的「内心独白」（思考链）；用户消息/无思考时为 null。 */
  thinking: string | null
  created_at: string
}

/** 群聊展示用消息快照；agent_id 为空表示用户发言。 */
export interface GroupMessageRecord {
  id: number
  role: string
  agent_id: number | null
  content: string
  /** 是否为角色自主闲聊（按钮触发的主动发言）。 */
  is_idle: boolean
  /** 该发言角色的「内心独白」（思考链）；用户发言/无思考时为 null。 */
  thinking: string | null
  created_at: string
}

export interface DonePayload {
  session_id: number
  agent_id: number
  reply: string
  /** M5：本轮角色连发的消息条（1~N 条），按顺序对应界面上的多个气泡。 */
  replies: { content: string; message_id: number; thinking?: string | null }[]
  user_message_id: number
  /** 是否为自主闲聊轮（无用户消息）。 */
  idle?: boolean
}

export interface Group {
  id: number
  name: string
  agent_ids: number[]
  created_at: string
}

export interface GroupSpeakerPayload {
  agent_id: number
  name: string
}

export interface GroupTokenPayload {
  agent_id: number
  text: string
}

export interface GroupTurnDonePayload {
  agent_id: number
  replies: string[]
  reply: string
}

export interface GroupDonePayload {
  group_session_id: number
  message_id: number
  order: number[]
  /** M5：本轮全部角色消息（可能一人多条），按落库顺序。 */
  replies: {
    agent_id: number
    message_id: number
    content: string
    /** 该条发言的「内心独白」（思考链）；无则为 null。 */
    thinking?: string | null
  }[]
  /** 是否为自主闲聊轮（无用户消息）。 */
  idle?: boolean
}

const BASE = '/api/v1'

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    return typeof body?.detail === 'string' ? body.detail : `HTTP ${res.status}`
  } catch {
    return `HTTP ${res.status}`
  }
}

async function request<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(await errorDetail(res))
  }
  return (await res.json()) as T
}

async function requestVoid(res: Response): Promise<void> {
  if (!res.ok) {
    throw new Error(await errorDetail(res))
  }
}

export async function listAgents(): Promise<Agent[]> {
  return request<Agent[]>(await fetch(`${BASE}/agents`))
}

/** 角色可写字段（新建/编辑共用）。 */
export interface AgentInput {
  name: string
  persona_prompt: string
  background: string | null
  avatar: string | null
  allowed_skills: string[]
}

export async function createAgent(input: AgentInput): Promise<Agent> {
  return request<Agent>(
    await fetch(`${BASE}/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),
  )
}

export async function updateAgent(id: number, input: Partial<AgentInput>): Promise<Agent> {
  return request<Agent>(
    await fetch(`${BASE}/agents/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),
  )
}

export async function deleteAgent(id: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/agents/${id}`, { method: 'DELETE' }))
}

/** 清除该角色的记忆：长期记忆 + 全部对话历史（角色档案保留）。 */
export async function clearAgentMemory(id: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/agents/${id}/memory/clear`, { method: 'POST' }))
}

/** 上传头像，返回后端静态 URL（形如 /static/avatars/xxx.png）。 */
export async function uploadAvatar(file: File): Promise<string> {
  const form = new FormData()
  form.append('file', file)
  const body = await request<{ url: string }>(
    await fetch(`${BASE}/uploads/avatar`, { method: 'POST', body: form }),
  )
  return body.url
}

export async function createSession(agentId: number, title?: string): Promise<Session> {
  return request<Session>(
    await fetch(`${BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: agentId, title: title ?? null }),
    }),
  )
}

/** 重命名会话（title ≤255；传 null 表示清空标题）。 */
export async function updateSession(sessionId: number, title: string | null): Promise<Session> {
  return request<Session>(
    await fetch(`${BASE}/sessions/${sessionId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }),
  )
}

/** 删除会话：仅清理该会话的消息/checkpoint/摘要，角色与跨会话记忆保留。 */
export async function deleteSession(sessionId: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' }))
}

/** 某角色的全部会话（按 id 升序）。 */
export async function listSessions(agentId: number): Promise<Session[]> {
  return request<Session[]>(await fetch(`${BASE}/sessions?agent_id=${agentId}`))
}

/** 某会话的历史消息（展示用快照，按时间序）。 */
export async function listMessages(sessionId: number): Promise<MessageRecord[]> {
  return request<MessageRecord[]>(await fetch(`${BASE}/sessions/${sessionId}/messages`))
}

/** 撤回一条用户消息：整轮（含角色回复）连带删除，并清除其产生的记忆。返回被移除的消息 id。 */
export async function recallMessage(sessionId: number, messageId: number): Promise<number[]> {
  const body = await request<{ removed_message_ids: number[] }>(
    await fetch(`${BASE}/sessions/${sessionId}/messages/${messageId}/recall`, { method: 'POST' }),
  )
  return body.removed_message_ids
}

/** 解析一帧 SSE（`event: x` + 若干 `data: y` 行）。 */
function parseFrame(frame: string): { event: string; data: Record<string, unknown> } {
  let event = ''
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  let data: Record<string, unknown> = {}
  if (dataLines.length > 0) {
    try {
      data = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
    } catch {
      data = {}
    }
  }
  return { event, data }
}

export interface StreamHandlers {
  onToken: (text: string) => void
  /** 角色连发的分条边界：界面应在下一条消息上新建气泡。 */
  onBreak: () => void
  onDone: (payload: DonePayload) => void
}

/** 统一读取 SSE 流：按帧回调；`error` 事件直接抛错。 */
async function readSse(
  res: Response,
  onFrame: (event: string, data: Record<string, unknown>) => void,
): Promise<void> {
  if (!res.ok || res.body === null) {
    throw new Error(await errorDetail(res))
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (frame: string) => {
    const { event, data } = parseFrame(frame)
    if (event === 'error') throw new Error(String(data.detail ?? 'stream error'))
    if (event !== '') onFrame(event, data)
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let idx = buffer.indexOf('\n\n')
    while (idx !== -1) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      dispatch(frame)
      idx = buffer.indexOf('\n\n')
    }
  }

  // 流结束：flush 解码器并处理可能残留的最后一帧（服务器未以空行结尾时，
  // 最后一帧仍留在 buffer 中，此前会被丢弃，导致 done 事件不派发）。
  buffer += decoder.decode()
  if (buffer.trim() !== '') dispatch(buffer)
}

/**
 * SSE 流式对话：`POST /chat/stream`。
 *
 * 服务端事件：`token`（增量）/ `done`（本轮结束，含完整 reply）/ `error`（流内异常）。
 * 推流前的校验失败（如 session 不存在）仍是普通 JSON 404，直接抛错。
 */
export async function streamChat(
  sessionId: number,
  message: string,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  })
  await readSse(res, (event, data) => {
    if (event === 'token') {
      handlers.onToken(String(data.text ?? ''))
    } else if (event === 'message_break') {
      handlers.onBreak()
    } else if (event === 'done') {
      handlers.onDone(data as unknown as DonePayload)
    }
  })
}

/**
 * SSE 自主闲聊（私聊）：`POST /chat/idle/stream`。用户点击按钮触发一次；事件同 `streamChat`。
 */
export async function streamChatIdle(
  sessionId: number,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/chat/idle/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  await readSse(res, (event, data) => {
    if (event === 'token') handlers.onToken(String(data.text ?? ''))
    else if (event === 'message_break') handlers.onBreak()
    else if (event === 'done') handlers.onDone(data as unknown as DonePayload)
  })
}

/** 清空该会话的全部自主闲聊消息（含其产生的角色记忆）。返回被移除的消息 id。 */
export async function clearIdle(sessionId: number): Promise<number[]> {
  const body = await request<{ removed_message_ids: number[] }>(
    await fetch(`${BASE}/sessions/${sessionId}/idle/clear`, { method: 'POST' }),
  )
  return body.removed_message_ids
}

// ===== 群聊（M4）=====

export async function listGroups(): Promise<Group[]> {
  return request<Group[]>(await fetch(`${BASE}/groups`))
}

/** 某群的历史消息（展示用快照，按时间序）。 */
export async function listGroupMessages(groupId: number): Promise<GroupMessageRecord[]> {
  return request<GroupMessageRecord[]>(await fetch(`${BASE}/groups/${groupId}/messages`))
}

/** 撤回一条群聊用户消息：整轮（含本轮全部角色回复）连带删除，并清除相关记忆。返回被移除的消息 id。 */
export async function recallGroupMessage(groupId: number, messageId: number): Promise<number[]> {
  const body = await request<{ removed_message_ids: number[] }>(
    await fetch(`${BASE}/groups/${groupId}/messages/${messageId}/recall`, { method: 'POST' }),
  )
  return body.removed_message_ids
}

export async function createGroup(name: string, agentIds: number[]): Promise<Group> {
  return request<Group>(
    await fetch(`${BASE}/groups`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, agent_ids: agentIds }),
    }),
  )
}

export interface GroupStreamHandlers {
  onSpeaker: (p: GroupSpeakerPayload) => void
  onToken: (p: GroupTokenPayload) => void
  /** 某角色连发的分条边界：界面应为同一角色新建气泡。 */
  onBreak: (p: { agent_id: number }) => void
  onTurnDone: (p: GroupTurnDonePayload) => void
  onDone: (p: GroupDonePayload) => void
}

/**
 * SSE 群聊：`POST /groups/{id}/chat/stream`。
 *
 * 服务端事件（ADR-0004 §3.3）：`speaker`（轮到谁）/ `token`（该角色增量）/
 * `turn_done`（该角色本轮结束）/ `done`（整轮结束，快照已落库）/ `error`。
 */
export async function streamGroupChat(
  groupId: number,
  message: string,
  handlers: GroupStreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/groups/${groupId}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  await readSse(res, (event, data) => {
    if (event === 'speaker') handlers.onSpeaker(data as unknown as GroupSpeakerPayload)
    else if (event === 'token') handlers.onToken(data as unknown as GroupTokenPayload)
    else if (event === 'message_break')
      handlers.onBreak(data as unknown as { agent_id: number })
    else if (event === 'turn_done') handlers.onTurnDone(data as unknown as GroupTurnDonePayload)
    else if (event === 'done') handlers.onDone(data as unknown as GroupDonePayload)
  })
}

/**
 * SSE 自主闲聊（群聊）：`POST /groups/{id}/idle/stream`。用户点击按钮触发一次；事件同 `streamGroupChat`。
 */
export async function streamGroupIdle(
  groupId: number,
  handlers: GroupStreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/groups/${groupId}/idle/stream`, { method: 'POST' })
  await readSse(res, (event, data) => {
    if (event === 'speaker') handlers.onSpeaker(data as unknown as GroupSpeakerPayload)
    else if (event === 'token') handlers.onToken(data as unknown as GroupTokenPayload)
    else if (event === 'message_break')
      handlers.onBreak(data as unknown as { agent_id: number })
    else if (event === 'turn_done') handlers.onTurnDone(data as unknown as GroupTurnDonePayload)
    else if (event === 'done') handlers.onDone(data as unknown as GroupDonePayload)
  })
}

/** 清空该群的全部自主闲聊消息（含写入各角色记忆的部分）。返回被移除的消息 id。 */
export async function clearGroupIdle(groupId: number): Promise<number[]> {
  const body = await request<{ removed_message_ids: number[] }>(
    await fetch(`${BASE}/groups/${groupId}/idle/clear`, { method: 'POST' }),
  )
  return body.removed_message_ids
}

/** 删除群：连同该群全部消息与各角色的群聊工作记忆一并清理。 */
export async function deleteGroup(groupId: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/groups/${groupId}`, { method: 'DELETE' }))
}

/** 将某角色移出群；若移出后群内已无成员，则连同该群一并删除。 */
export async function removeGroupMember(groupId: number, agentId: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/groups/${groupId}/members/${agentId}`, { method: 'DELETE' }))
}

// ===== 世界书（V2.1 契约 4.9）=====

/** 文档元数据；后端不返回正文（编辑只能同 slug 重新导入）。 */
export interface LoreDoc {
  id: number
  agent_id: number
  slug: string
  title: string
  version: number
  always_on: boolean
  status: string
  chunks: number
  tokens: number
  created_at: string
  updated_at: string
}

export interface LoreDocInput {
  agent_id: number
  slug: string
  title: string
  content: string
  always_on: boolean
}

export interface LoreImportResult {
  doc_id: number
  version: number
  chunks: number
  tokens: number
  changed: boolean
}

export interface LoreReindexResult {
  agent_id: number
  dry_run: boolean
  documents: number
  chunks_rebuilt: number
}

export async function listLoreDocs(agentId: number): Promise<LoreDoc[]> {
  return request<LoreDoc[]>(await fetch(`${BASE}/lore/docs?agent_id=${agentId}`))
}

/** 导入或更新一篇设定：同 slug 即更新，version+1。 */
export async function importLoreDoc(input: LoreDocInput): Promise<LoreImportResult> {
  return request<LoreImportResult>(
    await fetch(`${BASE}/lore/docs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),
  )
}

/** 软删一篇设定（文档与 chunk 置 deleted 并清索引）。 */
export async function deleteLoreDoc(docId: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/lore/docs/${docId}`, { method: 'DELETE' }))
}

/** 增量重建索引；`dryRun` 只统计不写入（前端「检查」用它判断导入是否完整）。 */
export async function reindexLore(agentId: number, dryRun: boolean): Promise<LoreReindexResult> {
  return request<LoreReindexResult>(
    await fetch(`${BASE}/lore/reindex`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: agentId, dry_run: dryRun }),
    }),
  )
}

// ===== 提醒 HITL（V2.2 契约 §2.8）=====

export interface Reminder {
  id: number
  agent_id: number
  session_id: number
  content: string
  remind_at: string
  /** pending / active / cancelled / fired */
  status: string
  idempotent: boolean
}

export async function listReminders(agentId: number, sessionId: number): Promise<Reminder[]> {
  return request<Reminder[]>(
    await fetch(`${BASE}/reminders?agent_id=${agentId}&session_id=${sessionId}`),
  )
}

/** pending → active（幂等）。 */
export async function confirmReminder(reminderId: number): Promise<Reminder> {
  return request<Reminder>(
    await fetch(`${BASE}/reminders/${reminderId}/confirm`, { method: 'POST' }),
  )
}

/** pending/active → cancelled（幂等）。 */
export async function cancelReminder(reminderId: number): Promise<void> {
  await requestVoid(await fetch(`${BASE}/reminders/${reminderId}`, { method: 'DELETE' }))
}

// ===== 运行状态（V2.4 /health）=====

export interface HealthInfo {
  status: string
  llm_provider: string
  fake_llm_calls: number | null
  database: string | null
  embedding_provider: string
  embedding_model: string | null
  lore_enabled: boolean
  lore_static_mode: boolean
  memory_extract_enabled: boolean
  memory_profile_enabled: boolean
  memory_l2_enabled: boolean
  memory_weighted_recall: boolean
  trace_enabled: boolean
  trace_dir: string
  trace_sample_rate: number
}

/** 健康检查不在 /api/v1 下；开发代理已在 vite.config.ts 中转发 /health。 */
export async function getHealth(): Promise<HealthInfo> {
  return request<HealthInfo>(await fetch('/health'))
}
