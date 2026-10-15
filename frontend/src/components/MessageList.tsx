import { useEffect, useRef, useState } from 'react'

import type { ChatMessage } from '../pages/ChatPage'

interface Props {
  messages: ChatMessage[]
  agentName: string
  avatar: string | null
  onRecall: (message: ChatMessage) => void
}

/** 头像为上传的图片 URL（/static/... 或 http...）时用图片渲染，否则回退首字。 */
function isImageAvatar(avatar: string | null): avatar is string {
  return avatar !== null && (avatar.startsWith('/') || avatar.startsWith('http'))
}

export default function MessageList({ messages, agentName, avatar, onRecall }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  // 「思考」展开状态：按消息下标记录（默认全部收起，点按钮才出现）。
  const [openThinks, setOpenThinks] = useState<Set<number>>(new Set())

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  function toggleThink(index: number) {
    setOpenThinks((prev) => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  if (messages.length === 0) {
    return <div className="message-list empty">发条消息，开始对话吧</div>
  }

  return (
    <div className="message-list">
      {messages.map((msg, index) => {
        const thinking = msg.role === 'assistant' ? (msg.thinking ?? '') : ''
        const hasThink = thinking !== ''
        const open = openThinks.has(index)
        return (
          <div key={index} className={msg.role === 'user' ? 'msg user' : 'msg assistant'}>
            <div className="msg-role">
              {msg.role === 'assistant' &&
                (isImageAvatar(avatar) ? (
                  <img className="msg-avatar" src={avatar} alt="" />
                ) : (
                  <span className="msg-avatar">{agentName.slice(0, 1)}</span>
                ))}
              <span>{msg.role === 'user' ? '我' : agentName}</span>
              {msg.role === 'assistant' && msg.isIdle && (
                <span className="msg-idle-tag" title="角色自主闲聊（可用「清空闲聊」清除）">
                  闲聊
                </span>
              )}
              {msg.role === 'assistant' && hasThink && (
                <button
                  type="button"
                  className="msg-think-btn"
                  title="查看这条发言的思考过程"
                  onClick={() => toggleThink(index)}
                >
                  {open ? '收起思考' : '思考'}
                </button>
              )}
              {msg.role === 'user' && msg.id > 0 && (
                <button
                  type="button"
                  className="msg-recall"
                  title="撤回这条消息（连同角色回复与该轮产生的记忆）"
                  onClick={() => onRecall(msg)}
                >
                  撤回
                </button>
              )}
            </div>
            <div className="msg-bubble">
              {msg.content !== '' ? msg.content : <span className="typing">正在输入…</span>}
            </div>
            {msg.role === 'assistant' && hasThink && open && (
              <div className="msg-think" title="角色这条发言的内心独白">
                {thinking}
              </div>
            )}
          </div>
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
