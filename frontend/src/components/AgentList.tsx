import type { Agent } from '../api/client'

interface Props {
  agents: Agent[]
  activeId: number | null
  disabled: boolean
  onSelect: (id: number) => void
}

/** 头像为上传的图片 URL（/static/... 或 http...）时用图片渲染，否则回退首字。 */
function isImageAvatar(avatar: string | null): avatar is string {
  return avatar !== null && (avatar.startsWith('/') || avatar.startsWith('http'))
}

export default function AgentList({ agents, activeId, disabled, onSelect }: Props) {
  if (agents.length === 0) {
    return <p className="hint">暂无角色，请到「角色管理」新建。</p>
  }

  return (
    <ul className="agent-list">
      {agents.map((agent) => (
        <li key={agent.id}>
          <button
            type="button"
            className={agent.id === activeId ? 'agent-item active' : 'agent-item'}
            disabled={disabled}
            onClick={() => onSelect(agent.id)}
          >
            {isImageAvatar(agent.avatar) ? (
              <img className="agent-avatar" src={agent.avatar} alt="" />
            ) : (
              <span className="agent-avatar">{agent.name.slice(0, 1)}</span>
            )}
            <span className="agent-name">{agent.name}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}
