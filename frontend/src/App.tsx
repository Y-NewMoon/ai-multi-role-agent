import { useState } from 'react'

import HealthBadge from './components/HealthBadge'
import AgentManagePage from './pages/AgentManagePage'
import ChatPage from './pages/ChatPage'
import GroupChatPage from './pages/GroupChatPage'
import LorePage from './pages/LorePage'

type Mode = 'single' | 'group' | 'lore' | 'manage'

const MODES: { value: Mode; label: string }[] = [
  { value: 'single', label: '单人对话' },
  { value: 'group', label: '群聊' },
  { value: 'lore', label: '世界书' },
  { value: 'manage', label: '角色管理' },
]

export default function App() {
  const [mode, setMode] = useState<Mode>('single')

  return (
    <div className="app-root">
      <header className="topbar">
        <div className="topbar-brand">
          <span className="brand-mark">AI</span>
          <span className="brand-text">多角色扮演</span>
        </div>
        <nav className="segmented" aria-label="对话模式">
          {MODES.map((item) => (
            <button
              key={item.value}
              type="button"
              aria-selected={mode === item.value}
              className={mode === item.value ? 'seg-btn active' : 'seg-btn'}
              onClick={() => setMode(item.value)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <HealthBadge />
      </header>
      {mode === 'single' && <ChatPage />}
      {mode === 'group' && <GroupChatPage />}
      {mode === 'lore' && <LorePage />}
      {mode === 'manage' && <AgentManagePage />}
    </div>
  )
}
