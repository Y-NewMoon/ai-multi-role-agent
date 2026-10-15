import { useState } from 'react'

import AgentManagePage from './pages/AgentManagePage'
import ChatPage from './pages/ChatPage'
import GroupChatPage from './pages/GroupChatPage'

type Mode = 'single' | 'group' | 'manage'

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
          <button
            type="button"
            aria-selected={mode === 'single'}
            className={mode === 'single' ? 'seg-btn active' : 'seg-btn'}
            onClick={() => setMode('single')}
          >
            单人对话
          </button>
          <button
            type="button"
            aria-selected={mode === 'group'}
            className={mode === 'group' ? 'seg-btn active' : 'seg-btn'}
            onClick={() => setMode('group')}
          >
            群聊
          </button>
          <button
            type="button"
            aria-selected={mode === 'manage'}
            className={mode === 'manage' ? 'seg-btn active' : 'seg-btn'}
            onClick={() => setMode('manage')}
          >
            角色管理
          </button>
        </nav>
      </header>
      {mode === 'single' && <ChatPage />}
      {mode === 'group' && <GroupChatPage />}
      {mode === 'manage' && <AgentManagePage />}
    </div>
  )
}
