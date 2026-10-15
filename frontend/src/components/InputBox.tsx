import { useState, type KeyboardEvent } from 'react'

interface Props {
  disabled: boolean
  streaming: boolean
  onSend: (text: string) => void
}

export default function InputBox({ disabled, streaming, onSend }: Props) {
  const [text, setText] = useState('')

  function submit() {
    const value = text.trim()
    if (value === '' || disabled) return
    onSend(value)
    setText('')
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="input-box">
      <textarea
        rows={2}
        value={text}
        disabled={disabled}
        placeholder={streaming ? '角色正在回复…' : '输入消息，Enter 发送，Shift+Enter 换行'}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button type="button" disabled={disabled || text.trim() === ''} onClick={submit}>
        {streaming ? '回复中' : '发送'}
      </button>
    </div>
  )
}
