import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Bot, User } from 'lucide-react'
import { motion } from 'framer-motion'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello, I'm MindGuard's empathetic AI companion. I'm here to listen and provide support. How are you feeling today?",
      state: null,
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        message: userMessage,
        history: messages.map((m) => ({ role: m.role, content: m.content })),
      })

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.data.response,
          state: response.data.detected_state,
          confidence: response.data.confidence,
        },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: "I'm sorry, I'm having trouble connecting right now. If you're in crisis, please call 988.",
          state: 'error',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const stateColors = {
    normal: 'bg-green-100 text-green-700',
    stress: 'bg-yellow-100 text-yellow-700',
    anxiety: 'bg-orange-100 text-orange-700',
    depression: 'bg-red-100 text-red-700',
    severe: 'bg-red-200 text-red-800',
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 max-w-3xl mx-auto">
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
            <Bot className="w-5 h-5 text-green-600" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900">Empathetic AI Companion</h2>
            <p className="text-xs text-gray-500">Powered by BERT + RAG Pipeline</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="h-[500px] overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
              msg.role === 'user' ? 'bg-blue-100' : 'bg-green-100'
            }`}>
              {msg.role === 'user' ? (
                <User className="w-4 h-4 text-blue-600" />
              ) : (
                <Bot className="w-4 h-4 text-green-600" />
              )}
            </div>
            <div className={`max-w-[75%] ${msg.role === 'user' ? 'text-right' : ''}`}>
              <div className={`p-3 rounded-2xl text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-none'
                  : 'bg-gray-100 text-gray-800 rounded-tl-none'
              }`}>
                {msg.content}
              </div>
              {msg.state && msg.state !== 'error' && (
                <span className={`inline-block mt-1 px-2 py-0.5 rounded-full text-xs ${stateColors[msg.state] || ''}`}>
                  Detected: {msg.state} ({(msg.confidence * 100).toFixed(0)}%)
                </span>
              )}
            </div>
          </motion.div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-green-100 flex items-center justify-center">
              <Bot className="w-4 h-4 text-green-600" />
            </div>
            <div className="bg-gray-100 p-3 rounded-2xl rounded-tl-none">
              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={sendMessage} className="p-4 border-t border-gray-100">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Share how you're feeling..."
            className="flex-1 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-all"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  )
}
