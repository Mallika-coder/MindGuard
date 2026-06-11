import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2, Sparkles, User, Mic, MicOff, Volume2 } from 'lucide-react'
import { useVoiceInput, speak } from '../hooks/useVoice'
import { classifyText, analyzeEmotions, detectDistortions } from '../utils/wellness'

const CBT_RESPONSES = {
  anxiety: [
    "I can sense the anxiety pulling at you. Your nervous system is working overtime — that's exhausting, and it makes sense given what you're dealing with.\n\n💭 Let's try something: breathe in for 4, hold for 7, out for 8. Repeat 3 times.\n\nWhat evidence do you have for your worry? And what evidence goes against it?",
    "Your mind is trying to protect you — even when it overreacts. That worry makes complete sense.\n\n💭 Ask yourself: What's the worst that could happen? The best? The most realistic?\n\nGrounding can help right now: name 5 things you can see around you.",
  ],
  depression: [
    "I can feel the heaviness in what you're sharing. That takes real courage to express.\n\n💭 What's one tiny thing within reach right now that might bring even 1% comfort? A warm drink, natural light, a song you used to love?\n\nRemember: depression lies to you. It says 'always' and 'never' — but that's the illness talking, not reality.",
    "The sadness you're describing is real and valid — I'm here with you in it.\n\n💭 If a close friend told you what you just told me, what would you say to them? Try saying that to yourself.\n\nYou don't have to fix everything today. One moment at a time.",
  ],
  stress: [
    "It sounds like you're carrying a heavy load. Stress is your body's signal that something needs attention.\n\n💭 Quick reset: What are the 3 most pressing things? Circle the ONE you can actually control right now. Let the others wait.\n\nRemember: it's okay to set boundaries and ask for help.",
    "That overwhelm is completely valid. You're not failing — you're human with finite capacity.\n\n💭 Try this: What would your day look like if you acted 'as if' things were manageable, just for one hour?\n\nBreak the mountain into pebbles. One task. Then the next.",
  ],
  severe: [
    "I hear you, and I want you to know that what you're feeling matters deeply. You're not alone in this.\n\n🆘 Please reach out to the 988 Suicide & Crisis Lifeline (call or text 988) or text HOME to 741741. A trained person is available 24/7.\n\nYour life has value. This moment of pain is real, but it is not permanent.",
  ],
  normal: [
    "It's great to hear that positive energy! Let's build on this momentum.\n\n✨ What's fueling this good feeling? Naming it helps you recreate it on harder days.\n\nConsider noting what went well today — gratitude journaling builds resilience for tougher times.",
  ],
}

const QUICK_PROMPTS = [
  "I'm feeling anxious about exams",
  "I can't stop procrastinating",
  "I feel lonely and disconnected",
  "Help me reframe a negative thought",
  "I'm having a good day!",
  "Guide me through breathing",
]

export default function Chat() {
  const [messages, setMessages] = useState([
    { role: 'ai', text: "Hey there! 🌱 I'm your MindGuard AI companion — trained in CBT techniques.\n\nI analyze *how* you're feeling (8 emotions), detect thinking patterns, and provide evidence-based strategies.\n\nWhat's on your mind?" }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setInput('')
    setLoading(true)

    setTimeout(() => {
      const classification = classifyText(msg)
      const emotions = analyzeEmotions(msg)
      const distortions = detectDistortions(msg)
      const label = classification.label
      const responses = CBT_RESPONSES[label] || CBT_RESPONSES.normal
      let response = responses[Math.floor(Math.random() * responses.length)]

      if (distortions.length > 0) {
        const d = distortions[0]
        response += `\n\n🔍 I noticed *${d.label}* in your words — ${d.description.toLowerCase()}. ${d.reframe}`
      }

      const topEmo = Object.entries(emotions).sort((a,b) => b[1]-a[1]).slice(0,2).map(e => e[0]).join(', ')
      response += `\n\n---\n🧠 ${label.toUpperCase()} (${(classification.confidence*100).toFixed(0)}%) · Emotions: ${topEmo}`

      setMessages(prev => [...prev, { role: 'ai', text: response }])
      setLoading(false)
      // Speak the response (clean text without markdown)
      const cleanResponse = response.split('---')[0].replace(/[*_#🧠🔍💭🆘✨]/g, '').trim()
      speak(cleanResponse)
    }, 1200)
  }

  const { isListening, startListening, stopListening } = useVoiceInput()

  return (
    <div className="bg-white border border-border rounded-card overflow-hidden max-w-3xl mx-auto">
      {/* Header */}
      <div className="px-5 py-3.5 bg-gradient-to-r from-brand-400 to-brand-300 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
            <Sparkles className="w-4 h-4"/>
          </div>
          <div>
            <div className="text-sm font-semibold">MindGuard AI</div>
            <div className="text-[10px] text-white/70">CBT-trained companion</div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse"/>
          <span className="text-[10px] text-white/70">Online</span>
        </div>
      </div>

      {/* Messages */}
      <div className="h-[440px] overflow-y-auto p-4 space-y-3 bg-gray-50/50">
        {messages.map((m, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex items-end gap-2 max-w-[80%] ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${m.role === 'user' ? 'bg-brand-50' : 'bg-lavender-50'}`}>
                {m.role === 'user' ? <User className="w-3.5 h-3.5 text-brand-300"/> : <Sparkles className="w-3.5 h-3.5 text-lavender-300"/>}
              </div>
              <div className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                m.role === 'user' ? 'bg-brand-300 text-white rounded-br-sm' : 'bg-white border border-border text-gray-700 rounded-bl-sm shadow-sm'
              }`}>{m.text}</div>
            </div>
          </motion.div>
        ))}
        {loading && (
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-lavender-50 flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5 text-lavender-300"/>
            </div>
            <div className="px-4 py-3 bg-white border border-border rounded-2xl rounded-bl-sm">
              <div className="flex gap-1">
                {[0, 150, 300].map(d => <div key={d} className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: `${d}ms` }}/>)}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      {/* Quick prompts */}
      {messages.length <= 2 && (
        <div className="px-4 py-2 border-t border-border bg-white">
          <div className="text-[10px] text-gray-400 mb-1.5">Try asking:</div>
          <div className="flex flex-wrap gap-1.5">
            {QUICK_PROMPTS.map(p => (
              <button key={p} onClick={() => sendMessage(p)}
                className="px-3 py-1.5 bg-gray-50 border border-border rounded-full text-[11px] font-medium text-gray-600 hover:bg-brand-50 hover:text-brand-400 hover:border-brand-200 transition-all">{p}</button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-3 border-t border-border bg-white">
        {isListening && (
          <div className="flex items-center gap-2 mb-2 px-2">
            <div className="flex gap-0.5">
              {[...Array(4)].map((_, i) => (
                <motion.div key={i} className="w-1 bg-red-400 rounded-full"
                  animate={{ height: [6, 14, 6] }}
                  transition={{ duration: 0.4, repeat: Infinity, delay: i * 0.1 }}/>
              ))}
            </div>
            <span className="text-[10px] text-red-500 font-medium">Listening...</span>
          </div>
        )}
        <div className="flex gap-2">
          {/* Mic button */}
          <button
            onClick={isListening ? stopListening : () => startListening((t) => setInput(t))}
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all flex-shrink-0 ${
              isListening ? 'bg-red-500 text-white shadow-lg shadow-red-500/30' : 'bg-gray-100 text-gray-400 hover:bg-brand-50 hover:text-brand-400'
            }`}>
            {isListening ? <MicOff className="w-4 h-4"/> : <Mic className="w-4 h-4"/>}
          </button>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder={isListening ? "Listening... speak now" : "Type or speak your thoughts..."}
            className="flex-1 px-4 py-2.5 bg-gray-50 border border-border rounded-full text-sm focus:ring-2 focus:ring-brand-300 focus:border-transparent outline-none"/>
          <button onClick={() => sendMessage()} disabled={!input.trim() || loading}
            className="w-10 h-10 bg-brand-300 rounded-full flex items-center justify-center text-white hover:scale-105 transition-all shadow-md shadow-brand-300/20 disabled:opacity-50">
            <Send className="w-4 h-4"/>
          </button>
        </div>
      </div>
    </div>
  )
}
