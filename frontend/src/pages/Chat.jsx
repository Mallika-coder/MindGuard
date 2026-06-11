import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2, Sparkles, User, Mic, MicOff } from 'lucide-react'
import { useVoiceInput, speak } from '../hooks/useVoice'
import { classifyText, analyzeEmotions, detectDistortions } from '../utils/wellness'
import axios from 'axios'

const SYSTEM_PROMPT = `You are Dr. Milo, a compassionate AI mental health companion trained in Cognitive Behavioral Therapy (CBT). You work inside MindGuard — an AI-powered mental health screening platform.

YOUR APPROACH:
1. Always validate feelings FIRST before offering advice
2. Identify cognitive distortions (catastrophizing, all-or-nothing, mind-reading, etc.)
3. Guide users to reframe negative thoughts using Socratic questioning
4. Suggest specific, actionable coping strategies
5. Know when to recommend professional help
6. ALWAYS respond in the SAME LANGUAGE the user writes in (Hindi, Spanish, etc.)

YOUR STYLE:
- Warm, empathetic, non-judgmental
- Keep responses 3-5 sentences (concise but meaningful)
- Use CBT techniques naturally (don't label them unless asked)
- Ask one follow-up question to keep conversation going
- Never diagnose or prescribe medication
- If someone is in crisis, gently direct them to 988 Suicide & Crisis Lifeline

IMPORTANT: You respond in whatever language the user speaks. If they write in Hindi, respond in Hindi. If English, respond in English. Match their language naturally.`

const LLM_API = "https://backend.buildpicoapps.com/aero/run/llm-api?pk=v1-Z0FBQUFBQm81Qnd5eHIwYkVfbkEwcGJOX21LYmpSU2tpWnlTYmlNYW9IWGluZE5TWEFKc2sxUTdQWFZrQldMVVVmSDJyQ3pWMEdFbFBxbkVleF9VMTR5ZDgzQVpTbmVrZ3c9PQ=="

const QUICK_PROMPTS = [
  "I'm feeling anxious about exams",
  "I can't stop procrastinating",
  "I feel lonely and disconnected",
  "Help me reframe a negative thought",
  "I'm having a good day!",
  "Guide me through breathing",
  "मुझे बहुत stress हो रहा है",
]

async function getAIResponse(message, analysisContext) {
  const contextualPrompt = `${SYSTEM_PROMPT}

CURRENT ANALYSIS OF USER'S MESSAGE:
- Detected state: ${analysisContext.label} (${Math.round(analysisContext.confidence * 100)}% confidence)
- Top emotions: ${analysisContext.topEmotions}
- Cognitive distortions found: ${analysisContext.distortions || 'None'}
- Risk level: ${analysisContext.riskLevel}

Use this analysis to inform your response, but don't just list these facts. Weave them into a natural, caring conversation.

User says: "${message}"

Respond with empathy and CBT-based guidance:`

  try {
    const res = await axios.post(LLM_API, { prompt: contextualPrompt }, {
      headers: { "Content-Type": "application/json" },
      timeout: 15000,
    })
    return res.data?.text || generateFallbackResponse(analysisContext)
  } catch {
    return generateFallbackResponse(analysisContext)
  }
}

function generateFallbackResponse(ctx) {
  const { label, topEmotions, distortions } = ctx

  const validations = {
    anxiety: [
      "I can sense the tension in your words. Your nervous system is trying to protect you — even when it overreacts.",
      "That worry you're feeling? It's your brain's alarm system. Let's turn down the volume together.",
      "Anxiety can feel so overwhelming, like it'll never end. But I want you to know — this wave will pass.",
    ],
    depression: [
      "I hear the heaviness in what you're sharing. Thank you for trusting me with it.",
      "When everything feels grey, even reaching out takes enormous strength. I see that in you right now.",
      "Depression has a way of making everything feel permanent. But feelings are visitors, not residents.",
    ],
    stress: [
      "You're carrying a lot. The fact that you're here, trying to process it — that matters.",
      "Feeling overwhelmed doesn't mean you're weak. It means you're human with real limits.",
      "Let's slow down for a moment. You don't have to solve everything right now.",
    ],
    severe: [
      "I hear you. What you're feeling is real, and you matter deeply. Please reach out to 988 — a real person is waiting to help right now.",
    ],
    normal: [
      "It's wonderful to hear that energy! What's contributing to this positive state?",
      "You seem grounded today. That's worth celebrating — what's working for you?",
      "I love this energy! Let's explore what's fueling it so you can come back to it on harder days.",
    ],
  }

  const strategies = {
    anxiety: [
      "\n\nLet's try something: Can you name 3 things you can physically see right now? This grounds you in the present.",
      "\n\nHere's a thought experiment: What would you tell your best friend if they told you what you just told me?",
      "\n\nBreathing check: Are your shoulders tight? Drop them. Unclench your jaw. Take one slow breath with me.",
    ],
    depression: [
      "\n\nOne tiny question: What's one thing within arm's reach that might bring even 1% comfort?",
      "\n\nI'm curious — when was the last time you felt differently, even for a moment? What was happening then?",
      "\n\nYou don't have to be productive today. Just being here is enough. What does rest look like for you?",
    ],
    stress: [
      "\n\nLet's sort this: Of everything on your plate, what's the ONE thing you can control right now?",
      "\n\nPermission granted to do less. What's one thing you can say 'no' to or delay?",
      "\n\nStress often shrinks when we name it. Can you finish this sentence: 'I'm stressed because...'?",
    ],
    normal: [
      "\n\nLet's anchor this: What made today feel good? Naming it builds resilience for harder days.",
      "\n\nGratitude moment: What's one thing you'd miss if it disappeared tomorrow?",
    ],
    severe: ["\n\n🆘 Please call 988 or text HOME to 741741. You deserve immediate support."],
  }

  const validation = validations[label]?.[Math.floor(Math.random() * validations[label].length)] || "Thank you for sharing that with me."
  const strategy = strategies[label]?.[Math.floor(Math.random() * (strategies[label]?.length || 1))] || ""

  let distortionNote = ""
  if (distortions && distortions !== 'None') {
    distortionNote = `\n\n🔍 I noticed a thinking pattern here: ${distortions}. That's something we can explore if you'd like.`
  }

  return `${validation}${strategy}${distortionNote}\n\nWhat comes up for you when you hear that?`
}

export default function Chat() {
  const [messages, setMessages] = useState([
    { role: 'ai', text: "Hey! 🌱 I'm Dr. Milo — your AI companion trained in CBT (Cognitive Behavioral Therapy).\n\nI analyze how you're feeling in real-time and respond with evidence-based support. I speak every language — talk to me in Hindi, English, Spanish, whatever feels natural.\n\n**What's on your mind today?**" }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const { isListening, startListening, stopListening } = useVoiceInput()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setInput('')
    setLoading(true)

    const classification = classifyText(msg)
    const emotions = analyzeEmotions(msg)
    const distortions = detectDistortions(msg)
    const topEmotions = Object.entries(emotions).sort((a,b) => b[1]-a[1]).slice(0,3).map(e => e[0]).join(', ')
    const distortionText = distortions.length > 0 ? distortions.map(d => d.label).join(', ') : 'None'
    const riskLevel = classification.severity_score >= 0.75 ? 'High' : classification.severity_score >= 0.5 ? 'Moderate' : 'Low'

    const analysisContext = {
      label: classification.label,
      confidence: classification.confidence,
      topEmotions,
      distortions: distortionText,
      riskLevel,
    }

    try {
      const response = await getAIResponse(msg, analysisContext)
      const badge = `\n\n---\n🧠 \`${classification.label.toUpperCase()}\` · ${(classification.confidence*100).toFixed(0)}% · ${topEmotions}`
      setMessages(prev => [...prev, { role: 'ai', text: response + badge }])

      const cleanResponse = response.split('---')[0].replace(/[*_#🧠🔍💭🆘✨🌱]/g, '').trim()
      speak(cleanResponse)
    } catch {
      const fallback = generateFallbackResponse(analysisContext)
      setMessages(prev => [...prev, { role: 'ai', text: fallback }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white border border-border rounded-card overflow-hidden max-w-3xl mx-auto">
      {/* Header */}
      <div className="px-5 py-3.5 bg-gradient-to-r from-brand-400 to-brand-300 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/milo-character.avif" alt="Dr. Milo" className="w-9 h-9 rounded-full object-cover border-2 border-white/30"/>
          <div>
            <div className="text-sm font-semibold">Dr. Milo</div>
            <div className="text-[10px] text-white/70">CBT-trained · Multilingual · Voice-enabled</div>
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
          <div className="text-[10px] text-gray-400 mb-1.5">Try asking (any language works):</div>
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
            <span className="text-[10px] text-red-500 font-medium">Listening... speak in any language</span>
          </div>
        )}
        <div className="flex gap-2">
          <button
            onClick={isListening ? stopListening : () => startListening((t) => setInput(t))}
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all flex-shrink-0 ${
              isListening ? 'bg-red-500 text-white shadow-lg shadow-red-500/30' : 'bg-gray-100 text-gray-400 hover:bg-brand-50 hover:text-brand-400'
            }`}>
            {isListening ? <MicOff className="w-4 h-4"/> : <Mic className="w-4 h-4"/>}
          </button>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder={isListening ? "Listening... 🎙️" : "Type or speak in any language..."}
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
