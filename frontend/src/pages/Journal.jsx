import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { useLocalState } from '../hooks/useLocalState'
import { classifyText, analyzeEmotions } from '../utils/wellness'

export default function Journal() {
  const [entries, setEntries] = useLocalState('mg-journal', [])
  const [text, setText] = useState('')
  const [mood, setMood] = useState('😐 Okay')

  const save = () => {
    if (text.trim().length < 5) return
    const classification = classifyText(text)
    const emotions = analyzeEmotions(text)
    const topEmo = Object.entries(emotions).sort((a,b) => b[1]-a[1])[0][0]
    setEntries([{ text, mood, classification: classification.label, topEmo, date: new Date().toISOString() }, ...entries])
    setText('')
  }

  const labelColors = { normal: '#4a9d6e', stress: '#f59e0b', anxiety: '#f97316', depression: '#8b5cf6', severe: '#ef4444' }

  return (
    <div>
      <MiloGuide message="Your AI-powered mood journal. Each entry is analyzed for emotional state and patterns. Write consistently — I'll detect trends over time." mood="caring" />

      <div className="grid grid-cols-5 gap-4">
        {/* Input */}
        <div className="col-span-2">
          <div className="bg-white border border-border rounded-card p-5 sticky top-4">
            <textarea value={text} onChange={e => setText(e.target.value)} rows={5}
              placeholder="How are you feeling right now? What happened today?"
              className="w-full resize-none border border-gray-200 rounded-input p-3 text-sm focus:ring-2 focus:ring-brand-300 outline-none mb-3"/>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {['😊 Good', '😐 Okay', '😔 Low', '😰 Anxious', '😤 Stressed'].map(m => (
                <button key={m} onClick={() => setMood(m)}
                  className={`px-3 py-1.5 rounded-full text-[11px] font-medium transition-all ${mood === m ? 'bg-brand-300 text-white' : 'bg-gray-50 border border-border text-gray-500 hover:border-brand-200'}`}>{m}</button>
              ))}
            </div>
            <button onClick={save} disabled={text.trim().length < 5}
              className="w-full py-2.5 bg-brand-300 text-white rounded-pill text-sm font-medium shadow-md hover:scale-[1.01] transition-all disabled:opacity-50">
              📓 Save & Analyze
            </button>
          </div>
        </div>

        {/* Entries */}
        <div className="col-span-3 space-y-3">
          {entries.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <div className="text-3xl mb-2">📓</div>
              <div className="text-sm">Your journal is empty. Write your first entry!</div>
            </div>
          ) : entries.slice(0, 10).map((e, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className="bg-white border-l-4 rounded-r-card p-4 shadow-sm" style={{ borderLeftColor: labelColors[e.classification] || '#6366f1' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full" style={{ background: `${labelColors[e.classification]}15`, color: labelColors[e.classification] }}>{e.classification}</span>
                <span className="text-[10px] text-gray-400">{e.mood}</span>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">{e.text.slice(0, 200)}{e.text.length > 200 ? '...' : ''}</p>
              <div className="text-[10px] text-gray-400 mt-2">Emotion: {e.topEmo} · {new Date(e.date).toLocaleDateString()}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}
