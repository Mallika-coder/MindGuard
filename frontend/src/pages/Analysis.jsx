import { useState } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2, RotateCcw } from 'lucide-react'
import MiloGuide from '../components/MiloGuide'
import { VoiceInputButton } from '../components/VoiceButton'
import { useVoiceInput, speak } from '../hooks/useVoice'
import { classifyText, analyzeEmotions, detectDistortions, computeLinguistics, computeRisk } from '../utils/wellness'

function RiskGauge({ score, level }) {
  const circumference = 2 * Math.PI * 45
  const offset = circumference - score * circumference
  const color = score >= 0.6 ? '#ef4444' : score >= 0.4 ? '#f59e0b' : score >= 0.2 ? '#4a9d6e' : '#06b6d4'
  return (
    <div className="text-center">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="none" stroke="#e8ecf0" strokeWidth="8"/>
        <motion.circle cx="50" cy="50" r="45" fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={circumference} initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }} transition={{ duration: 0.8, ease: 'easeOut' }}
          strokeLinecap="round" transform="rotate(-90 50 50)"/>
        <text x="50" y="46" textAnchor="middle" fill="#1a2332" fontSize="18" fontWeight="bold" fontFamily="JetBrains Mono">{(score*100).toFixed(0)}%</text>
        <text x="50" y="62" textAnchor="middle" fill="#5a6b7b" fontSize="9">{level}</text>
      </svg>
    </div>
  )
}

function EmotionBar({ emotion, score, color }) {
  return (
    <div className="flex items-center gap-2 mb-1.5">
      <span className="w-20 text-[11px] text-gray-500 capitalize">{emotion}</span>
      <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
        <motion.div className="h-full rounded-full" style={{ background: color }}
          initial={{ width: 0 }} animate={{ width: `${score * 100}%` }} transition={{ duration: 0.6, delay: 0.1 }}/>
      </div>
      <span className="w-8 text-[10px] text-gray-400 text-right">{(score*100).toFixed(0)}%</span>
    </div>
  )
}

export default function Analysis() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const { isListening, startListening, stopListening } = useVoiceInput()

  const analyze = () => {
    if (text.trim().length < 10) return
    setLoading(true)
    setTimeout(() => {
      const classification = classifyText(text)
      const emotions = analyzeEmotions(text)
      const distortions = detectDistortions(text)
      const linguistics = computeLinguistics(text)
      const risk = computeRisk(classification, emotions, distortions, linguistics)
      setResult({ classification, emotions, distortions, linguistics, risk })
      setLoading(false)
      speak(`Analysis complete. I detected ${classification.label} with ${Math.round(classification.confidence * 100)} percent confidence. Your risk level is ${risk.level}.${distortions.length > 0 ? ` I also noticed ${distortions.length} cognitive distortion${distortions.length > 1 ? 's' : ''}.` : ''}`)
    }, 800)
  }

  const emotionColors = { anger: '#ef4444', sadness: '#3b82f6', fear: '#8b5cf6', joy: '#10b981', disgust: '#84cc16', surprise: '#f59e0b', trust: '#06b6d4', anticipation: '#ec4899' }

  return (
    <div>
      <MiloGuide message="Share how you're feeling and I'll analyze it across 6 dimensions — classification, emotions, cognition, linguistics, risk, and personalized guidance." mood="happy" />

      {/* Dimension badges */}
      <div className="flex gap-1.5 flex-wrap mb-5">
        {['Classification', 'Emotions', 'Cognition', 'Linguistics', 'Risk', 'RAG Response'].map((d, i) => (
          <span key={d} className="px-2.5 py-1 bg-brand-50 text-brand-400 rounded-full text-[10px] font-semibold">{i+1}. {d}</span>
        ))}
      </div>

      {/* Input */}
      <div className="bg-white border border-border rounded-card p-5 mb-6">
        <div className="relative">
          <textarea value={text} onChange={e => setText(e.target.value)} rows={5}
            placeholder="Write or speak about how you've been feeling lately..."
            className="w-full resize-none border border-gray-200 rounded-input p-4 pr-14 text-sm focus:ring-2 focus:ring-brand-300 focus:border-transparent outline-none"/>
          {/* Voice input button inside textarea */}
          <div className="absolute right-3 top-3">
            <VoiceInputButton
              isListening={isListening}
              onStart={() => startListening((t) => setText(t))}
              onStop={stopListening}
            />
          </div>
        </div>
        {isListening && (
          <div className="flex items-center gap-2 mt-2 px-2">
            <div className="flex gap-0.5">
              {[...Array(5)].map((_, i) => (
                <motion.div key={i} className="w-1 bg-red-400 rounded-full"
                  animate={{ height: [8, 16, 8] }}
                  transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.1 }}/>
              ))}
            </div>
            <span className="text-xs text-red-500 font-medium">Listening... speak now</span>
          </div>
        )}
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-gray-400">{text.length} characters {isListening && '• 🎙️ Voice active'}</span>
          <div className="flex gap-2">
            <button onClick={() => { setText(''); setResult(null) }} className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700 transition-colors">
              <RotateCcw className="w-4 h-4 inline mr-1"/>Clear
            </button>
            <button onClick={analyze} disabled={loading || text.trim().length < 10}
              className="px-6 py-2.5 bg-brand-300 text-white rounded-pill font-medium text-sm shadow-md shadow-brand-300/20 hover:shadow-lg hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
              {loading ? <><Loader2 className="w-4 h-4 animate-spin"/>Analyzing...</> : <><Send className="w-4 h-4"/>Analyze</>}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
          {/* Top metrics */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white border border-border rounded-card p-5 text-center">
              <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Classification</div>
              <div className="text-2xl font-bold font-mono text-gray-900 capitalize">{result.classification.label}</div>
              <div className="text-xs text-gray-400 mt-1">{(result.classification.confidence*100).toFixed(0)}% confidence</div>
            </div>
            <div className="bg-white border border-border rounded-card p-5 flex justify-center">
              <RiskGauge score={result.risk.composite} level={result.risk.level}/>
            </div>
            <div className="bg-white border border-border rounded-card p-5 text-center">
              <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Wellness</div>
              <div className="text-2xl font-bold font-mono text-brand-300">{Math.max(0, Math.round((1 - result.risk.composite) * 100))}</div>
              <div className="text-xs text-gray-400 mt-1">/ 100</div>
            </div>
          </div>

          {/* Emotions + Classification grid */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white border border-border rounded-card p-5">
              <h3 className="text-sm font-semibold text-gray-800 mb-3">Emotional Profile</h3>
              {Object.entries(result.emotions).sort((a,b) => b[1]-a[1]).map(([emo, score]) => (
                <EmotionBar key={emo} emotion={emo} score={score} color={emotionColors[emo] || '#6366f1'}/>
              ))}
            </div>
            <div className="bg-white border border-border rounded-card p-5">
              <h3 className="text-sm font-semibold text-gray-800 mb-3">Probability Distribution</h3>
              {Object.entries(result.classification.probabilities).sort((a,b) => b[1]-a[1]).map(([cat, prob]) => (
                <EmotionBar key={cat} emotion={cat} score={prob} color={cat === 'normal' ? '#10b981' : cat === 'stress' ? '#f59e0b' : cat === 'anxiety' ? '#f97316' : cat === 'depression' ? '#8b5cf6' : '#ef4444'}/>
              ))}
            </div>
          </div>

          {/* Distortions */}
          <div className="bg-white border border-border rounded-card p-5">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">Cognitive Distortions</h3>
            {result.distortions.length > 0 ? (
              <div className="grid grid-cols-2 gap-3">
                {result.distortions.map(d => (
                  <div key={d.id} className="bg-amber-50 border border-amber-200 rounded-xl p-3">
                    <div className="text-xs font-semibold text-amber-800 mb-1">⚠️ {d.label}</div>
                    <div className="text-[11px] text-amber-700">{d.description}</div>
                    <div className="text-[10px] text-amber-600 mt-1.5 italic">💡 {d.reframe}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-brand-300">
                <div className="text-2xl mb-1">✅</div>
                <div className="text-sm font-medium">No distortions detected</div>
              </div>
            )}
          </div>

          {/* Linguistics */}
          <div className="bg-white border border-border rounded-card p-5">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">Linguistic Biomarkers</h3>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Words', value: result.linguistics.wordCount, icon: '📝' },
                { label: 'Self-focus', value: `${(result.linguistics.firstPersonRatio*100).toFixed(0)}%`, icon: '👤' },
                { label: 'Negativity', value: `${(result.linguistics.negationRatio*100).toFixed(0)}%`, icon: '➖' },
                { label: 'Diversity', value: `${(result.linguistics.lexicalDiversity*100).toFixed(0)}%`, icon: '🔤' },
                { label: 'Sentences', value: result.linguistics.sentenceCount, icon: '📄' },
                { label: 'Avg Length', value: `${result.linguistics.avgSentenceLength}w`, icon: '📏' },
              ].map(m => (
                <div key={m.label} className="bg-gray-50 rounded-xl p-3 text-center">
                  <div className="text-lg">{m.icon}</div>
                  <div className="text-base font-bold font-mono text-gray-800 mt-1">{m.value}</div>
                  <div className="text-[10px] text-gray-400">{m.label}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
