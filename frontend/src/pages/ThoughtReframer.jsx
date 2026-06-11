import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { detectDistortions, analyzeEmotions } from '../utils/wellness'

export default function ThoughtReframer() {
  const [thought, setThought] = useState('')
  const [result, setResult] = useState(null)

  const reframe = () => {
    if (thought.trim().length < 5) return
    const distortions = detectDistortions(thought)
    const emotions = analyzeEmotions(thought)
    let reframed = thought
    const replacements = [["I always fail", "I haven't succeeded at this yet"], ["always", "sometimes"], ["never", "not yet"], ["everyone", "some people"], ["nothing", "not everything"], ["I should", "I would like to"]]
    for (const [old, rep] of replacements) {
      if (reframed.toLowerCase().includes(old.toLowerCase())) {
        reframed = reframed.replace(new RegExp(old, 'i'), rep)
        break
      }
    }
    setResult({ distortions, emotions, reframed })
  }

  return (
    <div className="max-w-2xl mx-auto">
      <MiloGuide message="Enter a negative thought and I'll identify the thinking traps (cognitive distortions) and suggest a healthier alternative. Based on CBT — 50+ years of clinical evidence." mood="thinking" />

      <div className="bg-white border border-border rounded-card p-5 mb-5">
        <textarea value={thought} onChange={e => setThought(e.target.value)} rows={3}
          placeholder="e.g., 'I always mess everything up' or 'Nobody cares about me'"
          className="w-full resize-none border border-gray-200 rounded-input p-4 text-sm focus:ring-2 focus:ring-lavender-200 outline-none"/>
        <button onClick={reframe} disabled={thought.trim().length < 5}
          className="mt-3 w-full py-3 bg-lavender-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all disabled:opacity-50">
          🔄 Analyze & Reframe
        </button>
      </div>

      {result && (
        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          {/* Original */}
          <div className="bg-amber-50 border border-amber-200 rounded-card p-4">
            <div className="text-[10px] text-amber-700 uppercase font-semibold tracking-wide mb-1">Original Thought</div>
            <p className="text-sm text-amber-900 italic">"{thought}"</p>
          </div>

          {/* Distortions */}
          {result.distortions.length > 0 ? (
            result.distortions.map(d => (
              <div key={d.id} className="bg-white border border-amber-200 rounded-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-5 h-5 bg-amber-400 text-white rounded-full flex items-center justify-center text-[10px] font-bold">!</span>
                  <span className="text-sm font-semibold text-amber-800">{d.label}</span>
                  <span className="text-[10px] bg-amber-100 text-amber-600 px-2 py-0.5 rounded-full">"{d.trigger}"</span>
                </div>
                <p className="text-xs text-gray-600 mb-2">{d.description}</p>
                <div className="bg-green-50 border border-green-200 rounded-xl p-3">
                  <p className="text-xs text-green-700">💡 <strong>Reframe:</strong> {d.reframe}</p>
                </div>
              </div>
            ))
          ) : (
            <div className="bg-green-50 border border-green-200 rounded-card p-5 text-center">
              <div className="text-2xl mb-1">✅</div>
              <div className="text-sm font-medium text-green-700">Balanced thought — no distortions detected</div>
            </div>
          )}

          {/* Reframed */}
          <div className="bg-green-50 border border-green-200 rounded-card p-4">
            <div className="text-[10px] text-green-700 uppercase font-semibold tracking-wide mb-1">🔄 Reframed Version</div>
            <p className="text-sm text-green-800 font-medium">"{result.reframed}"</p>
          </div>

          <div className="text-[10px] text-gray-400 text-center">Based on Cognitive Behavioral Therapy (Beck, 1976; Burns, 1980)</div>
        </motion.div>
      )}
    </div>
  )
}
