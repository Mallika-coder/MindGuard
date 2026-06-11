import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { PHQ9_QUESTIONS } from '../utils/wellness'

export default function PHQ9() {
  const [answers, setAnswers] = useState(Array(9).fill(0))
  const [submitted, setSubmitted] = useState(false)

  const total = answers.reduce((a, b) => a + b, 0)
  const severity = total <= 4 ? { label: 'Minimal', color: '#4a9d6e' } : total <= 9 ? { label: 'Mild', color: '#f59e0b' } : total <= 14 ? { label: 'Moderate', color: '#f97316' } : total <= 19 ? { label: 'Moderately Severe', color: '#ef4444' } : { label: 'Severe', color: '#dc2626' }

  return (
    <div className="max-w-2xl mx-auto">
      <MiloGuide message="The PHQ-9 is a clinically validated depression screener used by doctors worldwide. Answer based on the last 2 weeks. I'll interpret your results." mood="caring" />

      <div className="bg-white border border-border rounded-card p-5 mb-5">
        <div className="text-xs text-gray-500 mb-4">Over the <strong>last 2 weeks</strong>, how often have you been bothered by:
          <span className="block mt-1 text-[10px]">0 = Not at all · 1 = Several days · 2 = More than half · 3 = Nearly every day</span>
        </div>

        {PHQ9_QUESTIONS.map((q, i) => (
          <div key={i} className="py-3 border-b border-gray-100 last:border-0">
            <div className="text-sm text-gray-700 mb-2">{i+1}. {q}</div>
            <div className="flex gap-2">
              {[0,1,2,3].map(v => (
                <button key={v} onClick={() => { const a = [...answers]; a[i] = v; setAnswers(a) }}
                  className={`w-10 h-10 rounded-xl text-sm font-medium transition-all ${answers[i] === v ? 'bg-brand-300 text-white shadow-md' : 'bg-gray-50 border border-border text-gray-600 hover:border-brand-200'}`}>
                  {v}
                </button>
              ))}
            </div>
          </div>
        ))}

        <button onClick={() => setSubmitted(true)}
          className="mt-5 w-full py-3 bg-brand-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all">
          📊 Calculate Score
        </button>
      </div>

      {submitted && (
        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-border rounded-card p-6 text-center">
          <div className="text-4xl font-bold font-mono" style={{ color: severity.color }}>{total}<span className="text-lg text-gray-400">/27</span></div>
          <div className="text-lg font-semibold mt-2" style={{ color: severity.color }}>{severity.label} Depression</div>
          <div className="text-xs text-gray-500 mt-3">PHQ-9 © Pfizer Inc. Sensitivity 88%, Specificity 88% at cutoff ≥10 (Kroenke et al., 2001)</div>
          {answers[8] >= 2 && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700 font-medium">
              🚨 Item 9 scored ≥2 — please contact 988 (Suicide & Crisis Lifeline)
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
