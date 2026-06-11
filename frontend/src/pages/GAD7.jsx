import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { GAD7_QUESTIONS } from '../utils/wellness'

export default function GAD7() {
  const [answers, setAnswers] = useState(Array(7).fill(0))
  const [submitted, setSubmitted] = useState(false)

  const total = answers.reduce((a, b) => a + b, 0)
  const severity = total <= 4 ? { label: 'Minimal', color: '#4a9d6e' } : total <= 9 ? { label: 'Mild', color: '#f59e0b' } : total <= 14 ? { label: 'Moderate', color: '#f97316' } : { label: 'Severe', color: '#ef4444' }

  return (
    <div className="max-w-2xl mx-auto">
      <MiloGuide message="The GAD-7 measures anxiety over the past 2 weeks. Used by healthcare professionals worldwide. Be honest — there's no right or wrong!" mood="caring" />

      <div className="bg-white border border-border rounded-card p-5 mb-5">
        <div className="text-xs text-gray-500 mb-4">Over the <strong>last 2 weeks</strong>, how often have you been bothered by:
          <span className="block mt-1 text-[10px]">0 = Not at all · 1 = Several days · 2 = More than half · 3 = Nearly every day</span>
        </div>

        {GAD7_QUESTIONS.map((q, i) => (
          <div key={i} className="py-3 border-b border-gray-100 last:border-0">
            <div className="text-sm text-gray-700 mb-2">{i+1}. {q}</div>
            <div className="flex gap-2">
              {[0,1,2,3].map(v => (
                <button key={v} onClick={() => { const a = [...answers]; a[i] = v; setAnswers(a) }}
                  className={`w-10 h-10 rounded-xl text-sm font-medium transition-all ${answers[i] === v ? 'bg-lavender-300 text-white shadow-md' : 'bg-gray-50 border border-border text-gray-600 hover:border-lavender-100'}`}>
                  {v}
                </button>
              ))}
            </div>
          </div>
        ))}

        <button onClick={() => setSubmitted(true)}
          className="mt-5 w-full py-3 bg-lavender-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all">
          📊 Calculate Score
        </button>
      </div>

      {submitted && (
        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-border rounded-card p-6 text-center">
          <div className="text-4xl font-bold font-mono" style={{ color: severity.color }}>{total}<span className="text-lg text-gray-400">/21</span></div>
          <div className="text-lg font-semibold mt-2" style={{ color: severity.color }}>{severity.label} Anxiety</div>
          <div className="text-xs text-gray-500 mt-3">GAD-7 © Pfizer Inc. Sensitivity 89%, Specificity 82% at cutoff ≥10 (Spitzer et al., 2006)</div>
        </motion.div>
      )}
    </div>
  )
}
