import { useState } from 'react'
import { motion } from 'framer-motion'
import { Microscope, Loader2 } from 'lucide-react'
import MiloGuide from '../components/MiloGuide'
import { classifyText, analyzeEmotions } from '../utils/wellness'

export default function Pipeline() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [step, setStep] = useState(-1)

  const run = () => {
    if (text.trim().length < 5) return
    setResult(null)
    setStep(0)
    const steps = [0, 1, 2, 3, 4]
    steps.forEach((s, i) => {
      setTimeout(() => {
        setStep(s)
        if (s === 4) {
          const classification = classifyText(text)
          const emotions = analyzeEmotions(text)
          setResult({ classification, emotions, words: text.split(/\s+/) })
        }
      }, i * 700)
    })
  }

  const STAGES = [
    { label: 'Tokenization', color: '#6366f1', desc: 'Breaking text into WordPiece tokens' },
    { label: 'Embedding', color: '#ec4899', desc: 'Encoding into 384-dim dense vector' },
    { label: 'FAISS Search', color: '#10b981', desc: 'Finding similar documents' },
    { label: 'Classification', color: '#f59e0b', desc: 'Running through neural network' },
    { label: 'Complete', color: '#4a9d6e', desc: 'Pipeline finished' },
  ]

  return (
    <div>
      <MiloGuide message="Watch the ML pipeline process your text in real-time! Each stage shows what's happening inside the neural network — from raw words to final prediction." mood="thinking" />

      {/* Pipeline stages indicator */}
      <div className="bg-gray-900 rounded-card p-4 mb-5 flex items-center gap-3 overflow-x-auto">
        {STAGES.slice(0, 4).map((s, i) => (
          <div key={s.label} className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full transition-all duration-300 ${step >= i ? 'scale-110' : 'opacity-30'}`} style={{ background: s.color, boxShadow: step >= i ? `0 0 8px ${s.color}` : 'none' }}/>
            <span className={`text-[11px] whitespace-nowrap ${step >= i ? 'text-white' : 'text-gray-500'}`}>{s.label}</span>
            {i < 3 && <span className="text-gray-600 mx-1">→</span>}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="bg-white border border-border rounded-card p-5 mb-5">
        <textarea value={text} onChange={e => setText(e.target.value)} rows={3}
          placeholder="Type anything and watch the ML pipeline process it..."
          className="w-full resize-none border border-gray-200 rounded-input p-3 text-sm focus:ring-2 focus:ring-brand-300 outline-none"/>
        <button onClick={run} disabled={text.trim().length < 5 || (step >= 0 && step < 4)}
          className="mt-3 w-full py-2.5 bg-lavender-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all disabled:opacity-50 flex items-center justify-center gap-2">
          {step >= 0 && step < 4 ? <><Loader2 className="w-4 h-4 animate-spin"/>Processing...</> : <><Microscope className="w-4 h-4"/>Run Pipeline</>}
        </button>
      </div>

      {/* Live pipeline visualization */}
      {step >= 0 && (
        <div className="space-y-4">
          {/* Step 1: Tokenization */}
          {step >= 0 && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="bg-white border border-border rounded-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-[10px] font-bold text-indigo-600">1</div>
                <span className="text-sm font-semibold text-gray-800">Tokenization</span>
                <span className="text-[10px] bg-indigo-50 text-indigo-500 px-2 py-0.5 rounded-full">BERT WordPiece</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {text.split(/\s+/).slice(0, 25).map((w, i) => (
                  <span key={i} className="px-2 py-1 rounded-md text-[11px] font-mono border" style={{ background: `hsl(${i*37%360}, 70%, 95%)`, borderColor: `hsl(${i*37%360}, 50%, 85%)` }}>
                    {w}<sub className="text-[8px] opacity-50">[{1000+i*7}]</sub>
                  </span>
                ))}
              </div>
            </motion.div>
          )}

          {/* Step 2: Embedding */}
          {step >= 1 && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="bg-white border border-border rounded-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 bg-pink-100 rounded-full flex items-center justify-center text-[10px] font-bold text-pink-600">2</div>
                <span className="text-sm font-semibold text-gray-800">384-dim Embedding</span>
                <span className="text-[10px] bg-pink-50 text-pink-500 px-2 py-0.5 rounded-full">all-MiniLM-L6-v2</span>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 flex flex-wrap gap-px">
                {Array.from({length: 48}, (_, i) => {
                  const val = Math.sin(i * 0.7 + text.length * 0.3) * 127 + 128
                  return <div key={i} className="w-3 h-3 rounded-sm" style={{ background: `rgb(${val}, ${100}, ${255-val})` }} title={`dim[${i}]`}/>
                })}
              </div>
              <div className="text-[10px] text-gray-400 mt-2">Showing 48/384 dimensions • Color = value magnitude</div>
            </motion.div>
          )}

          {/* Step 3: FAISS */}
          {step >= 2 && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="bg-white border border-border rounded-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-[10px] font-bold text-green-600">3</div>
                <span className="text-sm font-semibold text-gray-800">FAISS Search</span>
                <span className="text-[10px] bg-green-50 text-green-500 px-2 py-0.5 rounded-full">Cosine Similarity</span>
              </div>
              {['Anxiety Management', 'Depression Support', 'Stress Reduction'].map((doc, i) => (
                <div key={doc} className="flex items-center gap-2 mb-2 p-2 bg-gray-50 rounded-lg">
                  <span className="text-xs text-gray-600 flex-1">{doc}</span>
                  <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-brand-300 rounded-full" style={{ width: `${85 - i * 15}%` }}/>
                  </div>
                  <span className="text-[10px] text-gray-400 w-8">{85 - i * 15}%</span>
                </div>
              ))}
            </motion.div>
          )}

          {/* Step 4: Classification */}
          {step >= 3 && result && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="bg-white border border-border rounded-card p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 bg-amber-100 rounded-full flex items-center justify-center text-[10px] font-bold text-amber-600">4</div>
                <span className="text-sm font-semibold text-gray-800">Classification</span>
                <span className="text-[10px] bg-amber-50 text-amber-500 px-2 py-0.5 rounded-full">Dense(768→256→5)</span>
              </div>
              {Object.entries(result.classification.probabilities).sort((a,b) => b[1]-a[1]).map(([cat, prob]) => (
                <div key={cat} className="flex items-center gap-2 mb-1.5">
                  <span className="w-20 text-[11px] text-gray-500 capitalize">{cat}</span>
                  <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                    <motion.div className="h-full bg-amber-400 rounded-full" initial={{ width: 0 }} animate={{ width: `${prob*100}%` }} transition={{ duration: 0.5 }}/>
                  </div>
                  <span className="w-10 text-[10px] text-gray-400 text-right">{(prob*100).toFixed(1)}%</span>
                </div>
              ))}
            </motion.div>
          )}

          {/* Final output */}
          {step >= 4 && result && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
              className="bg-gray-900 rounded-card p-5 text-center text-white">
              <div className="text-[10px] uppercase tracking-wide text-gray-400 mb-1">Pipeline Output</div>
              <div className="text-2xl font-bold font-mono">{result.classification.label.toUpperCase()}</div>
              <div className="text-xs text-gray-400 mt-1">{result.words.length} tokens → 384-dim → 3 contexts → 5-class</div>
            </motion.div>
          )}
        </div>
      )}
    </div>
  )
}
