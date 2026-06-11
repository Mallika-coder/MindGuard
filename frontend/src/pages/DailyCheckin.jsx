import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { useLocalState } from '../hooks/useLocalState'

export default function DailyCheckin() {
  const [mood, setMood] = useState(3)
  const [sleep, setSleep] = useState(3)
  const [energy, setEnergy] = useState(3)
  const [social, setSocial] = useState(3)
  const [note, setNote] = useState('')
  const [history, setHistory] = useLocalState('mg-checkins', [])
  const [submitted, setSubmitted] = useState(false)

  const submit = () => {
    setHistory([...history, { mood, sleep, energy, social, note, date: new Date().toISOString() }])
    setSubmitted(true)
  }

  const wellness = Math.round((mood + sleep + energy + social) / 20 * 100)

  const Slider = ({ label, icon, value, onChange }) => (
    <div className="bg-white border border-border rounded-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">{icon} {label}</span>
        <span className="text-lg font-bold font-mono text-brand-300">{value}/5</span>
      </div>
      <input type="range" min={1} max={5} step={1} value={value} onChange={e => onChange(+e.target.value)}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-300"/>
      <div className="flex justify-between text-[9px] text-gray-400 mt-1"><span>Low</span><span>Great</span></div>
    </div>
  )

  return (
    <div className="max-w-xl mx-auto">
      <MiloGuide message="Quick daily check-in! Rate 4 dimensions of your wellbeing. Consistency builds powerful self-awareness — even 30 seconds counts." mood="happy" />

      {!submitted ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Slider label="Mood" icon="😊" value={mood} onChange={setMood}/>
            <Slider label="Sleep" icon="🌙" value={sleep} onChange={setSleep}/>
            <Slider label="Energy" icon="⚡" value={energy} onChange={setEnergy}/>
            <Slider label="Social" icon="👥" value={social} onChange={setSocial}/>
          </div>
          <div className="bg-white border border-border rounded-card p-4">
            <input value={note} onChange={e => setNote(e.target.value)} placeholder="Quick note (optional)..."
              className="w-full text-sm border-0 outline-none placeholder:text-gray-300"/>
          </div>
          <button onClick={submit} className="w-full py-3 bg-brand-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all">
            ✅ Submit Check-in
          </button>
        </div>
      ) : (
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="text-center">
          <div className="bg-gradient-to-b from-gray-900 to-gray-800 rounded-card p-8 mb-5">
            <div className="text-5xl font-bold font-mono text-brand-300">{wellness}%</div>
            <div className="text-sm text-gray-400 mt-2">Today's Wellness Score</div>
          </div>
          <div className="grid grid-cols-4 gap-3">
            {[{l:'Mood',v:mood,i:'😊'},{l:'Sleep',v:sleep,i:'🌙'},{l:'Energy',v:energy,i:'⚡'},{l:'Social',v:social,i:'👥'}].map(d => (
              <div key={d.l} className="bg-white border border-border rounded-card p-3 text-center">
                <div className="text-xl">{d.i}</div>
                <div className="text-lg font-bold text-brand-400 mt-1">{d.v}/5</div>
                <div className="text-[10px] text-gray-400">{d.l}</div>
              </div>
            ))}
          </div>
          <button onClick={() => setSubmitted(false)} className="mt-5 text-sm text-brand-300 hover:underline">Submit another</button>
        </motion.div>
      )}
    </div>
  )
}
