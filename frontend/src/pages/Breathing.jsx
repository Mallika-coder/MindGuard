import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'

const TECHNIQUES = [
  { name: '4-7-8 Breathing', phases: [{ label: 'Inhale', duration: 4 }, { label: 'Hold', duration: 7 }, { label: 'Exhale', duration: 8 }] },
  { name: 'Box Breathing', phases: [{ label: 'Inhale', duration: 4 }, { label: 'Hold', duration: 4 }, { label: 'Exhale', duration: 4 }, { label: 'Hold', duration: 4 }] },
  { name: 'Calming Breath', phases: [{ label: 'Inhale', duration: 4 }, { label: 'Exhale', duration: 6 }] },
]

export default function Breathing() {
  const [technique, setTechnique] = useState(0)
  const [active, setActive] = useState(false)
  const [phase, setPhase] = useState(0)
  const [timer, setTimer] = useState(0)
  const [cycles, setCycles] = useState(0)

  const currentTech = TECHNIQUES[technique]
  const currentPhase = currentTech.phases[phase]

  useEffect(() => {
    if (!active) return
    const interval = setInterval(() => {
      setTimer(t => {
        if (t >= currentPhase.duration) {
          const nextPhase = (phase + 1) % currentTech.phases.length
          if (nextPhase === 0) setCycles(c => c + 1)
          setPhase(nextPhase)
          return 0
        }
        return t + 1
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [active, phase, currentPhase, currentTech])

  const scale = currentPhase.label === 'Inhale' ? 1.4 : currentPhase.label === 'Exhale' ? 0.8 : 1.1

  return (
    <div className="max-w-xl mx-auto">
      <MiloGuide message="Let's breathe together. Extended exhalation activates your vagus nerve, switching on the calm-down system. Follow the circle." mood="calm" />

      {/* Technique selector */}
      <div className="flex gap-2 mb-6">
        {TECHNIQUES.map((t, i) => (
          <button key={t.name} onClick={() => { setTechnique(i); setPhase(0); setTimer(0) }}
            className={`flex-1 py-2.5 rounded-pill text-xs font-medium transition-all ${technique === i ? 'bg-brand-300 text-white shadow-md' : 'bg-white border border-border text-gray-600 hover:border-brand-200'}`}>
            {t.name}
          </button>
        ))}
      </div>

      {/* Breathing circle */}
      <div className="bg-gradient-to-b from-gray-900 to-gray-800 rounded-card p-12 flex flex-col items-center mb-6">
        <motion.div
          animate={{ scale: active ? scale : 1 }}
          transition={{ duration: currentPhase.duration, ease: 'easeInOut' }}
          className="w-40 h-40 rounded-full bg-brand-300/20 border-4 border-brand-300 flex items-center justify-center mb-6"
        >
          <div className="text-center">
            <div className="text-white text-lg font-semibold">{active ? currentPhase.label : 'Ready'}</div>
            {active && <div className="text-brand-200 text-3xl font-bold font-mono mt-1">{currentPhase.duration - timer}</div>}
          </div>
        </motion.div>

        <button onClick={() => { setActive(!active); if (!active) { setPhase(0); setTimer(0) } }}
          className={`px-8 py-3 rounded-pill font-medium text-sm transition-all ${active ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 'bg-brand-300 text-white shadow-lg shadow-brand-300/30'}`}>
          {active ? 'Stop' : 'Start Breathing'}
        </button>

        <div className="text-gray-400 text-xs mt-4">Cycles completed: {cycles}</div>
      </div>

      {/* 5-4-3-2-1 Grounding */}
      <div className="bg-white border border-border rounded-card p-5">
        <h3 className="text-sm font-semibold text-gray-800 mb-3">5-4-3-2-1 Grounding Technique</h3>
        <div className="grid grid-cols-5 gap-2">
          {[{ n: 5, sense: 'See', icon: '👁️' }, { n: 4, sense: 'Touch', icon: '✋' }, { n: 3, sense: 'Hear', icon: '👂' }, { n: 2, sense: 'Smell', icon: '👃' }, { n: 1, sense: 'Taste', icon: '👅' }].map(s => (
            <div key={s.n} className="bg-gray-50 rounded-xl p-3 text-center">
              <div className="text-xl">{s.icon}</div>
              <div className="text-xs font-bold text-gray-700 mt-1">{s.n}</div>
              <div className="text-[9px] text-gray-400">{s.sense}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
