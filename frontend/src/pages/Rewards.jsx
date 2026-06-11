import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'
import { useLocalState } from '../hooks/useLocalState'

const BADGES = [
  { name: 'First Analysis', icon: '🔬', desc: 'Complete your first analysis', tier: 'bronze', check: d => d.analyses >= 1 },
  { name: 'Pattern Seeker', icon: '🧠', desc: 'Complete 5 analyses', tier: 'silver', check: d => d.analyses >= 5 },
  { name: 'Mind Explorer', icon: '🌟', desc: 'Complete 15 analyses', tier: 'gold', check: d => d.analyses >= 15 },
  { name: 'Self-Aware', icon: '🌡️', desc: 'Complete a daily check-in', tier: 'bronze', check: d => d.checkins >= 1 },
  { name: 'Consistent', icon: '📅', desc: '3 daily check-ins', tier: 'silver', check: d => d.checkins >= 3 },
  { name: 'Dedicated', icon: '💪', desc: '7 daily check-ins', tier: 'gold', check: d => d.checkins >= 7 },
  { name: 'Reflective', icon: '📓', desc: 'Write a journal entry', tier: 'bronze', check: d => d.journal >= 1 },
  { name: 'Deep Thinker', icon: '✍️', desc: '5 journal entries', tier: 'silver', check: d => d.journal >= 5 },
  { name: 'Thought Reframer', icon: '🔄', desc: 'Use CBT reframing', tier: 'silver', check: d => d.reframes >= 1 },
  { name: 'CBT Master', icon: '🎓', desc: '5 CBT reframes', tier: 'gold', check: d => d.reframes >= 5 },
  { name: 'Healthy Mind', icon: '💚', desc: 'Get risk below 15%', tier: 'silver', check: d => d.lowRisk },
  { name: 'Thriving', icon: '🌈', desc: '80%+ wellness check-in', tier: 'gold', check: d => d.highWellness },
]

export default function Rewards() {
  const [checkins] = useLocalState('mg-checkins', [])
  const [journal] = useLocalState('mg-journal', [])

  const data = { analyses: 0, checkins: checkins.length, journal: journal.length, reframes: 0, lowRisk: false, highWellness: checkins.some(c => (c.mood + c.sleep + c.energy + c.social) / 20 >= 0.8) }
  const unlocked = BADGES.filter(b => b.check(data))
  const xp = unlocked.length * 100
  const level = Math.floor(xp / 300) + 1
  const xpInLevel = xp % 300

  const tierColors = { bronze: { bg: '#fef3c7', border: '#fbbf24', text: '#92400e' }, silver: { bg: '#f1f5f9', border: '#94a3b8', text: '#475569' }, gold: { bg: '#fefce8', border: '#eab308', text: '#854d0e' } }

  return (
    <div>
      <MiloGuide message={`You've earned ${unlocked.length}/${BADGES.length} badges and ${xp} XP! Every interaction counts toward your next achievement.`} mood="celebrate" />

      {/* Level display */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-card p-6 text-center mb-6">
        <div className="text-xs text-gray-400 uppercase tracking-wide">Wellness Level</div>
        <div className="text-4xl font-bold text-white mt-1">Level {level}</div>
        <div className="text-sm text-gray-400 mt-1">{xp} XP Total</div>
        <div className="mt-3 max-w-xs mx-auto h-3 bg-gray-700 rounded-full overflow-hidden">
          <motion.div className="h-full bg-brand-300 rounded-full" initial={{ width: 0 }} animate={{ width: `${xpInLevel/300*100}%` }} transition={{ duration: 0.8 }}/>
        </div>
        <div className="text-[10px] text-gray-500 mt-1">{xpInLevel}/300 XP to Level {level+1}</div>
      </div>

      {/* Badge grid */}
      <div className="grid grid-cols-4 gap-3">
        {BADGES.map(b => {
          const earned = b.check(data)
          const tc = tierColors[b.tier]
          return (
            <div key={b.name} className={`border rounded-card p-4 text-center transition-all ${earned ? 'opacity-100 shadow-sm' : 'opacity-40'}`}
              style={{ background: earned ? tc.bg : '#f8fafc', borderColor: earned ? tc.border : '#e8ecf0' }}>
              <div className="text-2xl mb-1">{b.icon}</div>
              <div className="text-[11px] font-semibold text-gray-800">{b.name}</div>
              <div className="text-[9px] text-gray-500 mt-0.5">{b.desc}</div>
              <div className="text-[8px] font-semibold uppercase mt-1" style={{ color: tc.text }}>{b.tier}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
