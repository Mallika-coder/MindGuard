import MiloGuide from '../components/MiloGuide'
import { useLocalState } from '../hooks/useLocalState'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

export default function Analytics() {
  const [checkins] = useLocalState('mg-checkins', [])
  const [journal] = useLocalState('mg-journal', [])

  const moodData = checkins.slice(-14).map((c, i) => ({ day: `D${i+1}`, mood: c.mood, sleep: c.sleep, energy: c.energy }))
  const journalDist = {}
  journal.forEach(e => { journalDist[e.classification] = (journalDist[e.classification] || 0) + 1 })
  const distData = Object.entries(journalDist).map(([name, value]) => ({ name, value }))

  return (
    <div>
      <MiloGuide message="Your session analytics! I track patterns across all interactions. Use Deep Analysis and Daily Check-ins to build your data." mood="thinking" />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { label: 'Check-ins', value: checkins.length, color: 'brand' },
          { label: 'Journal', value: journal.length, color: 'lavender' },
          { label: 'Avg Mood', value: checkins.length ? (checkins.reduce((a,c) => a+c.mood, 0)/checkins.length).toFixed(1) : '-', color: 'brand' },
          { label: 'Streak', value: `${checkins.length}d`, color: 'amber' },
        ].map(s => (
          <div key={s.label} className="bg-white border border-border rounded-card p-4 text-center">
            <div className="text-2xl font-bold font-mono text-gray-800">{s.value}</div>
            <div className="text-[10px] text-gray-400 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-border rounded-card p-5">
          <h3 className="text-sm font-semibold text-gray-800 mb-4">Mood Trend</h3>
          {moodData.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={moodData}>
                <defs><linearGradient id="moodGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#4a9d6e" stopOpacity={0.3}/><stop offset="95%" stopColor="#4a9d6e" stopOpacity={0}/></linearGradient></defs>
                <XAxis dataKey="day" tick={{ fontSize: 10 }}/>
                <YAxis domain={[1,5]} tick={{ fontSize: 10 }}/>
                <Tooltip/>
                <Area type="monotone" dataKey="mood" stroke="#4a9d6e" fill="url(#moodGrad)"/>
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="text-center py-12 text-sm text-gray-400">No check-in data yet</div>}
        </div>

        <div className="bg-white border border-border rounded-card p-5">
          <h3 className="text-sm font-semibold text-gray-800 mb-4">Journal Classification</h3>
          {distData.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={distData}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }}/>
                <YAxis tick={{ fontSize: 10 }}/>
                <Tooltip/>
                <Bar dataKey="value" fill="#7c6aad" radius={[4,4,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="text-center py-12 text-sm text-gray-400">No journal data yet</div>}
        </div>
      </div>
    </div>
  )
}
