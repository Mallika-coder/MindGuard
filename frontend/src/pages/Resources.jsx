import { useState } from 'react'
import { motion } from 'framer-motion'
import MiloGuide from '../components/MiloGuide'

const RESOURCES = [
  { title: 'Understanding CBT', category: 'Therapy', icon: '🧠', video: 'https://www.youtube.com/embed/ZdyOwZ4_RnI', content: 'CBT is the gold-standard treatment. It identifies negative thought patterns and systematically challenges them. 60-80% of patients improve in 12-16 sessions.' },
  { title: 'Sleep Hygiene', category: 'Sleep', icon: '🌙', video: 'https://www.youtube.com/embed/nm1TxQj9IsQ', content: 'Key practices: consistent schedule, dark/cool room (65-68°F), no screens 1hr before bed, limit caffeine after 2PM.' },
  { title: '5-4-3-2-1 Grounding', category: 'Mindfulness', icon: '🧘', video: 'https://www.youtube.com/embed/30VMIEmA114', content: 'Name 5 things you SEE, 4 TOUCH, 3 HEAR, 2 SMELL, 1 TASTE. Reduces acute anxiety within 2-3 minutes.' },
  { title: 'Healthy Boundaries', category: 'Relationships', icon: '💬', video: 'https://www.youtube.com/embed/4E1JiDFxFGk', content: 'Start small: "I need time to think." Practice saying no without over-explaining. Boundaries aren\'t walls.' },
  { title: 'Stress-Performance Curve', category: 'Stress', icon: '📈', video: 'https://www.youtube.com/embed/RcGyVTAoXEU', content: 'Some stress improves performance (eustress). Yerkes-Dodson law: performance peaks at moderate arousal.' },
  { title: 'Mindful Breathing', category: 'Mindfulness', icon: '🫁', video: 'https://www.youtube.com/embed/tEmt1Znux58', content: 'Extended exhalation stimulates the vagus nerve. 4-7-8 technique reduces cortisol within 2-3 cycles.' },
  { title: 'Digital Wellness', category: 'Productivity', icon: '📱', video: 'https://www.youtube.com/embed/3E7hkPZ-HTk', content: 'Social media >2hrs/day correlates with increased anxiety. Try: batch notifications, grayscale mode, app timers.' },
  { title: 'Exercise as Medicine', category: 'Self-Care', icon: '🏃', video: 'https://www.youtube.com/embed/DsVzKCk066g', content: '30 min exercise = antidepressants for mild-moderate depression (Blumenthal, 2007). Even 10-min walk helps.' },
]

const CATEGORIES = ['All', 'Therapy', 'Sleep', 'Mindfulness', 'Relationships', 'Stress', 'Productivity', 'Self-Care']

export default function Resources() {
  const [category, setCategory] = useState('All')
  const filtered = category === 'All' ? RESOURCES : RESOURCES.filter(r => r.category === category)

  return (
    <div>
      <MiloGuide message="Watch, read, and learn! Each resource includes a video plus evidence-based insights. Filter by topic to find what's relevant to you." mood="happy" />

      {/* Filter */}
      <div className="flex gap-1.5 flex-wrap mb-5">
        {CATEGORIES.map(c => (
          <button key={c} onClick={() => setCategory(c)}
            className={`px-3 py-1.5 rounded-full text-[11px] font-medium transition-all ${category === c ? 'bg-brand-300 text-white' : 'bg-white border border-border text-gray-500 hover:border-brand-200'}`}>{c}</button>
        ))}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 gap-4">
        {filtered.map((r, i) => (
          <motion.div key={r.title} initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
            className="bg-white border border-border rounded-card overflow-hidden hover:shadow-md hover:-translate-y-0.5 transition-all">
            <div className="aspect-video">
              <iframe src={r.video} className="w-full h-full border-0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen/>
            </div>
            <div className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">{r.icon}</span>
                <div>
                  <div className="text-sm font-semibold text-gray-800">{r.title}</div>
                  <span className="text-[10px] bg-brand-50 text-brand-400 px-2 py-0.5 rounded-full">{r.category}</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">{r.content}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
