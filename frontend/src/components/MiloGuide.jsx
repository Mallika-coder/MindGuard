import { motion } from 'framer-motion'

export default function MiloGuide({ message, mood = 'neutral' }) {
  const moodConfig = {
    neutral: { color: 'brand-300', bg: 'bg-brand-50', border: 'border-brand-100' },
    happy: { color: 'brand-300', bg: 'bg-brand-50', border: 'border-brand-100' },
    thinking: { color: 'lavender-300', bg: 'bg-lavender-50', border: 'border-lavender-100' },
    caring: { color: 'pink-500', bg: 'bg-pink-50', border: 'border-pink-100' },
    alert: { color: 'red-500', bg: 'bg-red-50', border: 'border-red-100' },
    celebrate: { color: 'amber-500', bg: 'bg-amber-50', border: 'border-amber-100' },
  }
  const config = moodConfig[mood] || moodConfig.neutral

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-start gap-4 ${config.bg} ${config.border} border rounded-card p-4 mb-5`}
    >
      {/* Milo Avatar */}
      <div className="flex-shrink-0 w-12 h-12 bg-brand-300 rounded-full flex items-center justify-center shadow-lg shadow-brand-300/20 animate-float">
        <svg width="28" height="28" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="white" opacity="0.2"/>
          <circle cx="40" cy="45" r="5" fill="white"/>
          <circle cx="60" cy="45" r="5" fill="white"/>
          <circle cx="41" cy="46" r="2.5" fill="#1a3d2e"/>
          <circle cx="61" cy="46" r="2.5" fill="#1a3d2e"/>
          <path d="M 38 60 Q 50 70 62 60" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round"/>
        </svg>
      </div>
      {/* Speech */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-semibold text-brand-400 uppercase tracking-wide">Milo</span>
          <span className="text-[10px] bg-brand-100 text-brand-400 px-2 py-0.5 rounded-full font-medium">AI Guide</span>
        </div>
        <p className="text-sm text-gray-700 leading-relaxed">{message}</p>
      </div>
    </motion.div>
  )
}
