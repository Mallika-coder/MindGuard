import { motion } from 'framer-motion'

export default function MiloGuide({ message, mood = 'neutral' }) {
  const moodColors = {
    neutral: { bg: 'bg-emerald-50', border: 'border-emerald-200', accent: 'text-emerald-700', dot: 'bg-emerald-400' },
    happy: { bg: 'bg-emerald-50', border: 'border-emerald-200', accent: 'text-emerald-700', dot: 'bg-emerald-400' },
    thinking: { bg: 'bg-indigo-50', border: 'border-indigo-200', accent: 'text-indigo-700', dot: 'bg-indigo-400' },
    caring: { bg: 'bg-pink-50', border: 'border-pink-200', accent: 'text-pink-700', dot: 'bg-pink-400' },
    alert: { bg: 'bg-red-50', border: 'border-red-200', accent: 'text-red-700', dot: 'bg-red-400' },
    celebrate: { bg: 'bg-amber-50', border: 'border-amber-200', accent: 'text-amber-700', dot: 'bg-amber-400' },
    calm: { bg: 'bg-cyan-50', border: 'border-cyan-200', accent: 'text-cyan-700', dot: 'bg-cyan-400' },
  }
  const colors = moodColors[mood] || moodColors.neutral

  return (
    <div className={`flex items-center gap-5 ${colors.bg} ${colors.border} border rounded-2xl p-4 pr-6 mb-6 relative overflow-hidden`}>
      {/* Background decoration */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-white/30 rounded-full -translate-y-1/2 translate-x-1/2"/>

      {/* Character Image */}
      <motion.div
        className="flex-shrink-0 relative z-10"
        animate={{ y: [0, -4, 0] }}
        transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
      >
        <img
          src="/milo-character.avif"
          alt="Dr. Milo - AI Health Guide"
          className="w-20 h-20 object-contain rounded-xl drop-shadow-lg"
        />
      </motion.div>

      {/* Speech bubble */}
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
        className="relative z-10 flex-1"
      >
        <div className="bg-white rounded-xl p-3.5 shadow-sm border border-white/80 relative">
          {/* Bubble arrow */}
          <div className="absolute left-[-6px] top-4 w-3 h-3 bg-white border-l border-b border-white/80 rotate-45"/>
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-[10px] font-bold uppercase tracking-wide ${colors.accent}`}>Dr. Milo</span>
            <motion.div
              className={`w-1.5 h-1.5 rounded-full ${colors.dot}`}
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            <span className="text-[9px] text-gray-400">AI Health Guide</span>
          </div>
          <p className="text-[13px] text-gray-700 leading-relaxed">{message}</p>
        </div>
      </motion.div>
    </div>
  )
}
