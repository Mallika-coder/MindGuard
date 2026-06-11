import { motion } from 'framer-motion'

function GuardianCharacter({ size = 80 }) {
  const s = size
  return (
    <motion.svg
      width={s} height={s} viewBox="0 0 200 200"
      animate={{ y: [0, -4, 0] }}
      transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
      style={{ filter: 'drop-shadow(0 6px 12px rgba(74, 157, 110, 0.25))' }}
    >
      {/* Hair */}
      <ellipse cx="100" cy="72" rx="42" ry="44" fill="#2d1b0e"/>
      <path d="M 58 72 Q 58 40 100 35 Q 142 40 142 72" fill="#3d2b1a"/>
      <path d="M 62 55 Q 80 30 100 32 Q 120 30 138 55" fill="#2d1b0e"/>
      {/* Side hair */}
      <path d="M 58 72 Q 52 90 55 110" fill="#2d1b0e" stroke="none"/>
      <path d="M 142 72 Q 148 90 145 110" fill="#2d1b0e" stroke="none"/>

      {/* Face */}
      <ellipse cx="100" cy="82" rx="34" ry="36" fill="#fcd5b4"/>
      {/* Face highlights */}
      <ellipse cx="92" cy="72" rx="18" ry="14" fill="#fde8d4" opacity="0.4"/>

      {/* Eyes */}
      <g>
        <ellipse cx="86" cy="82" rx="7" ry="8" fill="white"/>
        <ellipse cx="114" cy="82" rx="7" ry="8" fill="white"/>
        <motion.g animate={{ scaleY: [1, 1, 0.1, 1, 1] }} transition={{ duration: 4, repeat: Infinity, times: [0, 0.45, 0.5, 0.55, 1] }}>
          <circle cx="87" cy="83" r="4" fill="#2d5a3f"/>
          <circle cx="115" cy="83" r="4" fill="#2d5a3f"/>
          <circle cx="88.5" cy="81" r="1.5" fill="white"/>
          <circle cx="116.5" cy="81" r="1.5" fill="white"/>
        </motion.g>
      </g>

      {/* Eyebrows */}
      <path d="M 79 73 Q 86 69 93 72" fill="none" stroke="#3d2b1a" strokeWidth="2" strokeLinecap="round"/>
      <path d="M 107 72 Q 114 69 121 73" fill="none" stroke="#3d2b1a" strokeWidth="2" strokeLinecap="round"/>

      {/* Nose */}
      <path d="M 100 88 Q 97 92 100 94 Q 103 92 100 88" fill="#e8b896" opacity="0.6"/>

      {/* Smile */}
      <motion.path
        d="M 88 99 Q 100 110 112 99"
        fill="none" stroke="#c47a5a" strokeWidth="2.5" strokeLinecap="round"
        animate={{ d: ["M 88 99 Q 100 110 112 99", "M 88 99 Q 100 108 112 99", "M 88 99 Q 100 110 112 99"] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      {/* Blush */}
      <circle cx="76" cy="94" r="6" fill="#ffb3b3" opacity="0.35"/>
      <circle cx="124" cy="94" r="6" fill="#ffb3b3" opacity="0.35"/>

      {/* Body / Lab coat */}
      <path d="M 70 120 Q 70 115 80 112 L 100 110 L 120 112 Q 130 115 130 120 L 135 170 L 65 170 Z" fill="white" stroke="#e2e8f0" strokeWidth="1"/>
      {/* Coat lapels */}
      <path d="M 85 112 L 95 135 L 100 130" fill="none" stroke="#d1d5db" strokeWidth="1.5"/>
      <path d="M 115 112 L 105 135 L 100 130" fill="none" stroke="#d1d5db" strokeWidth="1.5"/>
      {/* Green scrub underneath */}
      <path d="M 88 112 L 88 125 Q 100 130 112 125 L 112 112" fill="#4a9d6e"/>

      {/* Stethoscope */}
      <path d="M 92 118 Q 85 130 88 140" fill="none" stroke="#64748b" strokeWidth="2" strokeLinecap="round"/>
      <circle cx="88" cy="142" r="4" fill="#94a3b8" stroke="#64748b" strokeWidth="1.5"/>

      {/* Badge */}
      <rect x="118" y="130" width="14" height="18" rx="3" fill="#fbbf24" stroke="#f59e0b" strokeWidth="0.8"/>
      <text x="125" y="142" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">MG</text>

      {/* Waving hand */}
      <motion.g
        animate={{ rotate: [0, 15, -8, 12, -4, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
        style={{ transformOrigin: '145px 135px' }}
      >
        <path d="M 130 120 Q 145 115 155 125" fill="none" stroke="#fcd5b4" strokeWidth="8" strokeLinecap="round"/>
        <circle cx="157" cy="125" r="7" fill="#fcd5b4"/>
        {/* Fingers */}
        <path d="M 155 119 L 157 114" fill="none" stroke="#fcd5b4" strokeWidth="3" strokeLinecap="round"/>
        <path d="M 159 120 L 162 116" fill="none" stroke="#fcd5b4" strokeWidth="3" strokeLinecap="round"/>
        <path d="M 162 123 L 165 120" fill="none" stroke="#fcd5b4" strokeWidth="3" strokeLinecap="round"/>
      </motion.g>

      {/* Glasses (optional health professional look) */}
      <circle cx="86" cy="82" r="10" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.7"/>
      <circle cx="114" cy="82" r="10" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.7"/>
      <path d="M 96 82 L 104 82" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.7"/>
    </motion.svg>
  )
}

const SPEECH_VARIANTS = {
  hidden: { opacity: 0, x: -10, scale: 0.95 },
  visible: { opacity: 1, x: 0, scale: 1, transition: { duration: 0.3, ease: 'easeOut' } },
}

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

      {/* Character */}
      <div className="flex-shrink-0 relative z-10">
        <GuardianCharacter size={90} />
      </div>

      {/* Speech bubble */}
      <motion.div variants={SPEECH_VARIANTS} initial="hidden" animate="visible" className="relative z-10 flex-1">
        <div className="bg-white rounded-xl p-3.5 shadow-sm border border-white/80 relative">
          {/* Bubble arrow */}
          <div className="absolute left-[-6px] top-4 w-3 h-3 bg-white border-l border-b border-white/80 rotate-45"/>
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-[10px] font-bold uppercase tracking-wide ${colors.accent}`}>Dr. Milo</span>
            <div className={`w-1.5 h-1.5 rounded-full ${colors.dot} animate-pulse`}/>
            <span className="text-[9px] text-gray-400">AI Health Guide</span>
          </div>
          <p className="text-[13px] text-gray-700 leading-relaxed">{message}</p>
        </div>
      </motion.div>
    </div>
  )
}
