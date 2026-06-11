import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Brain, Gamepad2, Microscope, MessageCircle, Shield } from 'lucide-react'

const FEATURES = [
  { icon: Brain, title: 'Deep Analysis', desc: '6-dimension NLP scan', color: 'from-indigo-500 to-purple-500' },
  { icon: Gamepad2, title: 'Emotion Challenge', desc: 'Test your EQ vs AI', color: 'from-emerald-500 to-teal-500' },
  { icon: Microscope, title: 'ML Pipeline', desc: 'Watch AI think live', color: 'from-violet-500 to-fuchsia-500' },
  { icon: MessageCircle, title: 'CBT Companion', desc: 'Therapy-grade chat', color: 'from-amber-500 to-orange-500' },
]

export default function Landing({ onNavigate }) {
  return (
    <div className="min-h-[85vh] flex flex-col items-center justify-center text-center px-6">
      {/* Floating particles */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-brand-300 rounded-full opacity-20"
            style={{ top: `${15 + i * 15}%`, left: `${10 + i * 16}%` }}
            animate={{ y: [-20, 20, -20], x: [-10, 10, -10] }}
            transition={{ duration: 5 + i, repeat: Infinity, ease: 'easeInOut' }}
          />
        ))}
      </div>

      {/* Dr. Milo character — large animated */}
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className="mb-4"
      >
        <svg width="160" height="160" viewBox="0 0 200 200" style={{ filter: 'drop-shadow(0 12px 24px rgba(74, 157, 110, 0.2))' }}>
          <defs>
            <radialGradient id="face-light" cx="40%" cy="35%"><stop offset="0%" stopColor="#fde8d4"/><stop offset="100%" stopColor="#fcd5b4"/></radialGradient>
          </defs>
          {/* Hair */}
          <ellipse cx="100" cy="72" rx="42" ry="44" fill="#2d1b0e"/>
          <path d="M 62 55 Q 80 30 100 32 Q 120 30 138 55" fill="#2d1b0e"/>
          <path d="M 58 72 Q 52 90 55 110" fill="#2d1b0e"/><path d="M 142 72 Q 148 90 145 110" fill="#2d1b0e"/>
          {/* Face */}
          <ellipse cx="100" cy="82" rx="34" ry="36" fill="url(#face-light)"/>
          {/* Eyes */}
          <ellipse cx="86" cy="82" rx="7" ry="8" fill="white"/><ellipse cx="114" cy="82" rx="7" ry="8" fill="white"/>
          <circle cx="87" cy="83" r="4" fill="#2d5a3f"/><circle cx="115" cy="83" r="4" fill="#2d5a3f"/>
          <circle cx="88.5" cy="81" r="1.5" fill="white"/><circle cx="116.5" cy="81" r="1.5" fill="white"/>
          {/* Eyebrows */}
          <path d="M 79 73 Q 86 69 93 72" fill="none" stroke="#3d2b1a" strokeWidth="2" strokeLinecap="round"/>
          <path d="M 107 72 Q 114 69 121 73" fill="none" stroke="#3d2b1a" strokeWidth="2" strokeLinecap="round"/>
          {/* Smile */}
          <path d="M 88 99 Q 100 110 112 99" fill="none" stroke="#c47a5a" strokeWidth="2.5" strokeLinecap="round"/>
          {/* Blush */}
          <circle cx="76" cy="94" r="6" fill="#ffb3b3" opacity="0.3"/><circle cx="124" cy="94" r="6" fill="#ffb3b3" opacity="0.3"/>
          {/* Glasses */}
          <circle cx="86" cy="82" r="10" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.6"/>
          <circle cx="114" cy="82" r="10" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.6"/>
          <path d="M 96 82 L 104 82" fill="none" stroke="#64748b" strokeWidth="1.5" opacity="0.6"/>
          {/* Body */}
          <path d="M 70 120 Q 70 115 80 112 L 100 110 L 120 112 Q 130 115 130 120 L 135 180 L 65 180 Z" fill="white" stroke="#e2e8f0" strokeWidth="1"/>
          <path d="M 88 112 L 88 125 Q 100 130 112 125 L 112 112" fill="#4a9d6e"/>
          {/* Stethoscope */}
          <path d="M 92 118 Q 85 130 88 142" fill="none" stroke="#64748b" strokeWidth="2" strokeLinecap="round"/>
          <circle cx="88" cy="144" r="4" fill="#94a3b8" stroke="#64748b" strokeWidth="1.5"/>
          {/* Badge */}
          <rect x="118" y="132" width="14" height="18" rx="3" fill="#fbbf24" stroke="#f59e0b" strokeWidth="0.8"/>
          <text x="125" y="144" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">MG</text>
        </svg>
      </motion.div>

      {/* Speech bubble from character */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5 }}
        className="bg-white border border-brand-100 rounded-2xl px-5 py-3 mb-6 shadow-sm max-w-sm relative"
      >
        <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-white border-l border-t border-brand-100 rotate-45"/>
        <p className="text-sm text-gray-600 text-center relative z-10">
          "Hi! I'm <strong className="text-brand-400">Dr. Milo</strong> — your AI health companion. Let me guide you through your mental wellness journey!"
        </p>
      </motion.div>

      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-5xl font-extrabold tracking-tight text-gray-900"
      >
        Welcome to <span className="bg-gradient-to-r from-brand-300 to-lavender-300 bg-clip-text text-transparent">MindGuard</span>
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-4 text-lg text-gray-500 max-w-lg"
      >
        AI-powered mental health screening with <strong className="text-gray-700">BERT</strong>, <strong className="text-gray-700">FAISS</strong>, and <strong className="text-gray-700">CBT therapy</strong> — running live in your browser.
      </motion.p>

      {/* CTA */}
      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        onClick={() => onNavigate('analysis')}
        className="mt-8 px-8 py-3.5 bg-brand-300 text-white font-semibold rounded-pill shadow-lg shadow-brand-300/25 hover:shadow-xl hover:shadow-brand-300/30 hover:scale-[1.02] transition-all flex items-center gap-2"
      >
        Start Screening <ArrowRight className="w-4 h-4" />
      </motion.button>

      {/* Feature cards */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-14 grid grid-cols-4 gap-4 w-full max-w-2xl"
      >
        {FEATURES.map((f, i) => {
          const Icon = f.icon
          return (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.08 }}
              className="bg-white border border-border rounded-card p-5 text-center hover:shadow-md hover:-translate-y-1 transition-all cursor-pointer"
              onClick={() => onNavigate(
                i === 0 ? 'analysis' : i === 1 ? 'challenge' : i === 2 ? 'pipeline' : 'chat'
              )}
            >
              <div className={`w-10 h-10 bg-gradient-to-br ${f.color} rounded-xl flex items-center justify-center mx-auto mb-3 shadow-sm`}>
                <Icon className="w-5 h-5 text-white" />
              </div>
              <div className="text-xs font-semibold text-gray-800">{f.title}</div>
              <div className="text-[10px] text-gray-400 mt-1">{f.desc}</div>
            </motion.div>
          )
        })}
      </motion.div>

      {/* Trust badges */}
      <div className="mt-10 flex gap-2 flex-wrap justify-center">
        {['BERT NLP', 'FAISS Vectors', 'LangChain RAG', 'Plutchik Emotions', 'CBT', 'PHQ-9 / GAD-7'].map(t => (
          <span key={t} className="px-3 py-1 bg-brand-50 text-brand-400 rounded-full text-[10px] font-semibold">{t}</span>
        ))}
      </div>

      {/* Safety */}
      <div className="mt-6 flex items-center gap-2 text-[11px] text-gray-400">
        <Shield className="w-3.5 h-3.5" />
        <span>Private & secure • No data stored • Not a medical diagnosis</span>
      </div>
    </div>
  )
}
