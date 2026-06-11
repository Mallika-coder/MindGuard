import { motion } from 'framer-motion'
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

      {/* Milo character */}
      <motion.div
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className="mb-6"
      >
        <div className="w-28 h-28 bg-gradient-to-br from-brand-300 to-brand-400 rounded-full flex items-center justify-center shadow-2xl shadow-brand-300/30">
          <svg width="64" height="64" viewBox="0 0 100 100">
            <circle cx="40" cy="42" r="7" fill="white"/>
            <circle cx="60" cy="42" r="7" fill="white"/>
            <circle cx="41" cy="43" r="3.5" fill="#1a3d2e"/>
            <circle cx="61" cy="43" r="3.5" fill="#1a3d2e"/>
            <circle cx="43" cy="40" r="1.5" fill="white"/>
            <circle cx="63" cy="40" r="1.5" fill="white"/>
            <path d="M 36 58 Q 50 72 64 58" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round"/>
            <circle cx="30" cy="52" r="5" fill="white" opacity="0.3"/>
            <circle cx="70" cy="52" r="5" fill="white" opacity="0.3"/>
          </svg>
        </div>
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
