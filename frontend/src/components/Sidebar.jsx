import { motion } from 'framer-motion'
import { Brain, Gamepad2, Microscope, RefreshCw, ClipboardList, MessageCircle, Thermometer, Wind, BookOpen, Trophy, Library, BarChart3, Cog, Phone, Home } from 'lucide-react'

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'analysis', label: 'Deep Analysis', icon: Brain },
  { id: 'challenge', label: 'Emotion Challenge', icon: Gamepad2 },
  { id: 'pipeline', label: 'ML Pipeline', icon: Microscope },
  { id: 'chat', label: 'AI Companion', icon: MessageCircle },
  { id: 'reframe', label: 'Thought Reframer', icon: RefreshCw },
  { id: 'phq9', label: 'PHQ-9', icon: ClipboardList },
  { id: 'gad7', label: 'GAD-7', icon: ClipboardList },
  { id: 'checkin', label: 'Daily Check-in', icon: Thermometer },
  { id: 'breathing', label: 'Breathing', icon: Wind },
  { id: 'journal', label: 'Journal', icon: BookOpen },
  { id: 'rewards', label: 'Rewards', icon: Trophy },
  { id: 'resources', label: 'Resources', icon: Library },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'how', label: 'How It Works', icon: Cog },
]

export default function Sidebar({ activeView, onNavigate }) {
  return (
    <aside className="w-64 h-screen bg-brand-500 text-white flex flex-col fixed left-0 top-0 z-40 overflow-y-auto">
      {/* Logo */}
      <div className="p-5 border-b border-brand-400/30">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-brand-300 rounded-xl flex items-center justify-center text-white font-bold text-sm">M</div>
          <div>
            <div className="font-bold text-sm tracking-tight">MindGuard</div>
            <div className="text-[10px] text-brand-200 opacity-70">AI Health Platform</div>
          </div>
        </div>
      </div>

      {/* Nav Items */}
      <nav className="flex-1 py-3 px-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(item => {
          const Icon = item.icon
          const isActive = activeView === item.id
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-left text-[13px] font-medium transition-all duration-200 relative ${
                isActive
                  ? 'bg-white/15 text-white shadow-sm'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="nav-active"
                  className="absolute inset-0 bg-white/10 rounded-xl"
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
              <Icon className="w-4 h-4 relative z-10 flex-shrink-0" />
              <span className="relative z-10 truncate">{item.label}</span>
            </button>
          )
        })}
      </nav>

      {/* Crisis Line */}
      <div className="p-4 border-t border-brand-400/30">
        <a href="tel:988" className="flex items-center gap-2 px-3 py-2.5 bg-red-500/20 rounded-xl text-red-200 hover:bg-red-500/30 transition-all">
          <Phone className="w-4 h-4" />
          <div>
            <div className="text-[11px] font-semibold">Crisis Line: 988</div>
            <div className="text-[9px] opacity-70">24/7 Support</div>
          </div>
        </a>
      </div>
    </aside>
  )
}
