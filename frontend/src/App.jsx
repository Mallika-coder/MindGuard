import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './components/Sidebar'
import Landing from './pages/Landing'
import Analysis from './pages/Analysis'
import EmotionChallenge from './pages/EmotionChallenge'
import Pipeline from './pages/Pipeline'
import Chat from './pages/Chat'
import ThoughtReframer from './pages/ThoughtReframer'
import PHQ9 from './pages/PHQ9'
import GAD7 from './pages/GAD7'
import DailyCheckin from './pages/DailyCheckin'
import Breathing from './pages/Breathing'
import Journal from './pages/Journal'
import Rewards from './pages/Rewards'
import Resources from './pages/Resources'
import Analytics from './pages/Analytics'
import HowItWorks from './pages/HowItWorks'

const PAGE_TITLES = {
  home: 'Home',
  analysis: 'Deep Analysis',
  challenge: 'Emotion Challenge',
  pipeline: 'ML Pipeline',
  chat: 'AI Companion',
  reframe: 'Thought Reframer',
  phq9: 'PHQ-9 Depression Screen',
  gad7: 'GAD-7 Anxiety Screen',
  checkin: 'Daily Check-in',
  breathing: 'Breathing Exercise',
  journal: 'Mood Journal',
  rewards: 'Rewards',
  resources: 'Resources',
  analytics: 'Analytics',
  how: 'How It Works',
}

export default function App() {
  const [activeView, setActiveView] = useState('home')

  const renderView = () => {
    switch (activeView) {
      case 'home': return <Landing onNavigate={setActiveView} />
      case 'analysis': return <Analysis />
      case 'challenge': return <EmotionChallenge />
      case 'pipeline': return <Pipeline />
      case 'chat': return <Chat />
      case 'reframe': return <ThoughtReframer />
      case 'phq9': return <PHQ9 />
      case 'gad7': return <GAD7 />
      case 'checkin': return <DailyCheckin />
      case 'breathing': return <Breathing />
      case 'journal': return <Journal />
      case 'rewards': return <Rewards />
      case 'resources': return <Resources />
      case 'analytics': return <Analytics />
      case 'how': return <HowItWorks />
      default: return <Landing onNavigate={setActiveView} />
    }
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar activeView={activeView} onNavigate={setActiveView} />
      <main className="flex-1 ml-64 p-8 relative">
        {/* Page header */}
        {activeView !== 'home' && (
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-gray-900">{PAGE_TITLES[activeView]}</h1>
            <p className="text-sm text-gray-500 mt-1">MindGuard AI Platform</p>
          </div>
        )}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeView}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            {renderView()}
          </motion.div>
        </AnimatePresence>

        {/* Disclaimer */}
        <div className="mt-12 p-3 bg-amber-50 border border-amber-200 rounded-input text-center">
          <p className="text-[11px] text-amber-800">
            ⚠️ MindGuard is an educational AI tool, NOT medical advice. In crisis, call <strong>988</strong> or text HOME to 741741.
          </p>
        </div>
      </main>
    </div>
  )
}
