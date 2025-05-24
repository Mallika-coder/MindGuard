import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ScreeningPanel from './components/ScreeningPanel'
import ChatPanel from './components/ChatPanel'
import ResultsPanel from './components/ResultsPanel'
import { Brain, MessageCircle, Shield } from 'lucide-react'

export default function App() {
  const [activeTab, setActiveTab] = useState('screen')
  const [screeningResult, setScreeningResult] = useState(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-green-500 rounded-xl flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">MindGuard</h1>
              <p className="text-xs text-gray-500">AI Mental Health Screening</p>
            </div>
          </div>
          <nav className="flex gap-1 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('screen')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeTab === 'screen'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Brain className="w-4 h-4 inline mr-1.5" />
              Screen
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeTab === 'chat'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <MessageCircle className="w-4 h-4 inline mr-1.5" />
              Chat
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          {activeTab === 'screen' && (
            <motion.div
              key="screen"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-8"
            >
              <ScreeningPanel onResult={setScreeningResult} />
              <ResultsPanel result={screeningResult} />
            </motion.div>
          )}
          {activeTab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <ChatPanel />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Disclaimer */}
        <div className="mt-12 p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-center">
          <p className="text-sm text-yellow-800">
            <strong>Disclaimer:</strong> MindGuard is an AI screening tool for educational purposes.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            If you're in crisis, please call <strong>988</strong> (Suicide & Crisis Lifeline).
          </p>
        </div>
      </main>
    </div>
  )
}
