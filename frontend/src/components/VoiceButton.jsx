import { motion } from 'framer-motion'
import { Mic, MicOff, Volume2 } from 'lucide-react'

export function VoiceInputButton({ isListening, onStart, onStop }) {
  return (
    <motion.button
      type="button"
      onClick={isListening ? onStop : onStart}
      whileTap={{ scale: 0.95 }}
      className={`relative w-10 h-10 rounded-full flex items-center justify-center transition-all ${
        isListening
          ? 'bg-red-500 text-white shadow-lg shadow-red-500/30'
          : 'bg-gray-100 text-gray-500 hover:bg-brand-50 hover:text-brand-400'
      }`}
      title={isListening ? 'Stop listening' : 'Voice input'}
    >
      {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
      {isListening && (
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-red-400"
          animate={{ scale: [1, 1.4, 1], opacity: [0.8, 0, 0.8] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </motion.button>
  )
}

export function SpeakButton({ onClick, speaking }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileTap={{ scale: 0.95 }}
      className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
        speaking ? 'bg-brand-300 text-white' : 'bg-gray-100 text-gray-400 hover:bg-brand-50 hover:text-brand-400'
      }`}
      title="Read aloud"
    >
      <Volume2 className="w-3.5 h-3.5" />
    </motion.button>
  )
}
