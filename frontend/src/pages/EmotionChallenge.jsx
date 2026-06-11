import { useState } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle, XCircle, ArrowRight } from 'lucide-react'
import MiloGuide from '../components/MiloGuide'
import { EMOTION_CHALLENGES, analyzeEmotions } from '../utils/wellness'

const EMOTION_OPTIONS = ['anger', 'sadness', 'fear', 'joy', 'disgust', 'surprise', 'trust', 'anticipation']

export default function EmotionChallenge() {
  const [idx, setIdx] = useState(0)
  const [guess, setGuess] = useState(null)
  const [submitted, setSubmitted] = useState(false)
  const [score, setScore] = useState({ correct: 0, total: 0 })

  const challenge = EMOTION_CHALLENGES[idx]
  const isCorrect = guess === challenge.answer

  const submit = () => {
    if (!guess) return
    setSubmitted(true)
    setScore(prev => ({ correct: prev.correct + (isCorrect ? 1 : 0), total: prev.total + 1 }))
  }

  const next = () => {
    setIdx((idx + 1) % EMOTION_CHALLENGES.length)
    setGuess(null)
    setSubmitted(false)
  }

  const emotions = analyzeEmotions(challenge.text)
  const topEmotions = Object.entries(emotions).sort((a,b) => b[1]-a[1]).slice(0, 5)

  return (
    <div>
      <MiloGuide message="Test your emotional intelligence! Read the passage, guess the dominant emotion, then see if you matched my ML model. Each round teaches you how NLP detects feelings." mood="celebrate" />

      {/* Score */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <span className="px-3 py-1 bg-brand-50 text-brand-400 rounded-full text-xs font-semibold">Round {idx + 1}/{EMOTION_CHALLENGES.length}</span>
          <span className="text-sm text-gray-500">Score: <strong className="text-brand-400">{score.correct}/{score.total}</strong></span>
        </div>
      </div>

      {/* Challenge text */}
      <div className="bg-white border border-border rounded-card p-6 mb-5">
        <div className="text-[10px] text-gray-400 uppercase tracking-wide mb-2">Read this passage:</div>
        <p className="text-base text-gray-800 leading-relaxed italic">"{challenge.text}"</p>
      </div>

      {/* Options */}
      {!submitted && (
        <div className="mb-5">
          <div className="text-sm font-medium text-gray-700 mb-3">Which emotion dominates?</div>
          <div className="grid grid-cols-4 gap-2">
            {EMOTION_OPTIONS.map(emo => (
              <button key={emo} onClick={() => setGuess(emo)}
                className={`py-2.5 px-3 rounded-xl text-xs font-medium capitalize transition-all ${
                  guess === emo ? 'bg-brand-300 text-white shadow-md' : 'bg-white border border-border text-gray-600 hover:border-brand-200 hover:bg-brand-50'
                }`}>{emo}</button>
            ))}
          </div>
          <button onClick={submit} disabled={!guess}
            className="mt-4 w-full py-3 bg-brand-300 text-white rounded-pill font-medium text-sm shadow-md hover:scale-[1.01] transition-all disabled:opacity-50">
            ✅ Check My Answer
          </button>
        </div>
      )}

      {/* Result */}
      {submitted && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          <div className={`p-5 rounded-card border ${isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
            <div className="flex items-center gap-3 mb-3">
              {isCorrect ? <CheckCircle className="w-6 h-6 text-green-500"/> : <XCircle className="w-6 h-6 text-red-500"/>}
              <div>
                <div className={`font-semibold ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>
                  {isCorrect ? 'Correct! You matched the AI!' : `Not quite — the answer is ${challenge.answer}`}
                </div>
                <div className="text-xs text-gray-500">Your guess: {guess} | Correct: {challenge.answer}</div>
              </div>
            </div>
            <p className="text-sm text-gray-600">{challenge.explanation}</p>
          </div>

          {/* AI breakdown */}
          <div className="bg-white border border-border rounded-card p-5">
            <div className="text-xs font-semibold text-gray-700 mb-3">ML Emotion Breakdown:</div>
            {topEmotions.map(([emo, score]) => (
              <div key={emo} className="flex items-center gap-2 mb-1.5">
                <span className="w-20 text-[11px] text-gray-500 capitalize">{emo}</span>
                <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full bg-lavender-200 rounded-full" style={{ width: `${score*100}%` }}/>
                </div>
                <span className="w-8 text-[10px] text-gray-400">{(score*100).toFixed(0)}%</span>
              </div>
            ))}
          </div>

          <button onClick={next} className="w-full py-3 bg-brand-300 text-white rounded-pill font-medium text-sm flex items-center justify-center gap-2 hover:scale-[1.01] transition-all">
            Next Round <ArrowRight className="w-4 h-4"/>
          </button>
        </motion.div>
      )}
    </div>
  )
}
