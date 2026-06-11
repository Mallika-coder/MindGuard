import { useState, useRef, useCallback } from 'react'

function detectLanguage(text) {
  const hindiRegex = /[ऀ-ॿ]/
  const arabicRegex = /[؀-ۿ]/
  const chineseRegex = /[一-鿿]/
  const japaneseRegex = /[぀-ゟ゠-ヿ]/
  const koreanRegex = /[가-힯]/
  const spanishIndicators = /[áéíóúñ¿¡]/i
  const frenchIndicators = /[àâçéèêëîïôùûü]/i

  if (hindiRegex.test(text)) return 'hi-IN'
  if (arabicRegex.test(text)) return 'ar-SA'
  if (chineseRegex.test(text)) return 'zh-CN'
  if (japaneseRegex.test(text)) return 'ja-JP'
  if (koreanRegex.test(text)) return 'ko-KR'
  if (spanishIndicators.test(text)) return 'es-ES'
  if (frenchIndicators.test(text)) return 'fr-FR'

  const hindiRomanized = /\b(hai|hain|mein|mujhe|kya|nahi|bahut|acha|theek|kaise|kyun|lekin|abhi|yeh|woh|kuch|sab|kar|raha|rahi|ho|hoon|tum|aap|bhai|didi|accha)\b/i
  if (hindiRomanized.test(text)) return 'hi-IN'

  return 'en-US'
}

export function useVoiceInput() {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const recognitionRef = useRef(null)

  const startListening = useCallback((onResult, lang) => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Voice input is not supported in this browser. Try Chrome.')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = lang || ''

    recognition.onstart = () => setIsListening(true)
    recognition.onend = () => setIsListening(false)
    recognition.onerror = () => setIsListening(false)

    recognition.onresult = (event) => {
      let final = ''
      let interim = ''
      for (let i = 0; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript
        } else {
          interim += event.results[i][0].transcript
        }
      }
      const full = final + interim
      setTranscript(full)
      if (onResult) onResult(full)
    }

    recognitionRef.current = recognition
    recognition.start()
  }, [])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    setIsListening(false)
  }, [])

  return { isListening, transcript, startListening, stopListening }
}

export function speak(text, options = {}) {
  if (!('speechSynthesis' in window)) return
  if (!text || text.trim().length === 0) return

  window.speechSynthesis.cancel()

  const lang = detectLanguage(text)

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = options.rate || 0.92
  utterance.pitch = options.pitch || 1.05
  utterance.volume = options.volume || 0.85
  utterance.lang = lang

  const loadAndSpeak = () => {
    const voices = window.speechSynthesis.getVoices()
    if (voices.length === 0) return

    const langPrefix = lang.split('-')[0]

    let voice = voices.find(v => v.lang === lang && v.name.toLowerCase().includes('female'))
      || voices.find(v => v.lang === lang)
      || voices.find(v => v.lang.startsWith(langPrefix) && v.name.toLowerCase().includes('female'))
      || voices.find(v => v.lang.startsWith(langPrefix))

    if (!voice && lang === 'hi-IN') {
      voice = voices.find(v => v.lang.includes('hi') || v.name.toLowerCase().includes('hindi'))
    }

    if (!voice) {
      voice = voices.find(v => v.name.includes('Samantha') || v.name.includes('Google UK English Female'))
        || voices.find(v => v.lang === 'en-US' && v.name.toLowerCase().includes('female'))
        || voices.find(v => v.lang === 'en-US')
        || voices[0]
    }

    if (voice) utterance.voice = voice
    window.speechSynthesis.speak(utterance)
  }

  const voices = window.speechSynthesis.getVoices()
  if (voices.length > 0) {
    loadAndSpeak()
  } else {
    window.speechSynthesis.onvoiceschanged = loadAndSpeak
  }

  return utterance
}
