import { useState, useRef, useCallback } from 'react'

export function useVoiceInput() {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const recognitionRef = useRef(null)

  const startListening = useCallback((onResult) => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Voice input is not supported in this browser. Try Chrome.')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

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

  window.speechSynthesis.cancel()

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = options.rate || 0.9
  utterance.pitch = options.pitch || 1.0
  utterance.volume = options.volume || 0.9

  const voices = window.speechSynthesis.getVoices()
  const preferred = voices.find(v =>
    v.name.includes('Samantha') || v.name.includes('Karen') ||
    v.name.includes('Female') || v.name.includes('Google UK English Female')
  ) || voices.find(v => v.lang === 'en-US') || voices[0]

  if (preferred) utterance.voice = preferred
  window.speechSynthesis.speak(utterance)
  return utterance
}
