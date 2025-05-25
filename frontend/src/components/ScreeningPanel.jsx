import { useState } from 'react'
import { Send, Loader2, Brain } from 'lucide-react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function ScreeningPanel({ onResult }) {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim() || text.trim().length < 10) return

    setLoading(true)
    try {
      const response = await axios.post(`${API_URL}/screen`, { text })
      onResult(response.data)
    } catch (error) {
      onResult({
        error: true,
        message: error.response?.data?.detail || 'Failed to connect to the screening service.',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
          <Brain className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Mental Health Screening</h2>
          <p className="text-sm text-gray-500">Share how you're feeling for an AI assessment</p>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write about how you've been feeling lately... (minimum 10 characters)"
          rows={8}
          className="w-full p-4 border border-gray-200 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 placeholder-gray-400"
        />

        <div className="flex items-center justify-between mt-4">
          <span className="text-xs text-gray-400">
            {text.length} characters
          </span>
          <button
            type="submit"
            disabled={loading || text.trim().length < 10}
            className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Screen
              </>
            )}
          </button>
        </div>
      </form>

      <div className="mt-6 p-3 bg-gray-50 rounded-lg">
        <p className="text-xs text-gray-500 leading-relaxed">
          <strong>How it works:</strong> Our fine-tuned BERT model analyzes your text to detect
          patterns associated with mental health conditions. The RAG pipeline then retrieves
          relevant support information from our knowledge base.
        </p>
      </div>
    </div>
  )
}
