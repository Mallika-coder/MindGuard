import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle, Info, Heart } from 'lucide-react'

const severityConfig = {
  normal: { color: 'green', icon: CheckCircle, label: 'Healthy', bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700' },
  stress: { color: 'yellow', icon: Info, label: 'Mild Stress', bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700' },
  anxiety: { color: 'orange', icon: AlertTriangle, label: 'Anxiety Detected', bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700' },
  depression: { color: 'red', icon: Heart, label: 'Depression Indicators', bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700' },
  severe: { color: 'red', icon: AlertTriangle, label: 'Immediate Support Needed', bg: 'bg-red-100', border: 'border-red-300', text: 'text-red-800' },
}

export default function ResultsPanel({ result }) {
  if (!result) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 flex items-center justify-center min-h-[400px]">
        <div className="text-center text-gray-400">
          <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Results will appear here after screening</p>
        </div>
      </div>
    )
  }

  if (result.error) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-red-200 p-6">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <AlertTriangle className="w-5 h-5" />
          <h3 className="font-semibold">Error</h3>
        </div>
        <p className="text-sm text-red-700">{result.message}</p>
      </div>
    )
  }

  const { classification, response, disclaimer } = result
  const config = severityConfig[classification.label] || severityConfig.normal
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6"
    >
      {/* Classification Badge */}
      <div className={`p-4 rounded-xl ${config.bg} ${config.border} border mb-6`}>
        <div className="flex items-center gap-3">
          <Icon className={`w-6 h-6 ${config.text}`} />
          <div>
            <h3 className={`font-semibold ${config.text}`}>{config.label}</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Confidence: {(classification.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Severity Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Severity Score</span>
          <span>{(classification.severity_score * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${classification.severity_score * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className={`h-full rounded-full ${
              classification.severity_score >= 0.75 ? 'bg-red-500' :
              classification.severity_score >= 0.5 ? 'bg-orange-500' :
              classification.severity_score >= 0.25 ? 'bg-yellow-500' : 'bg-green-500'
            }`}
          />
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Detection Probabilities</h4>
        <div className="space-y-2">
          {Object.entries(classification.probabilities).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-gray-500 w-20 capitalize">{key}</span>
              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-400 rounded-full"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-10 text-right">
                {(value * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* AI Response */}
      <div className="p-4 bg-blue-50 rounded-xl border border-blue-100">
        <h4 className="text-sm font-medium text-blue-700 mb-2">AI Support Response</h4>
        <p className="text-sm text-blue-800 leading-relaxed">{response}</p>
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-gray-400 mt-4 italic">{disclaimer}</p>
    </motion.div>
  )
}

function Brain({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9.5 2A5.5 5.5 0 0 0 4 7.5c0 1.5.5 2.8 1.3 3.8A5.5 5.5 0 0 0 4 15c0 3 2.5 5.5 5.5 5.5h1V2h-1z" />
      <path d="M14.5 2A5.5 5.5 0 0 1 20 7.5c0 1.5-.5 2.8-1.3 3.8A5.5 5.5 0 0 1 20 15c0 3-2.5 5.5-5.5 5.5h-1V2h1z" />
    </svg>
  )
}
