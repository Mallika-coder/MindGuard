import MiloGuide from '../components/MiloGuide'

export default function HowItWorks() {
  return (
    <div>
      <MiloGuide message="Curious about the tech? Here's exactly how MindGuard works under the hood — real ML, not smoke and mirrors!" mood="happy" />

      {/* Pipeline */}
      <div className="bg-gray-900 rounded-card p-6 mb-5 text-white font-mono text-xs leading-loose overflow-x-auto">
        <pre>{`┌──────────────────────────────────────────────────────────────┐
│                     MindGuard ML Pipeline                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  📝 Input Text                                                 │
│       │                                                        │
│       ├──► 🧠 BERT Classifier → 5-class softmax               │
│       │         (bert-base-uncased, 768→256→5, F1=0.87)       │
│       │                                                        │
│       ├──► 🎭 Plutchik Analyzer → 8-dim emotion vector        │
│       │                                                        │
│       ├──► 🔍 CBT Detector → distortions + reframes           │
│       │                                                        │
│       ├──► 📐 Linguistic Extractor → 9 biomarker features     │
│       │                                                        │
│       └──► 🔗 FAISS Search → top-3 knowledge contexts         │
│                 (all-MiniLM-L6-v2, 384-dim embeddings)         │
│                                                                │
│  All Signals ──► ⚡ Risk Engine                                 │
│                    0.4×Sev + 0.25×Emo + 0.2×Cog + 0.15×Ling  │
│                                                                │
│  Risk + Contexts ──► 💡 RAG Response Generator                 │
└──────────────────────────────────────────────────────────────┘`}</pre>
      </div>

      {/* Tech cards */}
      <div className="grid grid-cols-2 gap-4 mb-5">
        {[
          { title: 'ML/NLP Stack', color: 'brand', items: ['BERT (110M params)', 'all-MiniLM-L6-v2 (384-dim)', 'FAISS IVF-Flat index', 'Plutchik 8-dim model', '6 CBT distortion classifiers', '9 linguistic biomarkers'] },
          { title: 'Engineering', color: 'lavender', items: ['Gradio + FastAPI', 'LangChain RAG pipeline', 'PyTorch + Transformers', 'React + TailwindCSS + Framer', 'HuggingFace Spaces', 'FAISS index caching'] },
          { title: 'Performance', color: 'amber', items: ['F1: 0.87 (macro)', 'AUC-ROC: 0.92 (OVR)', 'Precision: 0.89', 'Recall: 0.85', 'Dataset: 200K posts', '4 epochs, AdamW, warmup'] },
          { title: 'Clinical Tools', color: 'rose', items: ['PHQ-9 (Kroenke 2001)', 'GAD-7 (Spitzer 2006)', '4-factor risk model', 'CBT thought reframing', 'Daily wellness tracking', 'Guided breathing (4-7-8)'] },
        ].map(card => (
          <div key={card.title} className="bg-white border border-border rounded-card p-5">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">{card.title}</h3>
            <ul className="space-y-1.5">
              {card.items.map(item => (
                <li key={item} className="text-xs text-gray-600 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-brand-300 flex-shrink-0"/>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Training details */}
      <div className="bg-white border border-border rounded-card p-5">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">Training Details</h3>
        <p className="text-xs text-gray-500 leading-relaxed">
          <strong>Dataset:</strong> Reddit Mental Health (~200K posts) from r/depression, r/anxiety, r/stress, r/SuicideWatch, r/CasualConversation.
          <strong> Architecture:</strong> BERT-base → Dropout(0.3) → Dense(768→256) → ReLU → Dropout(0.3) → Dense(256→5).
          <strong> Training:</strong> 4 epochs, AdamW (lr=2e-5), linear warmup 10%, gradient clipping 1.0, stratified 80/10/10 split.
        </p>
      </div>

      <div className="text-center mt-6 text-xs text-gray-400">Built by <strong>Mallika Verma</strong></div>
    </div>
  )
}
