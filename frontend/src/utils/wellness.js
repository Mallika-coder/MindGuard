export const EMOTION_LEXICON = {
  anger: ["angry", "furious", "rage", "hate", "irritated", "frustrated", "annoyed", "mad", "pissed", "livid"],
  sadness: ["sad", "crying", "tears", "grief", "lonely", "heartbroken", "miserable", "depressed", "down", "unhappy", "blue"],
  fear: ["afraid", "scared", "terrified", "panic", "nervous", "anxious", "worried", "tense", "dread", "apprehensive", "stuck"],
  joy: ["happy", "excited", "grateful", "proud", "content", "cheerful", "thrilled", "optimistic", "good", "great", "wonderful", "love", "enjoy"],
  disgust: ["disgusted", "repulsed", "revolted", "sick", "nauseated", "gross", "awful"],
  surprise: ["shocked", "amazed", "astonished", "startled", "unexpected", "stunned"],
  trust: ["believe", "faith", "confident", "secure", "reliable", "honest", "trust", "safe", "comfortable"],
  anticipation: ["expecting", "hopeful", "looking forward", "planning", "eager", "curious", "interested", "future"],
}

export const SEVERE_KEYWORDS = ["suicide", "kill myself", "end it all", "no reason to live", "self-harm", "want to die"]
export const DEPRESSION_KEYWORDS = ["hopeless", "worthless", "empty", "numb", "can't go on", "no energy", "don't care anymore", "failure", "burden", "pointless"]
export const ANXIETY_KEYWORDS = ["panic", "worried", "racing thoughts", "can't breathe", "nervous", "terrified", "dread", "restless", "on edge", "anxious", "fear", "scared"]
export const STRESS_KEYWORDS = ["overwhelmed", "pressure", "exhausted", "too much", "burned out", "can't keep up", "drowning", "stressed", "overloaded"]

export function classifyText(text) {
  const lower = text.toLowerCase()
  const scores = { normal: 0.1, stress: 0, anxiety: 0, depression: 0, severe: 0 }
  SEVERE_KEYWORDS.forEach(kw => { if (lower.includes(kw)) scores.severe += 0.35 })
  DEPRESSION_KEYWORDS.forEach(kw => { if (lower.includes(kw)) scores.depression += 0.2 })
  ANXIETY_KEYWORDS.forEach(kw => { if (lower.includes(kw)) scores.anxiety += 0.2 })
  STRESS_KEYWORDS.forEach(kw => { if (lower.includes(kw)) scores.stress += 0.2 })
  const total = Object.values(scores).reduce((a, b) => a + b, 0)
  const probabilities = Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, +(v / total).toFixed(4)]))
  const label = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0]
  const confidence = probabilities[label]
  const severityMap = { normal: 0, stress: 0.25, anxiety: 0.5, depression: 0.75, severe: 1.0 }
  return { label, confidence, severity_score: severityMap[label], probabilities }
}

export function analyzeEmotions(text) {
  const lower = text.toLowerCase()
  const words = lower.split(/\s+/)
  const scores = {}
  for (const [emotion, keywords] of Object.entries(EMOTION_LEXICON)) {
    let score = 0
    keywords.forEach(kw => { if (lower.includes(kw)) score += 1 })
    scores[emotion] = Math.min(score / 4, 1)
  }
  const total = Object.values(scores).reduce((a, b) => a + b, 0)
  if (total > 0) {
    for (const k of Object.keys(scores)) scores[k] = +(scores[k] / total).toFixed(3)
  } else {
    for (const k of Object.keys(scores)) scores[k] = +(1 / 8).toFixed(3)
  }
  return scores
}

export const COGNITIVE_DISTORTIONS = [
  { id: "all_or_nothing", patterns: ["always", "never", "everyone", "nobody", "everything", "nothing", "completely", "totally"], label: "All-or-Nothing Thinking", description: "Seeing things in black-and-white categories", reframe: "Try: 'sometimes', 'in this situation', 'this specific instance'" },
  { id: "catastrophizing", patterns: ["worst", "terrible", "horrible", "disaster", "ruined", "destroyed"], label: "Catastrophizing", description: "Expecting the worst-case scenario", reframe: "Ask: What's the most likely outcome? What would I tell a friend?" },
  { id: "mind_reading", patterns: ["they think", "everyone thinks", "people think", "judges me"], label: "Mind Reading", description: "Assuming you know what others think", reframe: "Remind yourself: I cannot read minds. What evidence do I have?" },
  { id: "should_statements", patterns: ["i should", "i must", "i have to", "ought to", "supposed to"], label: "Should Statements", description: "Rigid demands on yourself", reframe: "Replace 'should' with 'I would prefer' — give yourself permission to be human" },
  { id: "overgeneralization", patterns: ["always happens", "every time", "never works", "nothing ever"], label: "Overgeneralization", description: "Broad conclusions from single events", reframe: "Is this really ALWAYS? Can you think of even one exception?" },
  { id: "personalization", patterns: ["my fault", "because of me", "i caused", "blame myself"], label: "Personalization", description: "Taking excessive responsibility", reframe: "What percentage was actually within your control?" },
]

export function detectDistortions(text) {
  const lower = text.toLowerCase()
  return COGNITIVE_DISTORTIONS.filter(d => d.patterns.some(p => lower.includes(p)))
    .map(d => ({ ...d, trigger: d.patterns.find(p => lower.includes(p)) }))
}

export function computeLinguistics(text) {
  const words = text.split(/\s+/)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim())
  const firstPerson = words.filter(w => ["i", "me", "my", "myself", "mine", "i'm", "i've"].includes(w.toLowerCase())).length
  const negations = words.filter(w => ["not", "no", "never", "don't", "can't", "won't", "nothing", "nobody"].includes(w.toLowerCase())).length
  return {
    wordCount: words.length,
    sentenceCount: sentences.length,
    avgSentenceLength: +(words.length / Math.max(sentences.length, 1)).toFixed(1),
    firstPersonRatio: +(firstPerson / Math.max(words.length, 1)).toFixed(3),
    negationRatio: +(negations / Math.max(words.length, 1)).toFixed(3),
    lexicalDiversity: +(new Set(words.map(w => w.toLowerCase())).size / Math.max(words.length, 1)).toFixed(3),
  }
}

export function computeRisk(classification, emotions, distortions, linguistics) {
  const base = classification.severity_score
  const emotionRisk = (emotions.sadness || 0) * 0.3 + (emotions.fear || 0) * 0.25 - (emotions.joy || 0) * 0.2
  const distortionRisk = Math.min(distortions.length * 0.12, 0.35)
  const lingRisk = (linguistics.negationRatio || 0) * 0.4 + (linguistics.firstPersonRatio || 0) * 0.2
  const composite = Math.max(0, Math.min(1, base * 0.4 + emotionRisk * 0.25 + distortionRisk * 0.2 + lingRisk * 0.15))
  const level = composite >= 0.8 ? 'Critical' : composite >= 0.6 ? 'High' : composite >= 0.4 ? 'Moderate' : composite >= 0.2 ? 'Low' : 'Minimal'
  return { composite: +composite.toFixed(3), level, components: { keyword: +base.toFixed(3), emotion: +Math.max(emotionRisk, 0).toFixed(3), cognition: +distortionRisk.toFixed(3), linguistic: +Math.min(lingRisk, 1).toFixed(3) } }
}

export const PHQ9_QUESTIONS = [
  "Little interest or pleasure in doing things",
  "Feeling down, depressed, or hopeless",
  "Trouble falling or staying asleep, or sleeping too much",
  "Feeling tired or having little energy",
  "Poor appetite or overeating",
  "Feeling bad about yourself — or that you are a failure",
  "Trouble concentrating on things",
  "Moving or speaking slowly, or being fidgety/restless",
  "Thoughts that you would be better off dead, or of hurting yourself",
]

export const GAD7_QUESTIONS = [
  "Feeling nervous, anxious, or on edge",
  "Not being able to stop or control worrying",
  "Worrying too much about different things",
  "Trouble relaxing",
  "Being so restless that it's hard to sit still",
  "Becoming easily annoyed or irritable",
  "Feeling afraid, as if something awful might happen",
]

export const EMOTION_CHALLENGES = [
  { text: "I can't believe they said that to me. I'm so furious I could scream.", answer: "anger", explanation: "Strong aggression language indicates anger." },
  { text: "What if I fail? What if everyone laughs? I keep thinking about everything that could go wrong.", answer: "fear", explanation: "Future-oriented worry and catastrophizing signal fear/anxiety." },
  { text: "Nothing matters anymore. I feel empty inside. Everything is grey.", answer: "sadness", explanation: "Anhedonia and emptiness indicate deep sadness." },
  { text: "I got the promotion! Everything is falling into place!", answer: "joy", explanation: "Achievement and positive self-concept indicate joy." },
  { text: "I didn't expect that at all. Completely out of nowhere.", answer: "surprise", explanation: "Unexpectedness signals surprise." },
  { text: "I trust her completely. She's always been there for me.", answer: "trust", explanation: "Reliability and safety indicate trust." },
  { text: "I'm so excited for tomorrow! I've been planning this for weeks.", answer: "anticipation", explanation: "Forward-looking excitement indicates anticipation." },
  { text: "That's absolutely revolting. The whole situation makes my stomach turn.", answer: "disgust", explanation: "Physical revulsion signals disgust." },
]
