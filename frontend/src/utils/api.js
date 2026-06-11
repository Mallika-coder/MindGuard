import axios from 'axios'

const HF_API = 'https://mallikav-mindguard.hf.space'

export async function analyzeWithML(text) {
  try {
    const response = await axios.post(`${HF_API}/api/predict`, {
      data: [text],
    }, { timeout: 30000 })
    return response.data
  } catch (error) {
    return null
  }
}

export async function callGradioAPI(fnIndex, data) {
  try {
    const response = await axios.post(`${HF_API}/run/predict`, {
      data: data,
      fn_index: fnIndex,
    }, { timeout: 30000 })
    return response.data?.data?.[0] || null
  } catch {
    return null
  }
}
