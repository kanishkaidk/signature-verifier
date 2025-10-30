import { useState } from 'react'

export default function App() {
  const [img1, setImg1] = useState(null)
  const [img2, setImg2] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function verify() {
    if (!img1 || !img2) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('img1', img1)
      formData.append('img2', img2)
      const resp = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      })
      if (!resp.ok) throw new Error(`Server ${resp.status}`)
      const data = await resp.json()
      setResult(data)
    } catch (e) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 720, margin: '0 auto' }}>
      <h1>SignGuard — Signature Verification</h1>
      <p>Upload two images and check similarity.</p>

      <div style={{ display: 'grid', gap: 12, marginTop: 12 }}>
        <input type="file" accept="image/*" onChange={(e) => setImg1(e.target.files?.[0] ?? null)} />
        <input type="file" accept="image/*" onChange={(e) => setImg2(e.target.files?.[0] ?? null)} />
        <button onClick={verify} disabled={loading || !img1 || !img2}>
          {loading ? 'Verifying…' : 'Verify'}
        </button>
      </div>

      {error && <p style={{ color: 'crimson' }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: 16, padding: 12, border: '1px solid #ddd', borderRadius: 8 }}>
          <div>Similarity: {result.similarity_score}</div>
          <div>Verdict: {result.verdict}</div>
        </div>
      )}
    </div>
  )
}
