import { useMemo, useRef, useState } from 'react'

const API = 'http://127.0.0.1:5000'

export default function App() {
  const [tab, setTab] = useState('verify') // verify | batch | about

  // Verify tab state
  const [img1, setImg1] = useState(null)
  const [img2, setImg2] = useState(null)
  const [img1Url, setImg1Url] = useState('')
  const [img2Url, setImg2Url] = useState('')
  const [overlay, setOverlay] = useState(50)
  const [zoom, setZoom] = useState(100)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [heatmapUrl, setHeatmapUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [verifiedAt, setVerifiedAt] = useState('')

  // Batch tab state
  const [refFile, setRefFile] = useState(null)
  const [batchFiles, setBatchFiles] = useState([])
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchResult, setBatchResult] = useState(null)
  const [batchError, setBatchError] = useState(null)
  const [metrics, setMetrics] = useState(null)

  const ringColor = useMemo(() => {
    const score = result?.similarity_score ?? 0
    return score >= 0.85 ? '#16a34a' : '#dc2626'
  }, [result])

  // Load metrics on mount
  React.useEffect(() => {
    fetch(`${API}/metrics`).then(r=>r.json()).then(setMetrics).catch(()=>{})
  }, [])

  function onPick(setter, urlSetter) {
    return (e) => {
      const f = e.target.files?.[0] ?? null
      setter(f)
      urlSetter?.(f ? URL.createObjectURL(f) : '')
    }
  }

  function onDrop(setter, urlSetter) {
    return (e) => {
      e.preventDefault()
      const f = e.dataTransfer.files?.[0]
      if (f) {
        setter(f)
        urlSetter?.(URL.createObjectURL(f))
      }
    }
  }

  async function generateHeatmap() {
    if (!img1Url || !img2Url) return
    // Load both images and draw to same-sized canvas, compute abs diff as red heatmap
    const load = (src) => new Promise((resolve, reject) => {
      const im = new Image()
      im.crossOrigin = 'anonymous'
      im.onload = () => resolve(im)
      im.onerror = reject
      im.src = src
    })
    try {
      const [im1, im2] = await Promise.all([load(img1Url), load(img2Url)])
      const W = Math.min(im1.naturalWidth, im2.naturalWidth)
      const H = Math.min(im1.naturalHeight, im2.naturalHeight)
      const canvas = document.createElement('canvas')
      canvas.width = W
      canvas.height = H
      const ctx = canvas.getContext('2d')
      const c2 = document.createElement('canvas')
      c2.width = W
      c2.height = H
      const ctx2 = c2.getContext('2d')
      ctx.drawImage(im1, 0, 0, W, H)
      ctx2.drawImage(im2, 0, 0, W, H)
      const d1 = ctx.getImageData(0, 0, W, H)
      const d2 = ctx2.getImageData(0, 0, W, H)
      const out = ctx.createImageData(W, H)
      for (let i = 0; i < d1.data.length; i += 4) {
        const r1 = d1.data[i], g1 = d1.data[i+1], b1 = d1.data[i+2]
        const r2 = d2.data[i], g2 = d2.data[i+1], b2 = d2.data[i+2]
        const y1 = 0.299*r1 + 0.587*g1 + 0.114*b1
        const y2 = 0.299*r2 + 0.587*g2 + 0.114*b2
        const diff = Math.min(255, Math.abs(y1 - y2))
        // Map diff to red intensity; higher diff = stronger red
        out.data[i] = 255
        out.data[i+1] = 0
        out.data[i+2] = 0
        out.data[i+3] = diff // alpha
      }
      ctx.putImageData(out, 0, 0)
      setHeatmapUrl(canvas.toDataURL('image/png'))
      setShowHeatmap(true)
    } catch (e) {
      alert('Failed to generate heatmap')
    }
  }

  async function serverHeatmap() {
    if (!img1 || !img2) return
    try {
      const formData = new FormData()
      formData.append('img1', img1)
      formData.append('img2', img2)
      const resp = await fetch(`${API}/saliency`, { method: 'POST', body: formData })
      if (!resp.ok) throw new Error('Saliency failed')
      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      setHeatmapUrl(url)
      setShowHeatmap(true)
    } catch (e) {
      alert(e.message || 'Saliency failed')
    }
  }

  async function verify() {
    if (!img1 || !img2) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('img1', img1)
      formData.append('img2', img2)
      const resp = await fetch(`${API}/predict`, { method: 'POST', body: formData })
      if (!resp.ok) throw new Error(`Server ${resp.status}`)
      const data = await resp.json()
      setResult(data)
      setVerifiedAt(new Date().toLocaleString())
    } catch (e) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function downloadReport() {
    if (!img1 || !img2) return
    try {
      const formData = new FormData()
      formData.append('img1', img1)
      formData.append('img2', img2)
      if (showHeatmap && heatmapUrl) {
        // convert dataURL to Blob
        const res = await fetch(heatmapUrl)
        const blob = await res.blob()
        formData.append('heatmap', blob, 'heatmap.png')
      }
      const resp = await fetch(`${API}/report`, { method: 'POST', body: formData })
      if (!resp.ok) throw new Error('Report failed')
      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'signguard_report.pdf'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert(e.message || 'Report failed')
    }
  }

  async function runBatch() {
    if (!refFile || batchFiles.length === 0) return
    setBatchLoading(true)
    setBatchError(null)
    setBatchResult(null)
    try {
      const formData = new FormData()
      formData.append('reference', refFile)
      for (const f of batchFiles) formData.append('files', f)
      const resp = await fetch(`${API}/batch_predict`, { method: 'POST', body: formData })
      if (!resp.ok) throw new Error(`Server ${resp.status}`)
      const data = await resp.json()
      setBatchResult(data)
    } catch (e) {
      setBatchError(e.message || 'Batch failed')
    } finally {
      setBatchLoading(false)
    }
  }

  const scorePercent = useMemo(() => {
    if (!result) return 0
    const v = Math.max(0, Math.min(1, result.similarity_score))
    return Math.round(v * 100)
  }, [result])

  return (
    <div className="app-root">
      <header className="app-header navbar">
        <div className="brand">
          <div className="logo-circle">SG</div>
          <div>
            <h1>SignGuard</h1>
            <p className="tagline">Authenticate signatures instantly. Prevent fraud effortlessly.</p>
          </div>
        </div>
        <nav className="tabs">
          <button className={tab === 'verify' ? 'tab active' : 'tab'} onClick={() => setTab('verify')}>Verify</button>
          <button className={tab === 'batch' ? 'tab active' : 'tab'} onClick={() => setTab('batch')}>Batch</button>
          <button className={tab === 'about' ? 'tab active' : 'tab'} onClick={() => setTab('about')}>About</button>
        </nav>
      </header>

      <section className="hero">
        <h2>AI-Powered Signature Verification</h2>
        <p>Detect forgeries with deep learning precision.</p>
        <div className="hero-actions">
          <button onClick={() => { setTab('verify'); document.querySelector('.panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }}>Try Demo Now</button>
        </div>
      </section>

      {tab === 'verify' && (
        <section className="panel">
          <p>Upload two images (drag & drop or use pickers), then Verify.</p>

          <div className="pickers">
            <div className="dropzone" onDragOver={(e)=>e.preventDefault()} onDrop={onDrop(setImg1, setImg1Url)}>
              <p>Drop Image 1 here</p>
              <input type="file" accept="image/*" onChange={onPick(setImg1, setImg1Url)} />
            </div>
            <div className="dropzone" onDragOver={(e)=>e.preventDefault()} onDrop={onDrop(setImg2, setImg2Url)}>
              <p>Drop Image 2 here</p>
              <input type="file" accept="image/*" onChange={onPick(setImg2, setImg2Url)} />
            </div>
          </div>

          <div className="preview">
            <div className="side">
              {img1Url && <div className="zoom-wrap" style={{ ['--z']: `${zoom/100}` }}>
                <img src={img1Url} alt="img1" style={{ transform: `scale(${zoom/100})` }} />
              </div>}
            </div>
            <div className="side overlay">
              {img1Url && <div className="zoom-wrap" style={{ ['--z']: `${zoom/100}` }}>
                <img src={img1Url} alt="base" style={{ transform: `scale(${zoom/100})` }} />
              </div>}
              {showHeatmap && heatmapUrl && (
                <img src={heatmapUrl} alt="heatmap" style={{ opacity: overlay/100 }} />
              )}
              {!showHeatmap && img2Url && (
                <img src={img2Url} alt="overlay" style={{ opacity: overlay/100 }} />
              )}
            </div>
            <div className="side">
              {img2Url && <div className="zoom-wrap" style={{ ['--z']: `${zoom/100}` }}>
                <img src={img2Url} alt="img2" style={{ transform: `scale(${zoom/100})` }} />
              </div>}
            </div>
          </div>

          <div className="slider">
            <label>Overlay transparency: {overlay}%</label>
            <input type="range" min="0" max="100" value={overlay} onChange={(e)=>setOverlay(parseInt(e.target.value)||0)} />
            <label>Zoom: {zoom}%</label>
            <input type="range" min="50" max="200" value={zoom} onChange={(e)=>setZoom(parseInt(e.target.value)||100)} />
          </div>

          <div className="actions">
            <button onClick={verify} disabled={loading || !img1 || !img2}>{loading ? 'Verifying…' : 'Verify'}</button>
            <button onClick={downloadReport} disabled={!img1 || !img2}>Download Report</button>
            <button onClick={generateHeatmap} disabled={!img1Url || !img2Url}>Generate Heatmap</button>
            <button onClick={serverHeatmap} disabled={!img1 || !img2}>Server Heatmap</button>
            <button onClick={()=>setShowHeatmap((v)=>!v)} disabled={!heatmapUrl}>{showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}</button>
          </div>

          {loading && (
            <div className="progress"><div className="bar" /></div>
          )}

          {error && <p className="error">Error: {error}</p>}

          {result && (
            <div className="result">
              <div className="ring" style={{
                background: `conic-gradient(${ringColor} ${scorePercent}%, #e5e7eb 0)`
              }}>
                <div className="ring-inner">{scorePercent}%</div>
              </div>
              <div className="verdict" style={{ color: ringColor }}>{result.verdict}</div>
              {verifiedAt && <div className="tagline">Verified at: {verifiedAt}</div>}
            </div>
          )}

          <div className="cards">
            <div className="card">
              <div className="card-title">Confidence</div>
              <div className="card-value">{scorePercent ? `${scorePercent}%` : '--'}</div>
              <div className="card-sub">Derived from similarity vs threshold</div>
            </div>
            <div className="card">
              <div className="card-title">Model Accuracy</div>
              <div className="card-value">{metrics?.accuracy ? `${Math.round(metrics.accuracy*100)}%` : '--'}</div>
              <div className="card-sub">Threshold: {metrics?.threshold ?? 0.85}</div>
            </div>
          </div>
        </section>
      )}

      {tab === 'batch' && (
        <section className="panel">
          <p>Pick one reference signature and multiple files to compare against it.</p>
          <div className="pickers">
            <div className="dropzone">
              <p>Reference</p>
              <input type="file" accept="image/*" onChange={(e)=>setRefFile(e.target.files?.[0] ?? null)} />
            </div>
            <div className="dropzone">
              <p>Files</p>
              <input multiple type="file" accept="image/*" onChange={(e)=>setBatchFiles(Array.from(e.target.files || []))} />
            </div>
          </div>
          <div className="actions">
            <button onClick={runBatch} disabled={batchLoading || !refFile || batchFiles.length===0}>{batchLoading ? 'Running…' : 'Run Batch'}</button>
          </div>
          {batchError && <p className="error">Error: {batchError}</p>}
          {batchLoading && (<div className="progress"><div className="bar" /></div>)}
          {batchResult && (
            <div className="table-wrap">
              <table className="results">
                <thead><tr><th>File</th><th>Similarity</th><th>Verdict</th></tr></thead>
                <tbody>
                  {batchResult.results?.map((r, i)=> (
                    <tr key={i}>
                      <td>{r.filename || 'file'}</td>
                      <td>{(Math.max(0, Math.min(1, r.similarity_score||0))*100).toFixed(1)}%</td>
                      <td>{r.verdict || r.error}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {tab === 'about' && (
        <section className="panel">
          <h2>About SignGuard</h2>
          <p>AI-powered signature verification using a Siamese network (ResNet backbone). Compare two signatures, run batch checks, and generate PDF reports.</p>
          <ul>
            <li>Similarity score with visual ring and verdict</li>
            <li>Overlay and zoom tools for visual inspection</li>
            <li>Optional client-side heatmap overlay</li>
            <li>Batch mode with ranked results</li>
            <li>PDF report with previews and timestamp</li>
          </ul>
        </section>
      )}

      <footer className="footer">
        <span>Built by Kanishka | IGDTUW</span>
        <a href="https://github.com/kanishka/signature-verifier" target="_blank" rel="noreferrer">View on GitHub</a>
      </footer>
    </div>
  )
}
