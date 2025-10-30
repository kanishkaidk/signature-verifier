import { useState, useEffect, useRef } from 'react';
import ScoreRing from './components/ScoreRing';
import AnalyticsDashboard from './components/AnalyticsDashboard';

const API = 'http://127.0.0.1:5000';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [tab, setTab] = useState('verify');
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);

  const [img1, setImg1] = useState(null);
  const [img2, setImg2] = useState(null);
  const [img1Url, setImg1Url] = useState('');
  const [img2Url, setImg2Url] = useState('');
  const [overlay, setOverlay] = useState(50);
  const [zoom, setZoom] = useState(100);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapUrl, setHeatmapUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [verifiedAt, setVerifiedAt] = useState('');

  const [refFile, setRefFile] = useState(null);
  const [refFileUrl, setRefFileUrl] = useState('');
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [batchError, setBatchError] = useState(null);

  const [dragOver1, setDragOver1] = useState(false);
  const [dragOver2, setDragOver2] = useState(false);
  const [dragOverRef, setDragOverRef] = useState(false);
  const [dragOverBatch, setDragOverBatch] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setDarkMode(true);
      document.body.classList.add('dark-mode');
    }

    fetch(`${API}/metrics`).then(r => r.json()).then(setMetrics).catch(() => {});
    fetch(`${API}/history`).then(r => r.json()).then(data => setHistory(data.history || [])).catch(() => {});
  }, []);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', !darkMode ? 'dark' : 'light');
  };

  const handleFilePick = (setter, urlSetter) => (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setter(f);
      urlSetter(URL.createObjectURL(f));
    }
  };

  const handleDrop = (setter, urlSetter, dragSetter) => (e) => {
    e.preventDefault();
    dragSetter(false);
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setter(f);
      urlSetter(URL.createObjectURL(f));
    }
  };

  const handleBatchFilesPick = (e) => {
    const files = Array.from(e.target.files || []);
    setBatchFiles(files);
  };

  const handleBatchFilesDrop = (e) => {
    e.preventDefault();
    setDragOverBatch(false);
    const files = Array.from(e.dataTransfer.files || []);
    setBatchFiles(files);
  };

  async function verify() {
    if (!img1 || !img2) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('img1', img1);
      formData.append('img2', img2);
      const resp = await fetch(`${API}/predict`, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`Server ${resp.status}`);
      const data = await resp.json();
      setResult(data);
      setVerifiedAt(new Date().toLocaleString());

      const histResp = await fetch(`${API}/history`);
      if (histResp.ok) {
        const histData = await histResp.json();
        setHistory(histData.history || []);
      }
    } catch (e) {
      setError(e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  async function generateServerHeatmap() {
    if (!img1 || !img2) return;
    try {
      const formData = new FormData();
      formData.append('img1', img1);
      formData.append('img2', img2);
      const resp = await fetch(`${API}/saliency`, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Saliency failed');
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      setHeatmapUrl(url);
      setShowHeatmap(true);
    } catch (e) {
      alert(e.message || 'Heatmap generation failed');
    }
  }

  async function downloadReport() {
    if (!img1 || !img2) return;
    try {
      const formData = new FormData();
      formData.append('img1', img1);
      formData.append('img2', img2);
      if (showHeatmap && heatmapUrl) {
        const res = await fetch(heatmapUrl);
        const blob = await res.blob();
        formData.append('heatmap', blob, 'heatmap.png');
      }
      const resp = await fetch(`${API}/report`, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Report failed');
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `signguard_report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(e.message || 'Report download failed');
    }
  }

  async function runBatch() {
    if (!refFile || batchFiles.length === 0) return;
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);
    try {
      const formData = new FormData();
      formData.append('reference', refFile);
      for (const f of batchFiles) formData.append('files', f);
      const resp = await fetch(`${API}/batch_predict`, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`Server ${resp.status}`);
      const data = await resp.json();
      setBatchResult(data);

      const histResp = await fetch(`${API}/history`);
      if (histResp.ok) {
        const histData = await histResp.json();
        setHistory(histData.history || []);
      }
    } catch (e) {
      setBatchError(e.message || 'Batch failed');
    } finally {
      setBatchLoading(false);
    }
  }

  const scrollToPanel = () => {
    setTimeout(() => {
      document.querySelector('.panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="navbar">
          <div className="brand">
            <div className="logo-circle">SG</div>
            <div className="brand-text">
              <h1>SignGuard</h1>
              <p className="tagline">Authenticate signatures instantly. Prevent fraud effortlessly.</p>
            </div>
          </div>
          <div className="nav-right">
            <nav className="tabs">
              <button className={tab === 'verify' ? 'tab active' : 'tab'} onClick={() => setTab('verify')}>
                Verify
              </button>
              <button className={tab === 'batch' ? 'tab active' : 'tab'} onClick={() => setTab('batch')}>
                Batch
              </button>
              <button className={tab === 'analytics' ? 'tab active' : 'tab'} onClick={() => setTab('analytics')}>
                Analytics
              </button>
              <button className={tab === 'history' ? 'tab active' : 'tab'} onClick={() => setTab('history')}>
                History
              </button>
              <button className={tab === 'about' ? 'tab active' : 'tab'} onClick={() => setTab('about')}>
                About
              </button>
            </nav>
            <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </header>

      <section className="hero">
        <div className="hero-content">
          <h2>AI-Powered Signature Verification</h2>
          <p>Detect forgeries with deep learning precision using advanced Siamese networks</p>
          <div className="hero-actions">
            <button onClick={() => { setTab('verify'); scrollToPanel(); }}>
              Try Demo Now →
            </button>
            <button className="secondary" onClick={() => setTab('about')}>
              Learn More
            </button>
          </div>
        </div>
      </section>

      {tab === 'verify' && (
        <section className="panel fade-in">
          <h3>Signature Verification</h3>
          <p>Upload two signature images to compare and verify authenticity</p>

          <div className="pickers">
            <div
              className={`dropzone ${dragOver1 ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver1(true); }}
              onDragLeave={() => setDragOver1(false)}
              onDrop={handleDrop(setImg1, setImg1Url, setDragOver1)}
            >
              <div className="dropzone-icon">📄</div>
              <p>Drop Signature 1 here</p>
              <input type="file" accept="image/*" onChange={handleFilePick(setImg1, setImg1Url)} />
              {img1 && <small style={{ color: 'var(--success-600)' }}>✓ {img1.name}</small>}
            </div>

            <div
              className={`dropzone ${dragOver2 ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver2(true); }}
              onDragLeave={() => setDragOver2(false)}
              onDrop={handleDrop(setImg2, setImg2Url, setDragOver2)}
            >
              <div className="dropzone-icon">📄</div>
              <p>Drop Signature 2 here</p>
              <input type="file" accept="image/*" onChange={handleFilePick(setImg2, setImg2Url)} />
              {img2 && <small style={{ color: 'var(--success-600)' }}>✓ {img2.name}</small>}
            </div>
          </div>

          {(img1Url || img2Url) && (
            <div className="preview fade-in">
              <div className="preview-box">
                {img1Url && <img src={img1Url} alt="Signature 1" style={{ transform: `scale(${zoom / 100})` }} />}
              </div>
              <div className="preview-box overlay-container">
                {img1Url && <img src={img1Url} alt="Base" style={{ transform: `scale(${zoom / 100})` }} />}
                {showHeatmap && heatmapUrl && (
                  <img src={heatmapUrl} alt="Heatmap" style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }} />
                )}
                {!showHeatmap && img2Url && (
                  <img src={img2Url} alt="Overlay" style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }} />
                )}
              </div>
              <div className="preview-box">
                {img2Url && <img src={img2Url} alt="Signature 2" style={{ transform: `scale(${zoom / 100})` }} />}
              </div>
            </div>
          )}

          {(img1Url || img2Url) && (
            <div className="controls fade-in">
              <div className="control-group">
                <label>
                  <span>Overlay Opacity</span>
                  <span>{overlay}%</span>
                </label>
                <input type="range" min="0" max="100" value={overlay} onChange={(e) => setOverlay(parseInt(e.target.value) || 0)} />
              </div>
              <div className="control-group">
                <label>
                  <span>Zoom</span>
                  <span>{zoom}%</span>
                </label>
                <input type="range" min="50" max="200" value={zoom} onChange={(e) => setZoom(parseInt(e.target.value) || 100)} />
              </div>
            </div>
          )}

          <div className="actions">
            <button onClick={verify} disabled={loading || !img1 || !img2} className="success">
              {loading ? '🔄 Verifying...' : '✓ Verify Signatures'}
            </button>
            <button onClick={generateServerHeatmap} disabled={!img1 || !img2} className="secondary">
              🔥 Generate Heatmap
            </button>
            <button onClick={downloadReport} disabled={!img1 || !img2} className="secondary">
              📥 Download Report
            </button>
            {heatmapUrl && (
              <button onClick={() => setShowHeatmap(!showHeatmap)} className="secondary">
                {showHeatmap ? '👁️ Hide Heatmap' : '👁️ Show Heatmap'}
              </button>
            )}
          </div>

          {loading && (
            <div className="progress fade-in">
              <div className="progress-bar"></div>
            </div>
          )}

          {error && (
            <div className="error-message fade-in">
              ⚠️ Error: {error}
            </div>
          )}

          {result && (
            <div className="result fade-in">
              <ScoreRing score={result.similarity_score} threshold={metrics?.threshold || 0.85} />
              <div className="result-details">
                <div className={`verdict ${result.similarity_score >= (metrics?.threshold || 0.85) ? 'success' : 'error'}`}>
                  {result.verdict}
                </div>
                <div className="result-meta">
                  Similarity Score: <strong>{(result.similarity_score * 100).toFixed(2)}%</strong>
                  {verifiedAt && <span> • Verified: {verifiedAt}</span>}
                </div>
              </div>
            </div>
          )}

          <div className="stats-grid fade-in">
            <div className="stat-card">
              <div className="stat-label">Confidence</div>
              <div className="stat-value">{result ? `${Math.round(result.similarity_score * 100)}%` : '--'}</div>
              <div className="stat-sub">Based on cosine similarity</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Model Accuracy</div>
              <div className="stat-value">{metrics?.accuracy ? `${Math.round(metrics.accuracy * 100)}%` : '--'}</div>
              <div className="stat-sub">Threshold: {(metrics?.threshold || 0.85) * 100}%</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Verifications</div>
              <div className="stat-value">{history.filter(h => h.type === 'single').length}</div>
              <div className="stat-sub">Total single verifications</div>
            </div>
          </div>
        </section>
      )}

      {tab === 'batch' && (
        <section className="panel fade-in">
          <h3>Batch Verification</h3>
          <p>Upload one reference signature and multiple files to compare against it</p>

          <div className="pickers">
            <div
              className={`dropzone ${dragOverRef ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOverRef(true); }}
              onDragLeave={() => setDragOverRef(false)}
              onDrop={handleDrop(setRefFile, setRefFileUrl, setDragOverRef)}
            >
              <div className="dropzone-icon">⭐</div>
              <p>Reference Signature</p>
              <input type="file" accept="image/*" onChange={handleFilePick(setRefFile, setRefFileUrl)} />
              {refFile && <small style={{ color: 'var(--success-600)' }}>✓ {refFile.name}</small>}
            </div>

            <div
              className={`dropzone ${dragOverBatch ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOverBatch(true); }}
              onDragLeave={() => setDragOverBatch(false)}
              onDrop={handleBatchFilesDrop}
            >
              <div className="dropzone-icon">📚</div>
              <p>Comparison Signatures</p>
              <input multiple type="file" accept="image/*" onChange={handleBatchFilesPick} />
              {batchFiles.length > 0 && <small style={{ color: 'var(--success-600)' }}>✓ {batchFiles.length} files selected</small>}
            </div>
          </div>

          {refFileUrl && (
            <div className="preview-box fade-in" style={{ maxWidth: '400px', margin: '0 auto' }}>
              <img src={refFileUrl} alt="Reference" />
            </div>
          )}

          <div className="actions">
            <button onClick={runBatch} disabled={batchLoading || !refFile || batchFiles.length === 0} className="success">
              {batchLoading ? '🔄 Processing...' : '▶️ Run Batch Verification'}
            </button>
          </div>

          {batchLoading && (
            <div className="progress fade-in">
              <div className="progress-bar"></div>
            </div>
          )}

          {batchError && (
            <div className="error-message fade-in">
              ⚠️ Error: {batchError}
            </div>
          )}

          {batchResult && (
            <div className="table-wrap fade-in">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Similarity</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResult.results?.map((r, i) => (
                    <tr key={i}>
                      <td>{r.filename || 'unknown'}</td>
                      <td>
                        <strong>{r.similarity_score ? `${Math.round(r.similarity_score * 100)}%` : '--'}</strong>
                      </td>
                      <td>
                        {r.error ? (
                          <span className="verdict-badge error">{r.error}</span>
                        ) : (
                          <span className={`verdict-badge ${r.similarity_score >= (metrics?.threshold || 0.85) ? 'success' : 'error'}`}>
                            {r.verdict}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {tab === 'analytics' && (
        <section className="panel fade-in">
          <h3>Analytics Dashboard</h3>
          <p>View verification statistics and trends</p>
          <AnalyticsDashboard history={history} />
        </section>
      )}

      {tab === 'history' && (
        <section className="panel fade-in">
          <h3>Verification History</h3>
          <p>Review past verifications and results</p>
          {history.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📋</div>
              <h3>No history yet</h3>
              <p>Your verification history will appear here</p>
            </div>
          ) : (
            <div className="history-list">
              {history.slice().reverse().map((item, i) => (
                <div key={i} className="history-item">
                  <div className="history-item-header">
                    <span className="history-item-time">{new Date(item.timestamp).toLocaleString()}</span>
                    <span className={`verdict-badge ${item.type === 'single' && item.result?.similarity_score >= (metrics?.threshold || 0.85) ? 'success' : item.type === 'batch' ? '' : 'error'}`}>
                      {item.type === 'single' ? item.result?.verdict : `Batch (${item.count} files)`}
                    </span>
                  </div>
                  {item.type === 'single' && item.result && (
                    <div className="history-item-score">
                      Score: <strong>{(item.result.similarity_score * 100).toFixed(2)}%</strong>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {tab === 'about' && (
        <section className="panel fade-in">
          <h3>About SignGuard</h3>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.75', marginBottom: 'var(--space-3)' }}>
            SignGuard is an AI-powered signature verification platform built with cutting-edge deep learning technology.
            Using a Siamese neural network architecture with ResNet18 backbone and transformer layers, it achieves high accuracy
            in detecting forged signatures.
          </p>

          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Model Accuracy</div>
              <div className="stat-value">{metrics?.accuracy ? `${Math.round(metrics.accuracy * 100)}%` : '91%'}</div>
              <div className="stat-sub">On test dataset</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">F1 Score</div>
              <div className="stat-value">{metrics?.f1 ? `${Math.round(metrics.f1 * 100)}%` : '90%'}</div>
              <div className="stat-sub">Balanced precision & recall</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Detection Threshold</div>
              <div className="stat-value">{metrics?.threshold ? `${Math.round(metrics.threshold * 100)}%` : '85%'}</div>
              <div className="stat-sub">Optimized for best performance</div>
            </div>
          </div>

          <h4 style={{ marginTop: 'var(--space-4)', marginBottom: 'var(--space-2)' }}>Key Features</h4>
          <ul style={{ lineHeight: '1.75', marginLeft: 'var(--space-3)' }}>
            <li><strong>Real-time Verification:</strong> Upload and compare signatures instantly</li>
            <li><strong>Batch Processing:</strong> Verify multiple signatures against a reference</li>
            <li><strong>Visual Comparison:</strong> Overlay and zoom tools for manual inspection</li>
            <li><strong>AI Explainability:</strong> Saliency heatmaps show model attention</li>
            <li><strong>PDF Reports:</strong> Download professional verification reports</li>
            <li><strong>Analytics Dashboard:</strong> Track trends and statistics</li>
            <li><strong>Dark Mode:</strong> Comfortable viewing in any environment</li>
          </ul>

          <h4 style={{ marginTop: 'var(--space-4)', marginBottom: 'var(--space-2)' }}>Technology Stack</h4>
          <ul style={{ lineHeight: '1.75', marginLeft: 'var(--space-3)' }}>
            <li><strong>Backend:</strong> Flask, PyTorch, ResNet18 with Transformers</li>
            <li><strong>Frontend:</strong> React, Vite, Chart.js</li>
            <li><strong>Model:</strong> Siamese Network with Contrastive Learning</li>
            <li><strong>Features:</strong> Cosine similarity, Grad-CAM visualization</li>
          </ul>

          <div style={{ marginTop: 'var(--space-4)', textAlign: 'center' }}>
            <button onClick={() => { setTab('verify'); scrollToPanel(); }} className="success">
              Start Verifying →
            </button>
          </div>
        </section>
      )}

      <footer className="footer">
        <span>Built by Kanishka | IGDTUW | Powered by AI</span>
        <a href="https://github.com" target="_blank" rel="noreferrer">
          View on GitHub →
        </a>
      </footer>
    </div>
  );
}
