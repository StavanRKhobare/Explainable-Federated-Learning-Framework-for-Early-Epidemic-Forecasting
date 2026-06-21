import { useState, useEffect } from 'react'

const SAMPLE_JSON = {
  districts: [
    {
      censuscode: 572,
      weeks: [
        { temp_k: 300.2, preci_mm: 12.5, LAI: 0.45, cases_lag1: 3, cases_lag2: 1, cases_lag3: 0, week_sin: 0.87, week_cos: 0.5, is_monsoon: 1 },
        { temp_k: 301.1, preci_mm: 28.3, LAI: 0.48, cases_lag1: 5, cases_lag2: 3, cases_lag3: 1, week_sin: 0.97, week_cos: 0.26, is_monsoon: 1 },
        { temp_k: 299.8, preci_mm: 45.0, LAI: 0.52, cases_lag1: 8, cases_lag2: 5, cases_lag3: 3, week_sin: 1.0, week_cos: 0.0, is_monsoon: 1 },
        { temp_k: 298.5, preci_mm: 62.1, LAI: 0.55, cases_lag1: 12, cases_lag2: 8, cases_lag3: 5, week_sin: 0.97, week_cos: -0.26, is_monsoon: 1 },
      ]
    },
    {
      censuscode: 577,
      weeks: [
        { temp_k: 298.0, preci_mm: 8.0, LAI: 0.42, cases_lag1: 0, cases_lag2: 0, cases_lag3: 0, week_sin: 0.87, week_cos: 0.5, is_monsoon: 1 },
        { temp_k: 299.2, preci_mm: 15.4, LAI: 0.44, cases_lag1: 1, cases_lag2: 0, cases_lag3: 0, week_sin: 0.97, week_cos: 0.26, is_monsoon: 1 },
        { temp_k: 297.5, preci_mm: 32.0, LAI: 0.47, cases_lag1: 2, cases_lag2: 1, cases_lag3: 0, week_sin: 1.0, week_cos: 0.0, is_monsoon: 1 },
        { temp_k: 296.8, preci_mm: 48.5, LAI: 0.50, cases_lag1: 4, cases_lag2: 2, cases_lag3: 1, week_sin: 0.97, week_cos: -0.26, is_monsoon: 1 },
      ]
    }
  ]
}

function EmbeddingViz({ values, label }) {
  if (!values || values.length === 0) return null
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)', marginBottom: 3 }}>{label}</div>
      <div className="embedding-grid">
        {values.slice(0, 16).map((v, i) => {
          const intensity = Math.min(Math.abs(v) * 3, 1)
          const bg = v >= 0
            ? `rgba(37, 99, 235, ${0.15 + intensity * 0.75})`
            : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`
          return (
            <div key={i} className="embedding-cell" style={{ background: bg }}
              title={`dim[${i}] = ${v.toFixed(4)}`}>
              {v.toFixed(1)}
            </div>
          )
        })}
        {values.length > 16 && <span style={{ fontSize: '0.65rem', color: 'var(--slate-400)', alignSelf: 'center', marginLeft: 4 }}>+{values.length - 16} more</span>}
      </div>
    </div>
  )
}

function FeatureTable({ features }) {
  const labels = {
    temp_k: '🌡️ Temperature (K)', preci_mm: '🌧️ Rainfall (mm)',
    LAI: '🌿 Leaf Area Index', cases_lag1: '📊 Cases (t-1)',
    cases_lag2: '📊 Cases (t-2)', cases_lag3: '📊 Cases (t-3)',
    week_sin: '🔄 Week sin', week_cos: '🔄 Week cos', is_monsoon: '🌀 Monsoon',
  }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 10px', marginTop: '0.4rem' }}>
      {Object.entries(features).map(([k, v]) => (
        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid var(--slate-50)' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--slate-500)' }}>{labels[k] || k}</span>
          <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--blue-600)' }}>{typeof v === 'number' ? v.toFixed(3) : v}</span>
        </div>
      ))}
    </div>
  )
}

export default function FederatedDemo() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_JSON, null, 2))
  const [data, setData] = useState(null)
  const [step, setStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [autoPlay, setAutoPlay] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState('upload') // 'upload' or 'preset'
  const [districts, setDistricts] = useState([])
  const [activeClients, setActiveClients] = useState([])

  const totalSteps = 6

  useEffect(() => {
    const getClients = () => {
      fetch('/api/active-clients')
        .then(r => r.json())
        .then(setActiveClients)
        .catch(() => {})
    }
    getClients()
    const timer = setInterval(getClients, 2000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    fetch('/api/districts').then(r => r.json()).then(setDistricts).catch(() => {})
  }, [])

  useEffect(() => {
    if (!autoPlay) return
    if (step >= totalSteps) { setAutoPlay(false); return }
    const timer = setTimeout(() => setStep(s => s + 1), 2200)
    return () => clearTimeout(timer)
  }, [autoPlay, step])

  const runCustomPredict = async () => {
    setLoading(true); setError(''); setStep(0)
    try {
      const payload = JSON.parse(jsonInput)
      const res = await fetch('/api/custom-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const json = await res.json()
      if (json.error) { setError(json.error); setLoading(false); return }
      setData(json)
    } catch (e) {
      setError('Invalid JSON: ' + e.message)
    }
    setLoading(false)
  }

  const runPresetDemo = async (d1, d2) => {
    setLoading(true); setError(''); setStep(0)
    try {
      const res = await fetch(`/api/federated-demo?d1=${d1}&d2=${d2}&t=-1`)
      const json = await res.json()
      if (json.error) { setError(json.error); setLoading(false); return }
      setData(json)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  const startDemo = () => {
    setStep(0)
    setTimeout(() => { setAutoPlay(true); setStep(1) }, 200)
  }

  const d = data?.districts || []
  const d1 = d[0] || {}
  const d2 = d[1] || {}

  const stepsMeta = [
    { title: 'Raw District Data (4 Weeks)', desc: 'Each district holds local weather, case history, and population data. This data is PRIVATE and stays on the client device.' },
    { title: 'GRU Temporal Encoding', desc: '2-layer GRU processes the 4-week lookback sequence to capture time-series disease progression trends.' },
    { title: 'Temporal GAT Attention', desc: '4-head Graph Attention Network learns which of the 4 past weeks matter most for predicting the current week\'s outbreak.' },
    { title: 'Client Embedding (32-dim)', desc: 'GRU + TGAT + Static features are fused into a compact 32-dimensional vector. ⚡ Only THIS embedding crosses the client → server boundary — NO raw data is shared.' },
    { title: 'Server: Spatial DGAT', desc: `The central server receives embeddings from ALL ${data?.total_nodes_in_graph || 284} districts and runs a 4-head Spatial DGAT using ${data?.are_neighbors ? 'direct border connection' : 'graph distance'} between the two selected districts.` },
    { title: 'Dual-Task Prediction', desc: 'The final head produces outbreak probability (classification) and predicted case count (regression) for each district.' },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Split-Federated Learning Demo</h1>
        <p className="page-subtitle">Upload custom JSON data for 2 districts → watch the model process it step-by-step through the privacy-preserving pipeline</p>
      </div>

      {/* Active Clients Panel */}
      <div className="card" style={{ marginBottom: '1.5rem', background: 'rgba(16, 185, 129, 0.05)', borderColor: 'rgba(16, 185, 129, 0.3)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.8rem' }}>
          <div>
            <div className="card-title" style={{ color: 'var(--emerald-600)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, background: '#10b981', borderRadius: '50%' }}></span>
              Connected Edge Clients ({activeClients.length})
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--slate-500)', margin: '4px 0 0 0' }}>
              Hospitals running local models and transmitting weekly embeddings to this central server.
            </p>
          </div>
          {activeClients.length > 0 && (
            <button className="btn btn-outline" style={{ borderColor: '#ef4444', color: '#ef4444', padding: '6px 12px', fontSize: '0.75rem' }}
              onClick={() => fetch('/api/clear-active-clients', { method: 'POST' }).then(() => setActiveClients([]))}>
              Reset Overlay
            </button>
          )}
        </div>
        
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap' }}>
          {activeClients.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--slate-400)', fontStyle: 'italic' }}>
              No active edge nodes transmitting. Run client_app.py to connect a hospital.
            </div>
          ) : (
            activeClients.map(c => (
              <div key={c.censuscode} style={{ background: '#ffffff', border: '1px solid var(--slate-200)', padding: '8px 16px', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 8, height: 8, background: '#10b981', borderRadius: '50%' }}></div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{c.district}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Code: {c.censuscode} · State: {c.state}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Input Section */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title" style={{ marginBottom: '0.8rem' }}>📥 Input District Data (JSON)</div>
        <p style={{ fontSize: '0.8rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>
          Paste JSON with 4 weeks of features for 2 districts. Use the sample data or enter your own. Each week needs: temp_k, preci_mm, LAI, cases_lag1-3, week_sin, week_cos, is_monsoon.
        </p>
        <textarea
          value={jsonInput}
          onChange={e => setJsonInput(e.target.value)}
          rows={12}
          style={{ marginBottom: '0.8rem' }}
          spellCheck={false}
        />
        {error && <div className="alert-box alert-risk" style={{ marginBottom: '0.8rem' }}>❌ {error}</div>}
        <div style={{ display: 'flex', gap: '0.8rem', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={runCustomPredict} disabled={loading}>
            {loading ? '⏳ Processing...' : '🧠 Run Through Model'}
          </button>
          <button className="btn btn-outline" onClick={() => setJsonInput(JSON.stringify(SAMPLE_JSON, null, 2))}>
            Reset to Sample
          </button>
          {data && (
            <button className="btn btn-primary" onClick={startDemo} style={{ background: 'var(--emerald-500)' }}>
              ▶ Animate Step-by-Step
            </button>
          )}
          {data && <button className="btn btn-outline" onClick={() => setStep(totalSteps)}>Skip to End</button>}
        </div>
      </div>

      {data && step >= 4 && (
        <div className="privacy-banner">
          <span className="icon">🔒</span>
          <span className="text">Privacy Preserved — Only 32-dimensional embeddings cross the client→server boundary. No raw health or weather data is shared.</span>
        </div>
      )}

      {data && data.are_neighbors !== undefined && (
        <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <span className="badge badge-blue">{data.are_neighbors ? '✓ Neighboring Districts' : '○ Non-adjacent'}</span>
          {d1.edge_weight_km && <span className="badge badge-purple">Border: {d1.edge_weight_km} km</span>}
          <span className="badge badge-blue">{data.embed_dim}d Embedding</span>
        </div>
      )}

      {data && d.length >= 1 && (
        <div className={d.length >= 2 ? "grid-2" : ""}>
          {/* District 1 Column */}
          {d1.district && (
          <div>
            <div className="card" style={{ marginBottom: '0.8rem', borderLeft: '4px solid var(--blue-500)', padding: '0.8rem 1rem' }}>
              <div style={{ fontWeight: 800, fontSize: '1.05rem' }}>{d1.district}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{d1.state} · Client Node · Code {d1.censuscode}</div>
            </div>
            <div className="step-flow">
              {stepsMeta.map((sm, i) => {
                const stepNum = i + 1
                const isActive = step === stepNum
                const isComplete = step > stepNum
                return (
                  <div key={i} className={`step-card ${isActive ? 'active' : isComplete ? 'completed' : ''}`}>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <span className="step-number">{stepNum}</span>
                      <span className="step-title">{sm.title}</span>
                    </div>
                    {(isActive || isComplete) && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <div className="step-desc">{sm.desc}</div>
                        {stepNum === 1 && d1.raw_features && <FeatureTable features={d1.raw_features} />}
                        {stepNum === 2 && d1.gru_output && <EmbeddingViz values={d1.gru_output} label="GRU Hidden State" />}
                        {stepNum === 3 && d1.tgat_output && <EmbeddingViz values={d1.tgat_output} label="Temporal GAT Output" />}
                        {stepNum === 4 && d1.client_embedding && <EmbeddingViz values={d1.client_embedding} label="Fused Client Embedding (sent to server)" />}
                        {stepNum === 5 && d1.spatial_embedding && <EmbeddingViz values={d1.spatial_embedding} label="After Spatial DGAT (graph-enriched)" />}
                        {stepNum === 6 && (
                          <div style={{ textAlign: 'center', marginTop: '0.8rem' }}>
                            <div className="gauge-val" style={{ color: d1.outbreak_prob > 0.5 ? 'var(--red-500)' : d1.outbreak_prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                              {(d1.outbreak_prob * 100).toFixed(1)}%
                            </div>
                            <div className="metric-label">Outbreak Probability</div>
                            <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                              <div className="prob-bar-fill" style={{ width: `${d1.outbreak_prob * 100}%`, background: d1.outbreak_prob > 0.3 ? 'linear-gradient(90deg, #f97316, #ef4444)' : 'linear-gradient(90deg, #10b981, #34d399)' }} />
                            </div>
                            <div className="mono" style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--slate-400)' }}>
                              Predicted cases: {d1.cases_pred?.toFixed(2)}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
          )}

          {/* District 2 Column */}
          {d2.district && (
          <div>
            <div className="card" style={{ marginBottom: '0.8rem', borderLeft: '4px solid #8b5cf6', padding: '0.8rem 1rem' }}>
              <div style={{ fontWeight: 800, fontSize: '1.05rem' }}>{d2.district}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{d2.state} · Client Node · Code {d2.censuscode}</div>
            </div>
            <div className="step-flow">
              {stepsMeta.map((sm, i) => {
                const stepNum = i + 1
                const isActive = step === stepNum
                const isComplete = step > stepNum
                return (
                  <div key={i} className={`step-card ${isActive ? 'active' : isComplete ? 'completed' : ''}`}>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <span className="step-number" style={{ background: '#8b5cf6' }}>{stepNum}</span>
                      <span className="step-title">{sm.title}</span>
                    </div>
                    {(isActive || isComplete) && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <div className="step-desc">{sm.desc}</div>
                        {stepNum === 1 && d2.raw_features && <FeatureTable features={d2.raw_features} />}
                        {stepNum === 2 && d2.gru_output && <EmbeddingViz values={d2.gru_output} label="GRU Hidden State" />}
                        {stepNum === 3 && d2.tgat_output && <EmbeddingViz values={d2.tgat_output} label="Temporal GAT Output" />}
                        {stepNum === 4 && d2.client_embedding && <EmbeddingViz values={d2.client_embedding} label="Fused Client Embedding (sent to server)" />}
                        {stepNum === 5 && d2.spatial_embedding && <EmbeddingViz values={d2.spatial_embedding} label="After Spatial DGAT (graph-enriched)" />}
                        {stepNum === 6 && (
                          <div style={{ textAlign: 'center', marginTop: '0.8rem' }}>
                            <div className="gauge-val" style={{ color: d2.outbreak_prob > 0.5 ? 'var(--red-500)' : d2.outbreak_prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                              {(d2.outbreak_prob * 100).toFixed(1)}%
                            </div>
                            <div className="metric-label">Outbreak Probability</div>
                            <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                              <div className="prob-bar-fill" style={{ width: `${d2.outbreak_prob * 100}%`, background: d2.outbreak_prob > 0.3 ? 'linear-gradient(90deg, #f97316, #ef4444)' : 'linear-gradient(90deg, #10b981, #34d399)' }} />
                            </div>
                            <div className="mono" style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--slate-400)' }}>
                              Predicted cases: {d2.cases_pred?.toFixed(2)}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
          )}
        </div>
      )}

      {/* Spatial influence visualization when both predictions are done */}
      {data && step >= 6 && d.length >= 2 && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="card-title" style={{ marginBottom: '1rem' }}>🔗 Spatial Influence Analysis</div>
          <div className="grid-3">
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 700, color: 'var(--blue-600)' }}>{d1.district}</div>
              <div className="serif" style={{ fontSize: '2rem', color: d1.outbreak_prob > 0.3 ? 'var(--red-500)' : 'var(--emerald-500)', marginTop: '0.3rem' }}>
                {(d1.outbreak_prob * 100).toFixed(1)}%
              </div>
              <div className="metric-label">Outbreak Risk</div>
            </div>
            <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--slate-400)', marginBottom: '0.3rem' }}>
                {data.are_neighbors ? 'Direct Border Connection' : 'Graph Distance'}
              </div>
              <div style={{ width: '100%', height: 3, background: 'linear-gradient(90deg, var(--blue-500), #8b5cf6)', borderRadius: 2 }} />
              {d1.edge_weight_km && (
                <div className="mono" style={{ fontSize: '0.78rem', color: 'var(--blue-600)', marginTop: '0.3rem' }}>
                  {d1.edge_weight_km} km border
                </div>
              )}
              <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)', marginTop: '0.3rem' }}>
                Spatial DGAT propagates disease signals along this connection
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 700, color: '#8b5cf6' }}>{d2.district}</div>
              <div className="serif" style={{ fontSize: '2rem', color: d2.outbreak_prob > 0.3 ? 'var(--red-500)' : 'var(--emerald-500)', marginTop: '0.3rem' }}>
                {(d2.outbreak_prob * 100).toFixed(1)}%
              </div>
              <div className="metric-label">Outbreak Risk</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
