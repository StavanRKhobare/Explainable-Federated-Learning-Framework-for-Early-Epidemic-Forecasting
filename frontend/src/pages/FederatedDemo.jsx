import { useState, useEffect, useRef } from 'react'

const SAMPLE_JSON = {
  districts: [
    {
      censuscode: 572,
      weeks: [
        { temp_k: 300.2, preci_mm: 12.5, LAI: 0.45, cases_lag1: 3, cases_lag2: 1, cases_lag3: 0, week_sin: 0.87, week_cos: 0.5, is_monsoon: 1, ner_symptoms: 2, ner_diseases: 1, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 3 },
        { temp_k: 301.1, preci_mm: 28.3, LAI: 0.48, cases_lag1: 5, cases_lag2: 3, cases_lag3: 1, week_sin: 0.97, week_cos: 0.26, is_monsoon: 1, ner_symptoms: 3, ner_diseases: 2, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 5 },
        { temp_k: 299.8, preci_mm: 45.0, LAI: 0.52, cases_lag1: 8, cases_lag2: 5, cases_lag3: 3, week_sin: 1.0, week_cos: 0.0, is_monsoon: 1, ner_symptoms: 5, ner_diseases: 2, ner_pathogens: 1, ner_travel: 0, ner_total_notes: 8 },
        { temp_k: 298.5, preci_mm: 62.1, LAI: 0.55, cases_lag1: 12, cases_lag2: 8, cases_lag3: 5, week_sin: 0.97, week_cos: -0.26, is_monsoon: 1, ner_symptoms: 8, ner_diseases: 3, ner_pathogens: 1, ner_travel: 1, ner_total_notes: 12 },
      ]
    },
    {
      censuscode: 577,
      weeks: [
        { temp_k: 298.0, preci_mm: 8.0, LAI: 0.42, cases_lag1: 0, cases_lag2: 0, cases_lag3: 0, week_sin: 0.87, week_cos: 0.5, is_monsoon: 1, ner_symptoms: 0, ner_diseases: 0, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 0 },
        { temp_k: 299.2, preci_mm: 15.4, LAI: 0.44, cases_lag1: 1, cases_lag2: 0, cases_lag3: 0, week_sin: 0.97, week_cos: 0.26, is_monsoon: 1, ner_symptoms: 0, ner_diseases: 0, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 1 },
        { temp_k: 297.5, preci_mm: 32.0, LAI: 0.47, cases_lag1: 2, cases_lag2: 1, cases_lag3: 0, week_sin: 1.0, week_cos: 0.0, is_monsoon: 1, ner_symptoms: 1, ner_diseases: 0, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 1 },
        { temp_k: 296.8, preci_mm: 48.5, LAI: 0.50, cases_lag1: 4, cases_lag2: 2, cases_lag3: 1, week_sin: 0.97, week_cos: -0.26, is_monsoon: 1, ner_symptoms: 2, ner_diseases: 1, ner_pathogens: 0, ner_travel: 0, ner_total_notes: 3 },
      ]
    }
  ]
}

function FeatureTable({ features }) {
  if (!features) return null;
  const groups = {
    "Climate & Environment": [
      { key: 'temp_k', label: 'Temp (K)', val: features.temp_k },
      { key: 'preci_mm', label: 'Rain (mm)', val: features.preci_mm },
      { key: 'LAI', label: 'LAI Vegetation', val: features.LAI }
    ],
    "Case History": [
      { key: 'cases_lag1', label: 'Cases (t-1)', val: features.cases_lag1 },
      { key: 'cases_lag2', label: 'Cases (t-2)', val: features.cases_lag2 },
      { key: 'cases_lag3', label: 'Cases (t-3)', val: features.cases_lag3 }
    ],
    "Seasonality": [
      { key: 'is_monsoon', label: 'Monsoon', val: features.is_monsoon === 1 ? 'Yes' : 'No' },
      { key: 'week_sin', label: 'Week Sin', val: features.week_sin },
      { key: 'week_cos', label: 'Week Cos', val: features.week_cos }
    ],
    "Clinical NER Mentions": [
      { key: 'ner_symptoms', label: 'Symptoms', val: features.ner_symptoms },
      { key: 'ner_diseases', label: 'Diseases', val: features.ner_diseases },
      { key: 'ner_pathogens', label: 'Pathogens', val: features.ner_pathogens },
      { key: 'ner_travel', label: 'Travel', val: features.ner_travel },
      { key: 'ner_total_notes', label: 'Total Notes', val: features.ner_total_notes }
    ]
  };

  return (
    <div style={{ marginTop: '0.6rem', fontSize: '0.75rem' }}>
      <div style={{ fontWeight: 600, color: 'var(--slate-500)', marginBottom: '4px' }}>Input Features:</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {Object.entries(groups).map(([groupName, items]) => {
          if (items.every(item => item.val === undefined)) return null;
          return (
            <div key={groupName} style={{ background: 'var(--slate-50)', padding: '6px 8px', borderRadius: '6px', border: '1px solid var(--slate-100)' }}>
              <div style={{ fontWeight: 700, fontSize: '0.65rem', color: 'var(--slate-400)', textTransform: 'uppercase', marginBottom: '4px' }}>
                {groupName}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))', gap: '4px' }}>
                {items.map(item => {
                  if (item.val === undefined) return null;
                  const formattedVal = typeof item.val === 'number' ? item.val.toFixed(2).replace(/\.00$/, '') : item.val;
                  return (
                    <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--slate-200)', paddingBottom: '2px' }}>
                      <span style={{ color: 'var(--slate-500)', fontSize: '0.68rem' }}>{item.label}:</span>
                      <span className="mono" style={{ fontWeight: 'bold', color: 'var(--slate-700)', fontSize: '0.68rem' }}>{formattedVal}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EmbeddingViz({ values, label }) {
  if (!values || !Array.isArray(values)) return null;
  const displayValues = values.slice(0, 8);
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--slate-500)', marginBottom: '4px' }}>
        {label} <span style={{ fontWeight: 400, color: 'var(--slate-400)' }}>(dims 0-{displayValues.length - 1})</span>
      </div>
      <div className="embedding-grid">
        {displayValues.map((v, i) => {
          const intensity = Math.min(Math.abs(v) * 2.5, 1);
          const bg = v >= 0 ? `rgba(37, 99, 235, ${0.15 + intensity * 0.75})` : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`;
          return (
            <div key={i} className="embedding-cell" style={{ background: bg }} title={`[${i}]=${v}`}>
              {v.toFixed(1)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PipelineDemo() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_JSON, null, 2))
  const [data, setData] = useState(null)
  const [step, setStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [autoPlay, setAutoPlay] = useState(false)
  const [error, setError] = useState('')

  const totalSteps = 6

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
    { title: 'Client Embedding (64-dim)', desc: 'GRU + TGAT + Static features are fused into a compact 64-dimensional vector. ⚡ Only THIS embedding crosses the client → server boundary — NO raw patient data is shared.' },
    { title: 'Server: Spatial DGAT', desc: `The central server receives embeddings from ALL ${data?.total_nodes_in_graph || 284} districts and runs a 4-head Spatial DGAT using ${data?.are_neighbors ? 'direct border connection' : 'graph distance'} between the two selected districts.` },
    { title: 'Dual-Task Prediction', desc: 'The final head produces outbreak probability (classification) and predicted case count (regression) for each district.' },
  ]

  const DistrictColumn = ({ dist, color }) => {
    if (!dist.district) return null;
    const prob = dist.outbreak_prob ?? dist.outbreak_prob_softened ?? 0;
    return (
      <div>
        <div className="card" style={{ marginBottom: '0.8rem', borderLeft: `4px solid ${color}`, padding: '0.8rem 1rem' }}>
          <div style={{ fontWeight: 800, fontSize: '1.05rem' }}>{dist.district}</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{dist.state} · Client Node · Code {dist.censuscode}</div>
        </div>
        <div className="step-flow">
          {stepsMeta.map((sm, i) => {
            const stepNum = i + 1
            const isActive = step === stepNum
            const isComplete = step > stepNum
            return (
              <div key={i} className={`step-card ${isActive ? 'active' : isComplete ? 'completed' : ''}`}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span className="step-number" style={{ background: color }}>{stepNum}</span>
                  <span className="step-title">{sm.title}</span>
                </div>
                {(isActive || isComplete) && (
                  <div style={{ marginTop: '0.5rem' }}>
                    <div className="step-desc">{sm.desc}</div>
                    {stepNum === 1 && dist.raw_features && <FeatureTable features={dist.raw_features} />}
                    {stepNum === 2 && dist.gru_output && <EmbeddingViz values={dist.gru_output} label="GRU Hidden State" />}
                    {stepNum === 3 && dist.tgat_output && <EmbeddingViz values={dist.tgat_output} label="Temporal GAT Output" />}
                    {stepNum === 4 && dist.client_embedding && <EmbeddingViz values={dist.client_embedding} label="Fused Client Embedding (sent to server)" />}
                    {stepNum === 5 && dist.spatial_embedding && <EmbeddingViz values={dist.spatial_embedding} label="After Spatial DGAT (graph-enriched)" />}
                    {stepNum === 6 && (
                      <div style={{ textAlign: 'center', marginTop: '0.8rem' }}>
                        <div className="gauge-val" style={{ color: prob > 0.5 ? 'var(--red-500)' : prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                          {(prob * 100).toFixed(1)}%
                        </div>
                        <div className="metric-label">Outbreak Probability</div>
                        <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                          <div className="prob-bar-fill" style={{ width: `${prob * 100}%`, background: prob > 0.3 ? 'linear-gradient(90deg, #f97316, #ef4444)' : 'linear-gradient(90deg, #10b981, #34d399)' }} />
                        </div>
                        <div className="mono" style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--slate-400)' }}>
                          Predicted cases: {dist.cases_pred?.toFixed(2)}
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
    )
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title" style={{ marginBottom: '0.8rem' }}>📥 Input District Data (JSON)</div>
        <p style={{ fontSize: '0.8rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>
          Paste JSON with 4 weeks of features for 1–2 districts.
          Each week accepts <strong>14 features</strong>: <code>temp_k, preci_mm, LAI, cases_lag1–3, week_sin, week_cos, is_monsoon</code> + NER features <code>ner_symptoms, ner_diseases, ner_pathogens, ner_travel, ner_total_notes</code> (NER fields default to 0 if omitted).
        </p>
        <textarea value={jsonInput} onChange={e => setJsonInput(e.target.value)}
          rows={12} style={{ marginBottom: '0.8rem' }} spellCheck={false} />
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
          <span className="text">Privacy Preserved — Only 64-dimensional embeddings cross the client→server boundary. No raw health, EHR, or weather data is ever shared.</span>
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
          <DistrictColumn dist={d1} color="var(--blue-500)" />
          <DistrictColumn dist={d2} color="#8b5cf6" />
        </div>
      )}

      {data && step >= 6 && d.length >= 2 && (() => {
        const prob1 = d1.outbreak_prob ?? d1.outbreak_prob_softened ?? 0;
        const prob2 = d2.outbreak_prob ?? d2.outbreak_prob_softened ?? 0;
        return (
          <div className="card" style={{ marginTop: '1.5rem' }}>
            <div className="card-title" style={{ marginBottom: '1rem' }}>🔗 Spatial Influence Analysis</div>
            <div className="grid-3">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 700, color: 'var(--blue-600)' }}>{d1.district}</div>
                <div className="serif" style={{ fontSize: '2rem', color: prob1 > 0.3 ? 'var(--red-500)' : 'var(--emerald-500)', marginTop: '0.3rem' }}>
                  {(prob1 * 100).toFixed(1)}%
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
                <div className="serif" style={{ fontSize: '2rem', color: prob2 > 0.3 ? 'var(--red-500)' : 'var(--emerald-500)', marginTop: '0.3rem' }}>
                  {(prob2 * 100).toFixed(1)}%
                </div>
                <div className="metric-label">Outbreak Risk</div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  )
}

// --- Main Export ---
export default function FederatedDemo() {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Split-Federated Learning Demo</h1>
        <p className="page-subtitle">
          Upload custom JSON data for 2 districts → watch the model process it step-by-step through the privacy-preserving pipeline.
        </p>
      </div>
      <PipelineDemo />
    </div>
  )
}
