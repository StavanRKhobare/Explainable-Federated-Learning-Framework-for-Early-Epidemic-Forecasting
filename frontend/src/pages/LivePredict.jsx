import { useState, useEffect } from 'react'
import Plot from '../PlotlyChart'

export default function LivePredict() {
  const [data, setData] = useState(null)
  const [timeIdx, setTimeIdx] = useState(549)  // default: 2019 W34 — peak monsoon, 33 true outbreaks
  const [totalWindows, setTotalWindows] = useState(0)
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [nodeDetail, setNodeDetail] = useState(null)
  const [temporalXAI, setTemporalXAI] = useState(null)
  const [spatialXAI, setSpatialXAI] = useState(null)

  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    fetch('/api/model-info').then(r => r.json()).then(setModelInfo).catch(() => {})
    runPrediction(549)  // load peak monsoon window by default
  }, [])

  useEffect(() => {
    if (!isPlaying) return
    if (timeIdx >= totalWindows - 1) { setIsPlaying(false); return }
    const timer = setTimeout(() => {
      const nextIdx = timeIdx < 0 ? 0 : timeIdx + 1
      setTimeIdx(nextIdx)
      runPrediction(nextIdx)
    }, 1500)
    return () => clearTimeout(timer)
  }, [isPlaying, timeIdx, totalWindows])

  const runPrediction = async (t) => {
    setLoading(true)
    setSelectedNode(null)
    setNodeDetail(null)
    try {
      const res = await fetch(`/api/predict?t=${t}`)
      const json = await res.json()
      if (json.error) { console.error(json.error); setLoading(false); return }
      setData(json)
      setTimeIdx(json.t_idx)
      setTotalWindows(json.total_windows)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const handleNodeClick = async (event) => {
    const pointIndex = event.points?.[0]?.pointIndex
    if (pointIndex === undefined || !data) return
    const pred = data.predictions[pointIndex]
    if (!pred) return
    setSelectedNode(pred)
    setTemporalXAI(null)
    setSpatialXAI(null)
    try {
      const res = await fetch(`/api/district-node/${pred.code}`)
      const json = await res.json()
      setNodeDetail(json)
      
      // Fetch XAI attributions
      fetch(`/api/xai/temporal?censuscode=${pred.code}&t=${timeIdx}`)
        .then(r => r.json())
        .then(setTemporalXAI)
        .catch(err => console.error(err))
      fetch(`/api/xai/spatial?censuscode=${pred.code}&t=${timeIdx}`)
        .then(r => r.json())
        .then(setSpatialXAI)
        .catch(err => console.error(err))
    } catch (e) { console.error(e) }
  }

  const preds = data?.predictions || []
  const top10 = data?.top_10 || []
  const probBins = preds.map(p => p.prob)
  const mapData = preds.filter(p => p.lat && p.lon)

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Live Model Inference</h1>
        <p className="page-subtitle">
          Run the trained FedXGNN model on any time window — click districts on the map for detailed breakdown
        </p>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div className="card-title" style={{ marginBottom: '0.4rem' }}>Select Time Window</div>
            {totalWindows > 0 && (
              <input type="range" min={0} max={totalWindows - 1}
                value={timeIdx >= 0 ? timeIdx : 0}
                onChange={e => { setTimeIdx(parseInt(e.target.value)); setIsPlaying(false) }}
                style={{ width: '100%' }} />
            )}
            <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--slate-400)', marginTop: 4 }}>
              Window {timeIdx >= 0 ? timeIdx : totalWindows - 1} / {totalWindows - 1}
              {data && ` · ${data.year} Week ${data.week}`}
              {timeIdx === 549 && <span style={{ color: 'var(--emerald-500)', marginLeft: 6 }}>★ Peak monsoon (default)</span>}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)', marginTop: 2 }}>
              💡 Default shows 2019 W34 — peak outbreak season with 33 true outbreaks. Slide to explore other periods.
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.8rem' }}>
            <button className="btn btn-outline" onClick={() => runPrediction(timeIdx)} disabled={loading || isPlaying}>
              {loading && !isPlaying ? '⏳ Running...' : '🧠 Run Inference'}
            </button>
            <button className={`btn ${isPlaying ? 'btn-outline' : 'btn-primary'}`} 
              onClick={() => setIsPlaying(!isPlaying)}
              style={!isPlaying ? { background: 'var(--emerald-500)' } : {}}>
              {isPlaying ? '⏸ Pause' : '▶ Play Timeline'}
            </button>
          </div>
        </div>
      </div>

      {data && (
        <>
          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-value">{data.year} W{data.week}</div>
              <div className="metric-label">Prediction Week</div>
            </div>
            <div className="metric-card">
              <div className="metric-value danger">{data.n_high_risk}</div>
              <div className="metric-label">High-Risk Alerts</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{data.n_outbreaks_true}</div>
              <div className="metric-label">True Outbreaks</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{(data.max_prob * 100).toFixed(1)}%</div>
              <div className="metric-label">Peak Risk</div>
            </div>
          </div>

          <div className="grid-main">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Map */}
              <div className="card" style={{ padding: '0.5rem' }}>
                <Plot
                  data={[{
                    type: 'scattergeo',
                    lon: mapData.map(p => p.lon),
                    lat: mapData.map(p => p.lat),
                    mode: 'markers',
                    marker: {
                      size: mapData.map(p => 4 + p.prob * 22),
                      color: mapData.map(p => p.prob),
                      colorscale: [[0, '#10b981'], [0.2, '#34d399'], [0.4, '#fbbf24'], [0.6, '#f97316'], [1, '#ef4444']],
                      cmin: 0, cmax: Math.max(data.max_prob, 0.5),
                      line: { width: 1, color: 'rgba(255,255,255,0.8)' },
                      colorbar: {
                        title: { text: 'Risk', font: { size: 10, color: '#64748b', family: 'DM Sans' } },
                        tickfont: { size: 9, color: '#94a3b8' },
                        thickness: 14, len: 0.5,
                        bgcolor: 'rgba(255,255,255,0.9)', borderwidth: 0, outlinewidth: 0,
                      },
                    },
                    text: mapData.map(p => `<b>${p.name}</b>, ${p.state}<br>Risk: ${(p.prob * 100).toFixed(1)}%${p.truth ? '<br>⚠️ TRUE OUTBREAK' : ''}<br><i>Click for details</i>`),
                    hoverinfo: 'text',
                  }]}
                  layout={{
                    geo: {
                      scope: 'asia', projection: { type: 'mercator' },
                      center: { lat: 22, lon: 80 },
                      lonaxis: { range: [68, 98] }, lataxis: { range: [6, 36] },
                      bgcolor: '#f8fafc', landcolor: '#f1f5f9',
                      countrycolor: '#cbd5e1', coastlinecolor: '#94a3b8',
                      showland: true, showocean: true, oceancolor: '#eff6ff',
                      showlakes: false, framecolor: '#e2e8f0', framewidth: 1,
                    },
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    margin: { l: 0, r: 0, t: 10, b: 10 }, height: 480,
                    font: { family: 'DM Sans', color: '#334155' },
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                  onClick={handleNodeClick}
                />
              </div>

              {/* Distribution */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: '0.3rem' }}>Outbreak Probability Distribution</div>
                <Plot
                  data={[{
                    type: 'histogram', x: probBins,
                    marker: { color: 'rgba(37, 99, 235, 0.5)', line: { color: 'rgba(37, 99, 235, 0.8)', width: 1 } },
                    nbinsx: 40,
                  }]}
                  layout={{
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    xaxis: { title: { text: 'Probability', font: { size: 10, color: '#64748b' } }, gridcolor: '#f1f5f9', tickfont: { size: 9, color: '#94a3b8' } },
                    yaxis: { title: { text: 'Districts', font: { size: 10, color: '#64748b' } }, gridcolor: '#f1f5f9', tickfont: { size: 9, color: '#94a3b8' } },
                    margin: { l: 50, r: 20, t: 10, b: 40 }, height: 180,
                    font: { family: 'DM Sans', color: '#334155' }, bargap: 0.05,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Right sidebar */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Selected node detail */}
              {selectedNode && (
                <div className="card" style={{ borderColor: 'var(--blue-400)' }}>
                  <div className="card-title">Selected District Inference</div>
                  <h3 className="serif" style={{ fontSize: '1.3rem', marginTop: '0.3rem' }}>{selectedNode.name}</h3>
                  <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{selectedNode.state}</div>

                  <div style={{ textAlign: 'center', margin: '1rem 0' }}>
                    <div className="gauge-val" style={{ color: selectedNode.prob > 0.5 ? 'var(--red-500)' : selectedNode.prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                      {(selectedNode.prob * 100).toFixed(1)}%
                    </div>
                    <div className="metric-label">Outbreak Probability</div>
                    <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                      <div className="prob-bar-fill" style={{
                        width: `${selectedNode.prob * 100}%`,
                        background: selectedNode.prob > 0.5 ? 'linear-gradient(90deg, #f97316, #ef4444)' : selectedNode.prob > 0.1 ? 'linear-gradient(90deg, #fbbf24, #f97316)' : 'linear-gradient(90deg, #10b981, #34d399)'
                      }} />
                    </div>
                  </div>

                  {selectedNode.truth === 1 && (
                    <div className="alert-box alert-risk" style={{ justifyContent: 'center', marginBottom: '0.5rem' }}>⚠️ Confirmed outbreak</div>
                  )}
                  {selectedNode.truth === 0 && selectedNode.prob < 0.1 && (
                    <div className="alert-box alert-safe" style={{ justifyContent: 'center', marginBottom: '0.5rem' }}>✓ No outbreak detected</div>
                  )}

                  {nodeDetail?.district && (
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>Inference vs Ground Truth</div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', marginBottom: '1rem' }}>
                        <div className="mini-metric">
                          <div className="value">{nodeDetail.district.pred_cases}</div>
                          <div className="label">Pred. Cases</div>
                        </div>
                        <div className="mini-metric">
                          <div className="value">{nodeDetail.district.actual_cases}</div>
                          <div className="label">Actual Cases</div>
                        </div>
                      </div>

                      <div style={{ fontSize: '0.72rem', color: 'var(--slate-400)', marginBottom: '0.3rem' }}>Client Embedding (first 8 dims)</div>
                      <div className="embedding-grid">
                        {nodeDetail.district.client_embedding?.slice(0, 8).map((v, i) => {
                          const intensity = Math.min(Math.abs(v) * 3, 1)
                          const bg = v >= 0 ? `rgba(37, 99, 235, ${0.15 + intensity * 0.75})` : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`
                          return <div key={i} className="embedding-cell" style={{ background: bg }} title={`[${i}]=${v}`}>{v.toFixed(1)}</div>
                        })}
                      </div>

                      {/* SHAP Explainability */}
                      <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                        <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>SHAP Temporal Feature Importance</div>
                        {temporalXAI ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {Object.entries(
                              temporalXAI.reduce((acc, week) => {
                                Object.entries(week.contributions).forEach(([k, v]) => {
                                  acc[k] = (acc[k] || 0) + Math.abs(v)
                                })
                                return acc
                              }, {})
                            )
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 4)
                            .map(([feat, val]) => {
                              const pct = Math.min(val * 100, 100);
                              return (
                                <div key={feat} style={{ fontSize: '0.7rem' }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                    <span className="mono" style={{ textTransform: 'capitalize', color: 'var(--slate-700)', fontWeight: 500 }}>{feat.replace('_', ' ')}</span>
                                    <span style={{ color: 'var(--slate-400)' }}>{(val * 10).toFixed(2)} SHAP</span>
                                  </div>
                                  <div style={{ width: '100%', background: '#e2e8f0', height: 6, borderRadius: 3 }}>
                                    <div style={{ width: `${Math.max(pct, 5)}%`, background: 'var(--indigo-500)', height: '100%', borderRadius: 3 }} />
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        ) : (
                          <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Calculating temporal SHAP values...</div>
                        )}
                      </div>

                      {/* GNNExplainer Explainability */}
                      <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                        <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>GNNExplainer Spatial Edge Influence</div>
                        {spatialXAI ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {spatialXAI.slice(0, 3).map(n => (
                              <div key={n.censuscode} style={{ fontSize: '0.7rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                  <span style={{ color: 'var(--slate-700)', fontWeight: 500 }}>{n.district} ({n.importance > 0.5 ? '🔥 High' : 'Low'})</span>
                                  <span style={{ color: 'var(--slate-400)' }}>{(n.importance * 100).toFixed(0)}%</span>
                                </div>
                                <div style={{ width: '100%', background: '#e2e8f0', height: 6, borderRadius: 3 }}>
                                  <div style={{ width: `${Math.max(n.importance * 100, 5)}%`, background: 'var(--emerald-500)', height: '100%', borderRadius: 3 }} />
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Computing GNN spatial masks...</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Top 10 */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: '0.8rem' }}>🏆 Top 10 Riskiest</div>
                <div className="table-container" style={{ maxHeight: '400px' }}>
                  <table>
                    <thead><tr><th>#</th><th>District</th><th>Risk</th><th>Actual</th></tr></thead>
                    <tbody>
                      {top10.map((p, i) => (
                        <tr key={p.code} onClick={() => { setSelectedNode(p); fetch(`/api/district-node/${p.code}`).then(r => r.json()).then(setNodeDetail) }}
                          style={{ background: selectedNode?.code === p.code ? 'var(--blue-50)' : undefined }}>
                          <td className="mono" style={{ color: 'var(--slate-400)', fontSize: '0.78rem' }}>{i + 1}</td>
                          <td>
                            <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{p.name}</div>
                            <div style={{ fontSize: '0.68rem', color: 'var(--slate-400)' }}>{p.state}</div>
                          </td>
                          <td>
                            <div className="prob-bar" style={{ width: 70 }}>
                              <div className="prob-bar-fill" style={{
                                width: `${p.prob * 100}%`,
                                background: p.prob > 0.5 ? 'linear-gradient(90deg, #f97316, #ef4444)' : p.prob > 0.2 ? 'linear-gradient(90deg, #fbbf24, #f97316)' : 'linear-gradient(90deg, #3b82f6, #60a5fa)'
                              }} />
                            </div>
                            <div className="mono" style={{ fontSize: '0.72rem', color: p.prob > 0.5 ? 'var(--red-500)' : '#d97706', marginTop: 2 }}>
                              {(p.prob * 100).toFixed(1)}%
                            </div>
                          </td>
                          <td>
                            <div className="mono" style={{ fontSize: '0.85rem', fontWeight: 600, color: p.truth ? 'var(--red-500)' : 'var(--slate-400)' }}>
                              {p.actual_cases}
                            </div>
                            <div style={{ fontSize: '0.6rem', color: 'var(--slate-400)' }}>cases</div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Model info */}
              {modelInfo && (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: '0.5rem' }}>Model</div>
                  <div style={{ fontSize: '0.78rem', lineHeight: 1.9, color: 'var(--slate-600)' }}>
                    <div><span className="badge badge-purple">Client</span> {modelInfo.components?.client?.gru}</div>
                    <div><span className="badge badge-purple">Client</span> {modelInfo.components?.client?.tgat}</div>
                    <div><span className="badge badge-blue">Server</span> {modelInfo.components?.server?.dgat}</div>
                    <div><span className="badge badge-blue">Head</span> Dual-task (Regression + Classification)</div>
                    <div style={{ marginTop: '0.5rem', fontSize: '0.72rem', color: 'var(--slate-400)' }}>
                      {modelInfo.total_params?.toLocaleString()} params · {modelInfo.n_districts} districts · {modelInfo.n_edges} edges
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
