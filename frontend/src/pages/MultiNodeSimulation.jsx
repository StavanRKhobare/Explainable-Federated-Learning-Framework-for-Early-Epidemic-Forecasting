import { useState, useEffect } from 'react'
import Plot from '../PlotlyChart'

const PRESET_NODES = [
  { censuscode: 572, district: "Bangalore", hospital: "Bangalore General Hospital", port: 8001 },
  { censuscode: 632, district: "Coimbatore", hospital: "Chennai Medical College", port: 8002 },
  { censuscode: 94, district: "New Delhi", hospital: "New Delhi Hospital", port: 8003 }
]

export default function MultiNodeSimulation() {
  const [activeClients, setActiveClients] = useState([])
  const [data, setData] = useState(null)
  const [useSim, setUseSim] = useState(true)
  const [loading, setLoading] = useState(false)
  const [selectedNode, setSelectedNode] = useState(null)
  const [nodeDetail, setNodeDetail] = useState(null)
  const [temporalXAI, setTemporalXAI] = useState(null)
  const [spatialXAI, setSpatialXAI] = useState(null)
  const [embedAnalytics, setEmbedAnalytics] = useState(null)
  const [simLogs, setSimLogs] = useState([
    { time: new Date().toLocaleTimeString(), type: "info", text: "Multi-System Simulation Dashboard initialized." },
    { time: new Date().toLocaleTimeString(), type: "info", text: "Awaiting incoming 32-dimensional edge embeddings..." }
  ])

  // Fetch active clients
  const getClients = async () => {
    try {
      const res = await fetch('/api/active-clients')
      const active = await res.json()
      
      // Update logs if a new client connects
      active.forEach(c => {
        if (!activeClients.some(ac => ac.censuscode === c.censuscode)) {
          logSim("connect", `Edge client connected from ${c.district} (Census: ${c.censuscode})`);
        }
      })
      
      setActiveClients(active)
    } catch (e) {
      console.error(e)
    }
  }

  const logSim = (type, text) => {
    setSimLogs(prev => [
      { time: new Date().toLocaleTimeString(), type, text },
      ...prev.slice(0, 49) // Keep last 50 logs
    ])
  }

  useEffect(() => {
    getClients()
    const timer = setInterval(getClients, 2000)
    return () => clearInterval(timer)
  }, [activeClients])

  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/api/embedding-analytics')
      const json = await res.json()
      if (!json.error) setEmbedAnalytics(json)
    } catch (e) { console.error(e) }
  }

  useEffect(() => {
    fetchAnalytics()
    const timer = setInterval(fetchAnalytics, 5000)
    return () => clearInterval(timer)
  }, [])

  // Run prediction on current simulation window (defaulting to last window)
  const fetchSimPrediction = async () => {
    setLoading(true)
    try {
      const res = await fetch(`/api/predict?t=-1&use_sim=${useSim}`)
      const json = await res.json()
      if (json.error) {
        logSim("error", `Inference error: ${json.error}`)
      } else {
        setData(json)
        logSim("info", `Inference executed (${useSim ? "Simulation Overlay" : "Historical Baseline"}). Year ${json.year} Week ${json.week}. Alerts: ${json.n_high_risk}`)
      }
    } catch (e) {
      console.error(e)
      logSim("error", `Failed to contact API server: ${e.message}`)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchSimPrediction()
  }, [useSim])

  const handleReset = async () => {
    try {
      const res = await fetch('/api/clear-active-clients', { method: 'POST' })
      const json = await res.json()
      if (json.status === 'success') {
        setActiveClients([])
        logSim("info", "Cleared all simulation overrides on central server.")
        fetchSimPrediction()
      }
    } catch (e) {
      logSim("error", `Failed to reset simulation: ${e.message}`)
    }
  }

  const CLIENT_CODES = new Set([572, 632, 94])

  const handleNodeClick = async (event) => {
    const pointIndex = event.points?.[0]?.pointIndex
    if (pointIndex === undefined || !data) return
    const clientPreds = (data.predictions || []).filter(p => CLIENT_CODES.has(p.code))
    const pred = clientPreds[pointIndex]
    if (!pred) return
    setSelectedNode(pred)
    setTemporalXAI(null)
    setSpatialXAI(null)
    try {
      const res = await fetch(`/api/district-node/${pred.code}`)
      const json = await res.json()
      setNodeDetail(json)
      
      // Fetch XAI attributions
      fetch(`/api/xai/temporal?censuscode=${pred.code}&t=-1`)
        .then(r => r.json())
        .then(setTemporalXAI)
        .catch(err => console.error(err))
      fetch(`/api/xai/spatial?censuscode=${pred.code}&t=-1`)
        .then(r => r.json())
        .then(setSpatialXAI)
        .catch(err => console.error(err))
    } catch (e) { console.error(e) }
  }

  const CLIENT_CODES_SET = new Set([572, 632, 94])
  // For the edge simulation map: show ONLY the 3 client districts
  const mapData = (data?.predictions || []).filter(p => p.lat && p.lon && CLIENT_CODES_SET.has(p.code))

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Multi-System Edge Simulation</h1>
        <p className="page-subtitle">
          Receive privacy-preserving 32-dim embeddings from distributed hospital edge systems (Bangalore, Chennai/Coimbatore, New Delhi) and propagate risk updates live.
        </p>
      </div>

      {/* Edge Node Status Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        {PRESET_NODES.map(node => {
          const isActive = activeClients.some(ac => ac.censuscode === node.censuscode)
          return (
            <div key={node.censuscode} className="card" style={{ borderLeft: `4px solid ${isActive ? 'var(--emerald-500)' : 'var(--slate-300)'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ margin: '0 0 4px 0', fontSize: '1.1rem' }}>{node.hospital}</h3>
                  <div style={{ fontSize: '0.8rem', color: 'var(--slate-400)', marginBottom: '8px' }}>
                    District: <strong>{node.district}</strong> · Code: <strong>{node.censuscode}</strong>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: '50%', 
                    background: isActive ? '#10b981' : '#94a3b8',
                    boxShadow: isActive ? '0 0 8px #10b981' : 'none'
                  }} />
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isActive ? 'var(--emerald-600)' : 'var(--slate-500)' }}>
                    {isActive ? 'Active' : 'Offline'}
                  </span>
                </div>
              </div>
              <div style={{ borderTop: '1px solid var(--slate-100)', paddingTop: '8px', marginTop: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--slate-400)' }}>Local edge dashboard:</span>
                <a href={`http://localhost:${node.port}`} target="_blank" rel="noreferrer" className="mono" style={{ fontSize: '0.75rem', color: 'var(--blue-600)', textDecoration: 'none', fontWeight: 600 }}>
                  localhost:{node.port} ↗
                </a>
              </div>
            </div>
          )
        })}
      </div>

      {/* Control Panel */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Simulation Mode:</span>
            <div style={{ display: 'flex', border: '1px solid var(--slate-200)', borderRadius: 8, overflow: 'hidden' }}>
              <button 
                className={`btn`} 
                style={{ 
                  borderRadius: 0,
                  padding: '6px 16px',
                  background: !useSim ? 'var(--blue-600)' : 'transparent',
                  color: !useSim ? '#fff' : 'var(--slate-600)',
                  border: 'none',
                  fontSize: '0.8rem'
                }}
                onClick={() => setUseSim(false)}
              >
                Historical Baseline
              </button>
              <button 
                className={`btn`} 
                style={{ 
                  borderRadius: 0,
                  padding: '6px 16px',
                  background: useSim ? 'var(--emerald-600)' : 'transparent',
                  color: useSim ? '#fff' : 'var(--slate-600)',
                  border: 'none',
                  fontSize: '0.8rem'
                }}
                onClick={() => setUseSim(true)}
              >
                Simulation Overlay
              </button>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.8rem' }}>
            <button className="btn btn-outline" onClick={fetchSimPrediction} disabled={loading}>
              Refresh State
            </button>
            <button className="btn btn-outline" style={{ color: '#ef4444', borderColor: '#ef4444' }} onClick={handleReset}>
              Reset Telemetry
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {data && (
        <div>
          {/* Map */}
          <div className="card" style={{ padding: '0.5rem', marginBottom: '1.5rem' }}>
              <div style={{ padding: '8px 12px', fontSize: '0.85rem', color: 'var(--slate-500)' }}>
                <strong>Map View:</strong> {useSim ? "Displaying GNN prediction propagation with live simulated edge overrides" : "Displaying baseline GNN historical forecast"}
              </div>
              <Plot
                data={[
                  // Connections between the 3 client nodes
                  ...(() => {
                    const traces = []
                    for (let i = 0; i < mapData.length; i++) {
                      for (let j = i + 1; j < mapData.length; j++) {
                        traces.push({
                          type: 'scattergeo',
                          lon: [mapData[i].lon, mapData[j].lon],
                          lat: [mapData[i].lat, mapData[j].lat],
                          mode: 'lines',
                          line: { width: 2, color: 'rgba(99,102,241,0.4)', dash: 'dot' },
                          hoverinfo: 'skip',
                          showlegend: false,
                        })
                      }
                    }
                    return traces
                  })(),
                  {
                    type: 'scattergeo',
                    lon: mapData.map(p => p.lon),
                    lat: mapData.map(p => p.lat),
                    mode: 'markers+text',
                    text: mapData.map(p => p.name),
                    textposition: 'top center',
                    textfont: { size: 12, color: '#1e293b', family: 'DM Sans', weight: 700 },
                    marker: {
                      size: mapData.map(p => 22 + p.prob * 28),
                      color: mapData.map(p => p.prob),
                      colorscale: [[0, '#10b981'], [0.2, '#34d399'], [0.4, '#fbbf24'], [0.6, '#f97316'], [1, '#ef4444']],
                      cmin: 0, cmax: Math.max(data.max_prob, 0.5),
                      line: { width: 2.5, color: 'rgba(255,255,255,0.95)' },
                      colorbar: {
                        title: { text: 'Risk', font: { size: 10, color: '#64748b', family: 'DM Sans' } },
                        tickfont: { size: 9, color: '#94a3b8' },
                        thickness: 14, len: 0.5,
                        bgcolor: 'rgba(255,255,255,0.9)', borderwidth: 0, outlinewidth: 0,
                      },
                    },
                    customdata: mapData.map(p => ({
                      isActive: activeClients.some(ac => ac.censuscode === p.code)
                    })),
                    hovertext: mapData.map(p => {
                      const isActive = activeClients.some(ac => ac.censuscode === p.code)
                      return `<b>${p.name}</b> ${isActive ? '🟢 Active' : '⚫ Offline'}<br>` +
                        `Risk: <b>${(p.prob * 100).toFixed(1)}%</b><br>` +
                        `Pred Cases: ${p.pred_cases.toFixed(0)}<br>` +
                        (p.truth ? '⚠️ TRUE OUTBREAK THIS WEEK' : '') +
                        '<br><i>Click for details</i>'
                    }),
                    hoverinfo: 'text',
                    showlegend: false,
                  },
                ]}
                layout={{
                  geo: {
                    scope: 'asia', projection: { type: 'mercator' },
                    center: { lat: 20, lon: 77 },
                    lonaxis: { range: [75, 79] }, lataxis: { range: [10, 30] },
                    bgcolor: '#f8fafc', landcolor: '#f1f5f9',
                    subunitcolor: '#e2e8f0',
                    countrycolor: '#cbd5e1', coastlinecolor: '#94a3b8',
                    showland: true, showocean: true, oceancolor: '#eff6ff',
                    showsubunits: true,
                    showlakes: false, framecolor: '#e2e8f0', framewidth: 1,
                  },
                  paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                  margin: { l: 0, r: 0, t: 10, b: 10 }, height: 560,
                  font: { family: 'DM Sans', color: '#334155' },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
                onClick={handleNodeClick}
              />
          </div>

          {/* Central Embedding Analytics */}
          {embedAnalytics && (
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div className="card-title" style={{ marginBottom: '1rem' }}>Central Spatial Graph Analytics</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                {/* Embeddings grid */}
                {embedAnalytics.nodes?.map(node => (
                  <div key={node.censuscode} style={{ border: '1px solid var(--slate-100)', borderRadius: 8, padding: '1rem' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--blue-600)' }}>
                      {node.name} Client
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>
                      <span>L2 Norm: <strong>{node.l2_norm.toFixed(2)}</strong></span>
                      <span>Mean: <strong>{node.mean.toFixed(2)}</strong></span>
                      <span>Std: <strong>{node.std.toFixed(2)}</strong></span>
                    </div>
                    <div className="embedding-grid">
                      {node.embedding.slice(0, 32).map((v, i) => {
                        const intensity = Math.min(Math.abs(v) * 2.5, 1)
                        const bg = v >= 0 ? `rgba(37, 99, 235, ${0.15 + intensity * 0.75})` : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`
                        return (
                          <div key={i} className="embedding-cell" style={{ background: bg }} title={`dim[${i}]=${v.toFixed(3)}`}>
                            {v.toFixed(1)}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}

                {/* Similarity Matrix */}
                {embedAnalytics.cosine_similarity && (
                  <div style={{ border: '1px solid var(--slate-100)', borderRadius: 8, padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--purple-600)' }}>
                      Inter-Node Cosine Similarity
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>
                      Measuring alignment of disease progression features between hospitals.
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1, justifyContent: 'center' }}>
                      {embedAnalytics.node_names.map((n1, i) => (
                        <div key={n1} style={{ display: 'flex', gap: 4 }}>
                          {embedAnalytics.node_names.map((n2, j) => {
                            const sim = embedAnalytics.cosine_similarity[i][j]
                            return (
                              <div key={n2} style={{ 
                                flex: 1, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: `rgba(16, 185, 129, ${Math.max(0, (sim - 0.9) * 10)})`,
                                border: '1px solid var(--slate-200)', borderRadius: 4,
                                fontSize: '0.75rem', color: sim > 0.95 ? '#064e3b' : 'var(--slate-600)',
                                fontWeight: sim > 0.98 ? 700 : 400
                              }} title={`${n1} vs ${n2}`}>
                                {sim.toFixed(3)}
                              </div>
                            )
                          })}
                        </div>
                      ))}
                      <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                        {embedAnalytics.node_names.map(n => (
                          <div key={n} style={{ flex: 1, textAlign: 'center', fontSize: '0.65rem', color: 'var(--slate-400)' }}>
                            {n.substring(0, 3).toUpperCase()}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Bottom row: node detail + sim logs side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: selectedNode ? '1fr 1fr' : '1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

            {/* Selected node detail */}
            {selectedNode && (
              <div className="card" style={{ borderColor: 'var(--blue-400)' }}>
                <div className="card-title">Inference Analysis</div>
                <h3 className="serif" style={{ fontSize: '1.3rem', marginTop: '0.3rem' }}>{selectedNode.name}</h3>
                <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{selectedNode.state}</div>

                <div style={{ textAlign: 'center', margin: '1rem 0' }}>
                  <div className="gauge-val" style={{ color: selectedNode.prob > 0.5 ? 'var(--red-500)' : selectedNode.prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                    {(selectedNode.prob * 100).toFixed(1)}%
                  </div>
                  <div className="metric-label">Predicted Risk</div>
                  <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                    <div className="prob-bar-fill" style={{
                      width: `${selectedNode.prob * 100}%`,
                      background: selectedNode.prob > 0.5 ? 'linear-gradient(90deg, #f97316, #ef4444)' : selectedNode.prob > 0.1 ? 'linear-gradient(90deg, #fbbf24, #f97316)' : 'linear-gradient(90deg, #10b981, #34d399)'
                    }} />
                  </div>
                </div>

                {nodeDetail?.district && (
                  <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
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

                    <div style={{ fontSize: '0.72rem', color: 'var(--slate-400)', marginBottom: '0.3rem' }}>
                      Client Embedding Vector (dims 0-7)
                    </div>
                    <div className="embedding-grid">
                      {nodeDetail.district.client_embedding?.slice(0, 8).map((v, i) => {
                        const intensity = Math.min(Math.abs(v) * 3, 1)
                        const bg = v >= 0 ? `rgba(37, 99, 235, ${0.15 + intensity * 0.75})` : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`
                        return <div key={i} className="embedding-cell" style={{ background: bg }} title={`[${i}]=${v}`}>{v.toFixed(1)}</div>
                      })}
                    </div>
                    
                    {/* Report Download */}
                    <div style={{ marginTop: '1.2rem', textAlign: 'center' }}>
                      <button className="btn btn-outline" style={{ width: '100%', fontSize: '0.8rem', padding: '8px' }}
                        onClick={() => window.open(`http://localhost:8000/api/report-data/${selectedNode.code}`, '_blank')}>
                        📄 View Snapshot JSON Report
                      </button>
                    </div>

                    {/* SHAP Explainability Heatmap */}
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>SHAP Temporal Heatmap</div>
                      {temporalXAI ? (
                        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(80px, 1fr) repeat(4, 1fr)', gap: 2 }}>
                          <div />
                          {[0, 1, 2, 3].map(w => <div key={w} style={{ fontSize: '0.65rem', color: 'var(--slate-400)', textAlign: 'center' }}>t-{3-w}</div>)}
                          {Object.keys(temporalXAI[0]?.contributions || {})
                            .sort((a, b) => {
                              const sumA = temporalXAI.reduce((sum, w) => sum + Math.abs(w.contributions[a] || 0), 0)
                              const sumB = temporalXAI.reduce((sum, w) => sum + Math.abs(w.contributions[b] || 0), 0)
                              return sumB - sumA
                            })
                            .slice(0, 7)
                            .map(feat => (
                              <React.Fragment key={feat}>
                                <div style={{ fontSize: '0.65rem', color: 'var(--slate-600)', alignSelf: 'center', textTransform: 'capitalize' }}>
                                  {feat.replace('_', ' ').substring(0, 12)}
                                </div>
                                {temporalXAI.map(week => {
                                  const val = week.contributions[feat] || 0
                                  const intensity = Math.min(Math.abs(val) * 3, 1)
                                  const bg = val > 0 ? `rgba(239, 68, 68, ${0.1 + intensity * 0.9})` : `rgba(37, 99, 235, ${0.1 + intensity * 0.9})`
                                  const textCol = intensity > 0.6 ? '#fff' : 'var(--slate-700)'
                                  return (
                                    <div key={week.week_idx} style={{ 
                                      background: bg, color: textCol, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
                                      fontSize: '0.6rem', borderRadius: 2
                                    }} title={`${feat} at t-${3-week.week_idx}: ${val.toFixed(4)}`}>
                                      {val.toFixed(2)}
                                    </div>
                                  )
                                })}
                              </React.Fragment>
                            ))}
                        </div>
                      ) : (
                        <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Calculating temporal SHAP...</div>
                      )}
                    </div>

                    {/* GNNExplainer Explainability */}
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.8rem' }}>GNNExplainer Spatial Influence</div>
                      {spatialXAI ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          {spatialXAI.slice(0, 3).map(n => (
                            <div key={n.censuscode} style={{ fontSize: '0.7rem' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                <span style={{ color: 'var(--slate-700)', fontWeight: 500 }}>{n.district}</span>
                                <span style={{ color: 'var(--slate-400)' }}>{(n.importance * 100).toFixed(0)}%</span>
                              </div>
                              <div style={{ width: '100%', background: '#e2e8f0', height: 6, borderRadius: 3 }}>
                                <div style={{ width: `${Math.max(n.importance * 100, 5)}%`, background: 'var(--emerald-500)', height: '100%', borderRadius: 3 }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Computing spatial influence...</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Sim Logs */}
            <div className="card">
              <h2 style={{ fontSize: '1rem', marginBottom: '0.8rem' }}>Simulation System Logs</h2>
              <div className="console" style={{ height: '260px' }}>
                {simLogs.map((log, idx) => (
                  <div key={idx} className="console-line">
                    <span style={{ color: 'var(--slate-500)' }}>[{log.time}]</span>{' '}
                    <span style={{ color: log.type === 'error' ? '#ef4444' : log.type === 'connect' ? '#10b981' : '#34d399' }}>
                      {log.text}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
