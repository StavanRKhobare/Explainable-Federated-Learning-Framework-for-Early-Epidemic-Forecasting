import { useState, useEffect, useRef } from 'react'
import Plot from '../PlotlyChart'
import React from 'react'

const PRESET_NODES = [
  { censuscode: 572, district: "Bangalore", hospital: "Bangalore General Hospital", port: 8001 },
  { censuscode: 632, district: "Coimbatore", hospital: "Chennai Medical College", port: 8002 },
  { censuscode: 94, district: "New Delhi", hospital: "New Delhi Hospital", port: 8003 }
]

export default function MultiNodeSimulation() {
  const [activeClients, setActiveClients] = useState([])
  const [data, setData] = useState(null)
  const [allDistricts, setAllDistricts] = useState([])  // All India districts for map backdrop
  const [useSim, setUseSim] = useState(true)
  const [loading, setLoading] = useState(false)
  const [selectedNode, setSelectedNode] = useState(null)
  const [nodeDetail, setNodeDetail] = useState(null)
  const [temporalXAI, setTemporalXAI] = useState(null)
  const [spatialXAI, setSpatialXAI] = useState(null)
  const [embedAnalytics, setEmbedAnalytics] = useState(null)
  const [simClock, setSimClock] = useState(null)
  const [simLogs, setSimLogs] = useState([
    { time: new Date().toLocaleTimeString(), type: "info", text: "Multi-System Simulation Dashboard initialized." },
    { time: new Date().toLocaleTimeString(), type: "info", text: "Awaiting incoming 64-dimensional edge embeddings..." }
  ])
  const [shapSummary, setShapSummary] = useState(null) // shap-summary for selected node

  const logSim = (type, text) => {
    setSimLogs(prev => [
      { time: new Date().toLocaleTimeString(), type, text },
      ...prev.slice(0, 49) // Keep last 50 logs
    ])
  }

  // Fetch active clients
  const getClients = async () => {
    try {
      const res = await fetch('/api/active-clients')
      const active = await res.json()
      if (Array.isArray(active)) {
        setActiveClients(prev => {
          active.forEach(c => {
            if (!prev.some(ac => ac.censuscode === c.censuscode)) {
              logSim("connect", `Edge client connected from ${c.district} (Census: ${c.censuscode})`);
            }
          });
          if (JSON.stringify(prev) !== JSON.stringify(active)) {
            return active;
          }
          return prev;
        });
      }
    } catch (e) {
      console.error(e)
    }
  }

  // Fetch simulation clock state
  const fetchSimClock = async () => {
    try {
      const res = await fetch('/api/sim-clock')
      const json = await res.json()
      if (!json.error) {
        setSimClock(prev => JSON.stringify(prev) !== JSON.stringify(json) ? json : prev)
      }
    } catch (e) { console.error(e) }
  }

  // Fetch embedding analytics
  const fetchAnalytics = async () => {
    try {
      const res = await fetch('/api/embedding-analytics')
      const json = await res.json()
      if (!json.error) {
        setEmbedAnalytics(prev => {
          if (prev && prev.nodes) {
            json.nodes.forEach(node => {
              const prevNode = prev.nodes.find(pn => pn.censuscode === node.censuscode);
              if (prevNode && JSON.stringify(prevNode.embedding) !== JSON.stringify(node.embedding)) {
                logSim("info", `Received updated 64-dim embedding from ${node.name} Hospital`);
              }
            });
          }
          if (JSON.stringify(prev) !== JSON.stringify(json)) {
            return json;
          }
          return prev;
        });
      }
    } catch (e) { console.error(e) }
  }

  // Run prediction on current simulation window (defaulting to last window)
  const fetchSimPrediction = async () => {
    setLoading(true)
    try {
      const res = await fetch(`/api/predict?t=-1&use_sim=${useSim}`)
      const json = await res.json()
      if (json.error) {
        logSim("error", `Inference error: ${json.error}`)
      } else {
        setData(prev => JSON.stringify(prev) !== JSON.stringify(json) ? json : prev)
      }
    } catch (e) {
      console.error(e)
      logSim("error", `Failed to contact API server: ${e.message}`)
    }
    setLoading(false)
  }

  // Clock advancement handler
  const handleAdvanceClock = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/sim-clock/advance?step=1', { method: 'POST' })
      const json = await res.json()
      if (json.error) {
        logSim("error", `Clock advance error: ${json.error}`)
      } else {
        setSimClock(json)
        logSim("info", `Simulation clock advanced to Year ${json.year} Week ${json.week}`)
        // Fetch fresh state immediately
        await getClients()
        await fetchSimPrediction()
        await fetchAnalytics()
      }
    } catch (e) {
      logSim("error", `Failed to advance clock: ${e.message}`)
    }
    setLoading(false)
  }

  // Load all India districts once for map backdrop
  useEffect(() => {
    fetch('/api/districts')
      .then(r => r.json())
      .then(setAllDistricts)
      .catch(e => console.error('district load error', e))
  }, [])

  // Fast 5-second polling loop
  useEffect(() => {
    const poll = async () => {
      await getClients()
      await fetchSimClock()
      await fetchAnalytics()
      await fetchSimPrediction()
    }
    poll()
    const timer = setInterval(poll, 5000)
    return () => clearInterval(timer)
  }, [useSim])

  const handleReset = async () => {
    try {
      const res = await fetch('/api/clear-active-clients', { method: 'POST' })
      const json = await res.json()
      if (json.status === 'success') {
        setActiveClients([])
        logSim("info", "Cleared all simulation overrides on central server.")
        fetchSimPrediction()
        fetchAnalytics()
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
      
      // Fetch SHAP summary (heatmap)
      fetch(`/api/shap-summary/${pred.code}?t=-1`)
        .then(r => r.json())
        .then(setShapSummary)
        .catch(err => console.error(err))

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
  // For the edge simulation map: show ALL districts with client cities highlighted
  const mapData = (data?.predictions || []).filter(p => CLIENT_CODES_SET.has(p.code) && p.lat && p.lon)
  // Background districts (non-client, from allDistricts fetch)
  const bgDistricts = allDistricts.filter(d => !CLIENT_CODES_SET.has(d.censuscode) && d.lat && d.lon)

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Multi-System Edge Simulation</h1>
        <p className="page-subtitle">
          Receive privacy-preserving 64-dim embeddings from distributed hospital edge systems (Bangalore, Chennai/Coimbatore, New Delhi) and propagate risk updates live.
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
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
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
            
            <div style={{ width: '1px', height: '24px', background: 'var(--slate-200)' }}></div>

            <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--purple-600)' }}>Global Week:</span>
              <div style={{ padding: '6px 12px', background: 'var(--purple-50)', color: 'var(--purple-700)', border: '1px solid var(--purple-200)', borderRadius: 8, fontSize: '0.85rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
                {simClock ? `${simClock.year}-W${simClock.week}` : 'Loading...'}
              </div>
              <button 
                className="btn btn-outline" 
                style={{ fontSize: '0.75rem', padding: '6px 12px', borderColor: 'var(--purple-300)', color: 'var(--purple-700)', background: 'var(--purple-50)' }}
                onClick={handleAdvanceClock}
                disabled={loading}
              >
                Next Week ➡️
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
                  // Background: all India districts (dim grey)
                  {
                    type: 'scattergeo',
                    lon: bgDistricts.map(d => d.lon),
                    lat: bgDistricts.map(d => d.lat),
                    mode: 'markers',
                    marker: { size: 4, color: 'rgba(148,163,184,0.25)', symbol: 'circle' },
                    hoverinfo: 'skip',
                    showlegend: false,
                  },
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
                          line: { width: 1.5, color: 'rgba(99,102,241,0.3)', dash: 'dot' },
                          hoverinfo: 'skip',
                          showlegend: false,
                        })
                      }
                    }
                    return traces
                  })(),
                  // Highlighted client cities
                  {
                    type: 'scattergeo',
                    lon: mapData.map(p => p.lon),
                    lat: mapData.map(p => p.lat),
                    mode: 'markers+text',
                    text: mapData.map(p => p.name),
                    textposition: 'top center',
                    textfont: { size: 12, color: '#1e293b', family: 'DM Sans', weight: 700 },
                    marker: {
                      size: mapData.map(p => 16 + p.prob * 20),
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
                    center: { lat: 23, lon: 80 },
                    lonaxis: { range: [66, 98] }, lataxis: { range: [6, 38.5] },
                    bgcolor: '#f8fafc', landcolor: '#f1f5f9',
                    subunitcolor: '#e2e8f0',
                    countrycolor: '#cbd5e1', coastlinecolor: '#94a3b8',
                    showland: true, showocean: true, oceancolor: '#eff6ff',
                    showsubunits: true,
                    showlakes: false, framecolor: '#e2e8f0', framewidth: 1,
                    resolution: 50,
                  },
                  paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                  margin: { l: 0, r: 0, t: 10, b: 10 }, height: 580,
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
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.5rem' }}>
                        SHAP Temporal Feature Importance
                        <span style={{ fontSize: '0.6rem', fontWeight: 400, marginLeft: 6, color: 'var(--slate-400)' }}>
                          🟢 green = increases risk &nbsp; 🔴 red = reduces risk
                        </span>
                      </div>
                      {shapSummary?.matrix ? (() => {
                        const FEATURE_EXPLANATIONS = {
                          'temp_k': { name: 'Temperature (K)', desc: 'Higher temperatures speed up mosquito lifecycle.' },
                          'preci_mm': { name: 'Precipitation (mm)', desc: 'Rainfall creates standing breeding pools.' },
                          'lai': { name: 'Vegetation Index (LAI)', desc: 'Dense vegetation provides cover for vectors.' },
                          'cases_lag1': { name: 'Cases (1 wk ago)', desc: 'Active cases propagate immediate local spread.' },
                          'cases_lag2': { name: 'Cases (2 wks ago)', desc: 'Infected carriers from previous cycle.' },
                          'cases_lag3': { name: 'Cases (3 wks ago)', desc: 'Establishes baseline epidemiological momentum.' },
                          'week_sin': { name: 'Seasonality (Sin)', desc: 'Sinusoidal component of seasonal pattern.' },
                          'week_cos': { name: 'Seasonality (Cos)', desc: 'Cos-component identifying monsoon timing.' },
                          'is_monsoon': { name: 'Monsoon Status', desc: 'Identifies high-risk rain season.' },
                          'ner_symptoms': { name: 'NLP: Symptomatic Notes', desc: 'Clinical notes mentioning fever or rash.' },
                          'ner_diseases': { name: 'NLP: Dengue Mentions', desc: 'Mentions of Dengue or vector-borne disease.' },
                          'ner_pathogens': { name: 'NLP: Pathogen Tests', desc: 'Clinical testing requests or pathogen matches.' },
                          'ner_travel': { name: 'NLP: Travel History', desc: 'Indicates imported risk from hot zones.' },
                          'ner_total_notes': { name: 'NLP: EHR Document Volume', desc: 'Total volume of processed electronic records.' }
                        };

                        const topFeats = shapSummary.feature_importance.slice(0, 7)
                        const weekLabels = shapSummary.week_labels || ['t-4','t-3','t-2','t-1'];
                        
                        const z = [];
                        const hoverText = [];
                        for (const f of topFeats) {
                          const fi = shapSummary.features.indexOf(f.feature);
                          const fname_lower = f.feature.toLowerCase();
                          const info = FEATURE_EXPLANATIONS[fname_lower] || { name: f.feature, desc: 'Dynamic feature contributing to temporal risk.' };
                          
                          const zRow = [];
                          const textRow = [];
                          for (let w = 0; w < shapSummary.matrix.length; w++) {
                            const val = shapSummary.matrix[w][fi];
                            zRow.push(val);
                            
                            const signText = val > 0 ? "⚠️ INCREASES risk" : "🟢 REDUCES risk";
                            textRow.push(
                              `<b>Feature:</b> ${info.name}<br>` +
                              `<b>Time:</b> ${weekLabels[w]}<br>` +
                              `<b>SHAP Impact (Log-odds):</b> ${val > 0 ? '+' : ''}${val.toFixed(4)} (${signText})<br>` +
                              `<b>Explanation:</b> ${info.desc}`
                            );
                          }
                          z.push(zRow);
                          hoverText.push(textRow);
                        }

                        const featNames = topFeats.map(f => (FEATURE_EXPLANATIONS[f.feature.toLowerCase()]?.name || f.feature).toUpperCase())
                        return (
                          <Plot
                            data={[{
                              type: 'heatmap',
                              z,
                              x: weekLabels,
                              y: featNames,
                              text: hoverText,
                              hoverinfo: 'text',
                              colorscale: [[0,'#ef4444'],[0.5,'#f9fafb'],[1,'#22c55e']],
                              zmid: 0,
                              showscale: true,
                              colorbar: { thickness: 8, len: 0.9, tickfont: { size: 7, color:'#94a3b8' } }
                            }]}
                            layout={{
                              margin: { t:5, b:30, l:150, r:20 },
                              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                              height: 220,
                              xaxis: { tickfont: { size:9, family:'DM Sans', color:'#64748b' } },
                              yaxis: { tickfont: { size:8, family:'DM Sans', color:'#475569' }, automargin: true }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        )
                      })() : (
                        <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Loading SHAP heatmap...</div>
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
