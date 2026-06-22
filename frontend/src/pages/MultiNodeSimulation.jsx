import { useState, useEffect, useRef } from 'react'
import Plot from '../PlotlyChart'
import React from 'react'

const PRESET_NODES = [
  { censuscode: 572, district: "Bangalore", hospital: "Bangalore General Hospital", port: 8001 },
  { censuscode: 632, district: "Coimbatore", hospital: "Chennai Medical College", port: 8002 },
  { censuscode: 94, district: "New Delhi", hospital: "New Delhi Hospital", port: 8003 },
  { censuscode: 577, district: "Mysore", hospital: "Mysore District Hospital", port: 8004 }
]

const casesData = [
  {
    name: "Rohith S. Panchamukhi",
    age: 22,
    ip: "192.168.1.42",
    text: "Patient Rohith S. Panchamukhi (Age 22, IP 192.168.1.42) presenting with sudden onset of high fever, joint pain, and retro-orbital pain. Resident of High-Density District 572.",
    cleanText: "Patient [REDACTED NAME] ([REDACTED AGE], [REDACTED IP]) presenting with sudden onset of high fever, joint pain, and retro-orbital pain. Resident of District 572.",
    features: { symptoms: 2, temp: "301.2 K", preci: "4.2 mm" }
  },
  {
    name: "Aditi Sharma",
    age: 31,
    ip: "192.168.2.115",
    text: "Aditi Sharma, Age 31, IP 192.168.2.115. Patient reports severe muscle pain, skin rash, and high fever since last Tuesday. Clinical notes indicate suspected dengue infection.",
    cleanText: "[REDACTED NAME], [REDACTED AGE], [REDACTED IP]. Patient reports severe muscle pain, skin rash, and high fever since last Tuesday. Clinical notes indicate suspected dengue infection.",
    features: { symptoms: 3, temp: "300.8 K", preci: "12.5 mm" }
  }
];

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

  // New features state
  const [pcaData, setPcaData] = useState(null)
  const [whatIfSliders, setWhatIfSliders] = useState({ temp: 0, preci: 0, cases: 0, symptoms: 0 })
  const [isWhatIfActive, setIsWhatIfActive] = useState(false)
  const [whatIfPredictions, setWhatIfPredictions] = useState(null)
  

  
  // Privacy Firewall Console States
  const [showFirewall, setShowFirewall] = useState(false)
  const [fwCase, setFwCase] = useState(0) // 0: Rohith, 1: Aditi
  const [fwMode, setFwMode] = useState('federated') // 'raw' or 'federated'
  const [fwStatus, setFwStatus] = useState('idle') // 'idle', 'scanning', 'blocked', 'passed'
  const [fwLogs, setFwLogs] = useState([])

  // Tab control
  const [activeTab, setActiveTab] = useState('logs')

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

  const fetchPca = async () => {
    try {
      const res = await fetch('/api/embedding-space-pca')
      const json = await res.json()
      if (!json.error) {
        setPcaData(json)
      }
    } catch (e) { console.error(e) }
  }

  const handleSliderChange = (name, val) => {
    setWhatIfSliders(prev => ({ ...prev, [name]: parseFloat(val) }))
  }

  const runWhatIfSimulation = async () => {
    if (!selectedNode) return
    try {
      const query = new URLSearchParams({
        censuscode: selectedNode.code,
        temp_shift: whatIfSliders.temp,
        preci_shift: whatIfSliders.preci,
        cases_shift: whatIfSliders.cases,
        symptoms_shift: whatIfSliders.symptoms
      }).toString()
      
      const res = await fetch(`/api/simulate_what_if?${query}`)
      const json = await res.json()
      if (!json.error) {
        setWhatIfPredictions(json.predictions)
        setIsWhatIfActive(true)
        logSim('info', `What-If simulation completed for ${selectedNode.name}.`)
      }
    } catch (e) {
      console.error(e)
      logSim('error', `What-If simulation failed.`)
    }
  }

  const handleResetWhatIf = () => {
    setWhatIfSliders({ temp: 0, preci: 0, cases: 0, symptoms: 0 })
    setIsWhatIfActive(false)
    setWhatIfPredictions(null)
  }

  const runSecurityAudit = () => {
    setFwStatus('scanning');
    setFwLogs([
      `[${new Date().toLocaleTimeString()}] INFO: Initializing Edge Node Security Auditor...`,
      `[${new Date().toLocaleTimeString()}] INFO: Retrieving client clinical data from local storage...`
    ]);

    const activeCase = casesData[fwCase];

    setTimeout(() => {
      setFwLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] AUDIT: Local PII Scanner engaged...`,
        `[${new Date().toLocaleTimeString()}] AUDIT: Scanning unstructured text for HIPAA/GDPR identifiers...`
      ]);
    }, 600);

    setTimeout(() => {
      if (fwMode === 'raw') {
        setFwLogs(prev => [
          ...prev,
          `[${new Date().toLocaleTimeString()}] WARNING: Found clinical identifier Name: "${activeCase.name}"`,
          `[${new Date().toLocaleTimeString()}] WARNING: Found network identifier IP: "${activeCase.ip}"`,
          `[${new Date().toLocaleTimeString()}] CRITICAL: Egress scanner detected raw/unencrypted text payload containing direct patient identifiers.`,
          `[${new Date().toLocaleTimeString()}] ERROR: TRANSMISSION BLOCKED. Privacy Firewall Rule #42 triggered (No cross-boundary raw clinical text allowed).`
        ]);
        setFwStatus('blocked');
      } else {
        setFwLogs(prev => [
          ...prev,
          `[${new Date().toLocaleTimeString()}] AUDIT: De-identification filter succeeded. Zero raw identifiers detected.`,
          `[${new Date().toLocaleTimeString()}] AUDIT: Passing structured features into local GRU Client Encoder...`,
          `[${new Date().toLocaleTimeString()}] AUDIT: Generated 64-dimensional secure embedding vector.`,
          `[${new Date().toLocaleTimeString()}] INFO: Payload contains only abstract 64-dimensional float signature.`,
          `[${new Date().toLocaleTimeString()}] INFO: Establishing TLS 1.3 socket link to Central Server Ingestion Queue...`,
          `[${new Date().toLocaleTimeString()}] PASS: Transmission authorized. Embedding successfully ingested.`
        ]);
        setFwStatus('passed');
      }
    }, 1500);
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
      await fetchPca()
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

  const CLIENT_CODES = new Set([572, 632, 94, 577])

  const handleNodeClick = async (event) => {
    const code = event.points?.[0]?.customdata
    if (!code || !data) return
    const pred = (data.predictions || []).find(p => p.code === code)
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

  const CLIENT_CODES_SET = new Set([572, 632, 94, 577])
  // For the edge simulation map: show ALL districts with client cities highlighted
  let predictionsToUse = data?.predictions || []
  if (isWhatIfActive && whatIfPredictions) {
    predictionsToUse = predictionsToUse.map(p => {
      const simMatch = whatIfPredictions.find(sp => sp.code === p.code)
      if (simMatch) {
        return {
          ...p,
          prob: simMatch.prob,
          pred_cases: simMatch.pred_cases,
          isSimulated: true
        }
      }
      return p
    })
  }
  const mapData = predictionsToUse.filter(p => CLIENT_CODES_SET.has(p.code) && p.lat && p.lon)
  // Background districts (non-client, from allDistricts fetch)
  const bgDistricts = allDistricts.filter(d => !CLIENT_CODES_SET.has(d.censuscode) && d.lat && d.lon)

  const displayedNode = selectedNode ? (
    isWhatIfActive && whatIfPredictions 
      ? {
          ...selectedNode,
          prob: whatIfPredictions.find(sp => sp.code === selectedNode.code)?.prob ?? selectedNode.prob
        }
      : selectedNode
  ) : null;

  const displayedPredCases = (isWhatIfActive && whatIfPredictions && selectedNode)
    ? (whatIfPredictions.find(sp => sp.code === selectedNode.code)?.pred_cases ?? nodeDetail?.district?.pred_cases)
    : nodeDetail?.district?.pred_cases;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Multi-System Edge Simulation</h1>
        <p className="page-subtitle">
          Receive privacy-preserving 64-dim embeddings from distributed hospital edge systems (Bangalore, Coimbatore, New Delhi, Mysore) and propagate risk updates live.
        </p>
      </div>

      {/* Edge Node Status Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        {PRESET_NODES.map(node => {
          const activeInfo = activeClients.find(ac => ac.censuscode === node.censuscode)
          const isActive = !!activeInfo
          const clientIp = activeInfo?.ip || window.location.hostname || 'localhost'
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
                <a href={`http://${clientIp}:${node.port}`} target="_blank" rel="noreferrer" className="mono" style={{ fontSize: '0.75rem', color: 'var(--blue-600)', textDecoration: 'none', fontWeight: 600 }}>
                  {clientIp}:{node.port} ↗
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
            <button 
              className="btn btn-outline" 
              style={{ color: 'var(--emerald-600)', borderColor: 'var(--emerald-600)' }}
              onClick={() => {
                setShowFirewall(true);
                setFwStatus('idle');
                setFwLogs([]);
              }}
            >
              🛡️ Privacy Firewall Demo
            </button>
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
                    customdata: bgDistricts.map(d => d.censuscode),
                    mode: 'markers',
                    marker: { size: 4, color: 'rgba(148,163,184,0.25)', symbol: 'circle' },
                    hovertext: bgDistricts.map(d => {
                      const p = predictionsToUse.find(pred => pred.code === d.censuscode)
                      const probStr = p ? `${(p.prob * 100).toFixed(1)}%` : 'N/A'
                      return `<b>${d.name}</b><br>Risk: <b>${probStr}</b><br><i>Click to inspect details</i>`
                    }),
                    hoverinfo: 'text',
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
                    customdata: mapData.map(p => p.code),
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
                        `Risk: <b>${(p.prob * 100).toFixed(1)}%</b>${p.isSimulated ? ' (Simulated)' : ''}<br>` +
                        `Pred Cases: ${p.pred_cases.toFixed(0)}<br>` +
                        (p.truth ? '⚠️ TRUE OUTBREAK THIS WEEK' : '') +
                        '<br><i>Click for details</i>'
                    }),
                    hoverinfo: 'text',
                    showlegend: false,
                  },
                ]}
                layout={{
                  uirevision: 'constant',
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
                  <div key={node.censuscode} style={{ border: '1px solid var(--slate-100)', borderRadius: 8, padding: '1rem', background: '#ffffff' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--blue-600)' }}>
                        {node.name} Client
                      </div>
                      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                        {node.is_live ? (
                          <span style={{ background: '#ecfdf5', color: '#059669', fontSize: '0.65rem', fontWeight: 600, padding: '2px 6px', borderRadius: 4, border: '1px solid #d1fae5' }}>Active</span>
                        ) : (
                          <span style={{ background: '#fef2f2', color: '#dc2626', fontSize: '0.65rem', fontWeight: 600, padding: '2px 6px', borderRadius: 4, border: '1px solid #fee2e2' }}>Offline</span>
                        )}
                        <span style={{ 
                          fontSize: '0.75rem', 
                          fontWeight: 700, 
                          padding: '2px 6px', 
                          borderRadius: 4, 
                          background: node.outbreak_prob > 0.5 ? '#fef2f2' : node.outbreak_prob > 0.3 ? '#fffbeb' : '#f0fdf4',
                          color: node.outbreak_prob > 0.5 ? '#dc2626' : node.outbreak_prob > 0.3 ? '#d97706' : '#16a34a',
                          border: `1px solid ${node.outbreak_prob > 0.5 ? '#fee2e2' : node.outbreak_prob > 0.3 ? '#fef3c7' : '#dcfce7'}`
                        }}>
                          Risk: {(node.outbreak_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-start', fontSize: '0.7rem', color: 'var(--slate-500)', marginBottom: '0.5rem' }}>
                      <span>Predicted Cases: <strong style={{ color: 'var(--slate-700)' }}>{node.pred_cases.toFixed(0)}</strong></span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--slate-400)', marginBottom: '0.8rem', borderTop: '1px dashed var(--slate-100)', paddingTop: '0.4rem' }}>
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


              </div>
            </div>
          )}

          {/* Bottom row: node detail + sim logs side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: selectedNode ? '1fr 1fr' : '1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

            {/* Selected node detail */}
            {selectedNode && displayedNode && (
              <div className="card" style={{ borderColor: 'var(--blue-400)' }}>
                <div className="card-title">Inference Analysis</div>
                <h3 className="serif" style={{ fontSize: '1.3rem', marginTop: '0.3rem' }}>{displayedNode.name}</h3>
                <div style={{ fontSize: '0.78rem', color: 'var(--slate-400)' }}>{displayedNode.state}</div>

                <div style={{ textAlign: 'center', margin: '1rem 0' }}>
                  <div className="gauge-val" style={{ color: displayedNode.prob > 0.5 ? 'var(--red-500)' : displayedNode.prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                    {(displayedNode.prob * 100).toFixed(1)}%
                  </div>
                  <div className="metric-label">Predicted Risk</div>
                  <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                    <div className="prob-bar-fill" style={{
                      width: `${displayedNode.prob * 100}%`,
                      background: displayedNode.prob > 0.5 ? 'linear-gradient(90deg, #f97316, #ef4444)' : displayedNode.prob > 0.1 ? 'linear-gradient(90deg, #fbbf24, #f97316)' : 'linear-gradient(90deg, #10b981, #34d399)'
                    }} />
                  </div>
                </div>

                {nodeDetail?.district && (
                  <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '0.8rem', marginBottom: '1rem' }}>
                      <div className="mini-metric">
                        <div className="value">{displayedPredCases?.toFixed(0)}</div>
                        <div className="label">Pred. Cases</div>
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
                        onClick={() => window.open(`http://${window.location.hostname || 'localhost'}:8000/api/report-data/${selectedNode.code}`, '_blank')}>
                        📄 View Snapshot JSON Report
                      </button>
                    </div>

                    {/* SHAP Explainability Waterfall Chart */}
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.5rem' }}>
                        SHAP Temporal Feature Attribution (Risk Change %)
                      </div>
                      {shapSummary?.matrix ? (() => {
                        const FEATURE_EXPLANATIONS = {
                          'temp_k': { name: 'Temperature', desc: 'Higher temperatures speed up mosquito lifecycle.' },
                          'preci_mm': { name: 'Precipitation', desc: 'Rainfall creates standing breeding pools.' },
                          'lai': { name: 'Vegetation (LAI)', desc: 'Dense vegetation provides cover for vectors.' },
                          'cases_lag1': { name: 'Cases (1 wk ago)', desc: 'Active cases propagate immediate local spread.' },
                          'cases_lag2': { name: 'Cases (2 wks ago)', desc: 'Infected carriers from previous cycle.' },
                          'cases_lag3': { name: 'Cases (3 wks ago)', desc: 'Establishes baseline epidemiological momentum.' },
                          'week_sin': { name: 'Seasonality (Sin)', desc: 'Sinusoidal component of seasonal pattern.' },
                          'week_cos': { name: 'Seasonality (Cos)', desc: 'Cos-component identifying monsoon timing.' },
                          'is_monsoon': { name: 'Monsoon Status', desc: 'Identifies high-risk rain season.' },
                          'ner_symptoms': { name: 'NLP: Symptoms', desc: 'Clinical notes mentioning fever or rash.' },
                          'ner_diseases': { name: 'NLP: Dengue', desc: 'Mentions of Dengue or vector-borne disease.' },
                          'ner_pathogens': { name: 'NLP: Pathogens', desc: 'Clinical testing requests or pathogen matches.' },
                          'ner_travel': { name: 'NLP: Travel', desc: 'Indicates imported risk from hot zones.' },
                          'ner_total_notes': { name: 'NLP: EHR Vol', desc: 'Total volume of processed electronic records.' }
                        };

                        const topFeats = shapSummary.feature_importance.slice(0, 5);
                        const p = displayedNode?.prob ?? selectedNode.prob;
                        const final_logit = Math.log(p / (1 - p + 1e-9));

                        // Sum SHAP values across all 4 weeks for each feature
                        const featureSums = {};
                        topFeats.forEach(f => {
                          const fi = shapSummary.features.indexOf(f.feature);
                          let sum = 0;
                          for (let w = 0; w < 4; w++) {
                            sum += shapSummary.matrix[w][fi];
                          }
                          featureSums[f.feature] = sum;
                        });

                        // Compute total SHAP sum for all features (to find the baseline odds)
                        const allFeaturesSums = {};
                        shapSummary.features.forEach((feat, fi) => {
                          let sum = 0;
                          for (let w = 0; w < 4; w++) {
                            sum += shapSummary.matrix[w][fi];
                          }
                          allFeaturesSums[feat] = sum;
                        });
                        const total_shap = Object.values(allFeaturesSums).reduce((a, b) => a + b, 0);
                        const base_logit = final_logit - total_shap;

                        // Calculate sequential probability steps
                        let current_logit = base_logit;
                        let current_prob = 1 / (1 + Math.exp(-current_logit));

                        const steps = [];
                        steps.push({
                          name: 'Base Risk',
                          value: current_prob * 100,
                          measure: 'absolute',
                          text: `${(current_prob * 100).toFixed(0)}%`
                        });

                        topFeats.forEach(f => {
                          const shapVal = featureSums[f.feature] || 0;
                          const next_logit = current_logit + shapVal;
                          const next_prob = 1 / (1 + Math.exp(-next_logit));
                          const diff = (next_prob - current_prob) * 100;
                          
                          const displayName = FEATURE_EXPLANATIONS[f.feature.toLowerCase()]?.name || f.feature;
                          steps.push({
                            name: displayName,
                            value: diff,
                            measure: 'relative',
                            text: `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`
                          });
                          
                          current_logit = next_logit;
                          current_prob = next_prob;
                        });

                        // Add a step for other combined features if there's a residual difference
                        const target_prob = p;
                        const residual = (target_prob - current_prob) * 100;
                        if (Math.abs(residual) > 0.1) {
                          steps.push({
                            name: 'Other',
                            value: residual,
                            measure: 'relative',
                            text: `${residual >= 0 ? '+' : ''}${residual.toFixed(1)}%`
                          });
                        }

                        // Add final predicted risk
                        steps.push({
                          name: 'Final Risk',
                          value: p * 100,
                          measure: 'total',
                          text: `${(p * 100).toFixed(1)}%`
                        });

                        const xLabels = steps.map(s => s.name);
                        const yValues = steps.map(s => s.value);
                        const measures = steps.map(s => s.measure);
                        const textValues = steps.map(s => s.text);

                        return (
                          <Plot
                            data={[{
                              type: 'waterfall',
                              orientation: 'v',
                              measure: measures,
                              x: xLabels,
                              y: yValues,
                              text: textValues,
                              textposition: 'outside',
                              connector: {
                                line: { color: 'rgba(148,163,184,0.3)', width: 1.5 }
                              },
                              decreasing: { marker: { color: '#ef4444' } },
                              increasing: { marker: { color: '#10b981' } },
                              totals: { marker: { color: '#6366f1' } }
                            }]}
                            layout={{
                              margin: { t: 25, b: 60, l: 30, r: 20 },
                              paper_bgcolor: 'transparent',
                              plot_bgcolor: 'transparent',
                              height: 230,
                              xaxis: { tickfont: { size: 8, family: 'DM Sans', color: '#64748b' }, automargin: true },
                              yaxis: { tickfont: { size: 9, family: 'DM Sans', color: '#475569' }, title: { text: 'Risk Probability (%)', font: { size: 10 } } }
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            style={{ width: '100%' }}
                          />
                        );
                      })() : (
                        <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>Loading SHAP waterfall...</div>
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

                    {/* Counterfactual "What-If" Analysis */}
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--slate-100)', paddingTop: '1rem' }}>
                      <div className="card-title" style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.5rem' }}>
                        Counterfactual "What-If" Simulator
                      </div>
                      <div style={{ display: 'grid', gridTemplateRows: 'repeat(4, 1fr)', gap: '0.6rem', fontSize: '0.72rem' }}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span>Temperature Shift:</span>
                            <strong>{whatIfSliders.temp >= 0 ? '+' : ''}{whatIfSliders.temp.toFixed(1)} K</strong>
                          </div>
                          <input 
                            type="range" min="-5" max="5" step="0.5" 
                            value={whatIfSliders.temp} 
                            onChange={(e) => handleSliderChange('temp', e.target.value)} 
                            style={{ width: '100%' }} 
                          />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span>Precipitation Shift:</span>
                            <strong>{whatIfSliders.preci >= 0 ? '+' : ''}{whatIfSliders.preci.toFixed(0)} mm</strong>
                          </div>
                          <input 
                            type="range" min="-50" max="100" step="5" 
                            value={whatIfSliders.preci} 
                            onChange={(e) => handleSliderChange('preci', e.target.value)} 
                            style={{ width: '100%' }} 
                          />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span>Case Count Shift:</span>
                            <strong>{whatIfSliders.cases >= 0 ? '+' : ''}{whatIfSliders.cases.toFixed(0)} cases</strong>
                          </div>
                          <input 
                            type="range" min="-20" max="100" step="2" 
                            value={whatIfSliders.cases} 
                            onChange={(e) => handleSliderChange('cases', e.target.value)} 
                            style={{ width: '100%' }} 
                          />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span>Symptomatic Notes Shift:</span>
                            <strong>{whatIfSliders.symptoms >= 0 ? '+' : ''}{whatIfSliders.symptoms.toFixed(0)} occurrences</strong>
                          </div>
                          <input 
                            type="range" min="-10" max="50" step="1" 
                            value={whatIfSliders.symptoms} 
                            onChange={(e) => handleSliderChange('symptoms', e.target.value)} 
                            style={{ width: '100%' }} 
                          />
                        </div>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: isWhatIfActive ? '1fr 1fr' : '1fr', gap: '0.5rem', marginTop: '0.8rem' }}>
                        <button 
                          className="btn btn-primary" 
                          style={{ width: '100%', padding: '8px', fontSize: '0.75rem', fontWeight: 'bold' }}
                          onClick={runWhatIfSimulation}
                        >
                          ▶ Run Simulation
                        </button>
                        {isWhatIfActive && (
                          <button 
                            className="btn btn-outline" 
                            style={{ width: '100%', padding: '8px', fontSize: '0.75rem', borderColor: '#ef4444', color: '#ef4444' }}
                            onClick={handleResetWhatIf}
                          >
                            ✕ Clear
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Console / PCA Tabs */}
            <div className="card">
              <div style={{ display: 'flex', borderBottom: '1px solid var(--slate-200)', marginBottom: '0.8rem', paddingBottom: '0.4rem', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <button 
                    style={{
                      background: 'none', border: 'none', 
                      fontWeight: activeTab === 'logs' ? 'bold' : 'normal',
                      color: activeTab === 'logs' ? 'var(--blue-600)' : 'var(--slate-400)',
                      cursor: 'pointer', borderBottom: activeTab === 'logs' ? '2px solid var(--blue-600)' : 'none',
                      paddingBottom: '4px', fontSize: '0.9rem'
                    }}
                    onClick={() => setActiveTab('logs')}
                  >
                    System Logs
                  </button>
                  <button 
                    style={{
                      background: 'none', border: 'none', 
                      fontWeight: activeTab === 'pca' ? 'bold' : 'normal',
                      color: activeTab === 'pca' ? 'var(--blue-600)' : 'var(--slate-400)',
                      cursor: 'pointer', borderBottom: activeTab === 'pca' ? '2px solid var(--blue-600)' : 'none',
                      paddingBottom: '4px', fontSize: '0.9rem'
                    }}
                    onClick={() => {
                      setActiveTab('pca');
                      fetchPca();
                    }}
                  >
                    Epidemiological Clusters (PCA)
                  </button>
                </div>
              </div>

              {activeTab === 'logs' ? (
                <div className="console" style={{ minHeight: '300px', height: 'auto', overflow: 'visible' }}>
                  {simLogs.map((log, idx) => (
                    <div key={idx} className="console-line">
                      <span style={{ color: 'var(--slate-500)' }}>[{log.time}]</span>{' '}
                      <span style={{ color: log.type === 'error' ? '#ef4444' : log.type === 'connect' ? '#10b981' : '#34d399' }}>
                        {log.text}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ minHeight: '300px', height: 'auto', overflow: 'visible' }}>
                  {pcaData ? (
                    <Plot
                      data={[{
                        x: pcaData.nodes.map(n => n.x),
                        y: pcaData.nodes.map(n => n.y),
                        mode: 'markers',
                        type: 'scatter',
                        text: pcaData.nodes.map(n => `<b>${n.name}</b><br>Risk: ${(n.prob * 100).toFixed(1)}%<br>Cases: ${n.cases.toFixed(0)}`),
                        hoverinfo: 'text',
                        marker: {
                          size: pcaData.nodes.map(n => n.is_client ? 12 : 6),
                          color: pcaData.nodes.map(n => n.prob),
                          colorscale: [[0, '#10b981'], [0.2, '#34d399'], [0.4, '#fbbf24'], [0.6, '#f97316'], [1, '#ef4444']],
                          line: {
                            width: pcaData.nodes.map(n => n.is_client ? 2 : 0),
                            color: '#1e293b'
                          }
                        }
                      }]}
                      layout={{
                        margin: { t: 10, b: 30, l: 30, r: 10 },
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        height: 250,
                        xaxis: { title: 'PCA Dimension 1', tickfont: { size: 8 } },
                        yaxis: { title: 'PCA Dimension 2', tickfont: { size: 8 } }
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: '100%' }}
                    />
                  ) : (
                    <div style={{ fontSize: '0.8rem', color: 'var(--slate-400)', textAlign: 'center', paddingTop: '4rem' }}>
                      Loading cluster projection...
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showFirewall && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(8px)',
          display: 'flex', justifyContent: 'center', alignItems: 'center',
          zIndex: 1000, padding: '2rem'
        }}>
          <div className="card" style={{
            width: '100%', maxWidth: '950px', background: '#0f172a', color: '#f8fafc',
            border: '1px solid #334155', borderRadius: '16px', padding: '2rem',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid #334155', paddingBottom: '1rem' }}>
              <div>
                <h2 style={{ margin: 0, fontSize: '1.5rem', color: '#38bdf8' }}>Edge Node Security Auditor Console</h2>
                <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
                  Simulate local HIPAA de-identification and the Split-Federated privacy firewall.
                </p>
              </div>
              <button 
                className="btn btn-outline" 
                style={{ color: '#94a3b8', borderColor: '#475569', padding: '4px 12px' }}
                onClick={() => {
                  setShowFirewall(false);
                  setFwStatus('idle');
                  setFwLogs([]);
                }}
              >
                ✕ Close
              </button>
            </div>

            {/* Select EHR Record Dropdown */}
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1.2rem', background: '#1e293b', padding: '10px', borderRadius: '8px' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#e2e8f0' }}>Select Patient EHR:</span>
              <select 
                value={fwCase} 
                onChange={(e) => {
                  setFwCase(parseInt(e.target.value));
                  setFwStatus('idle');
                  setFwLogs([]);
                }}
                style={{ background: '#0f172a', color: '#f8fafc', border: '1px solid #475569', borderRadius: '4px', padding: '4px 8px', fontSize: '0.8rem' }}
              >
                <option value={0}>Case A: Rohith S. Panchamukhi (Fever & Joint Pain)</option>
                <option value={1}>Case B: Aditi Sharma (Suspected Dengue Infection)</option>
              </select>
              <div style={{ display: 'flex', gap: '0.8rem', marginLeft: 'auto' }}>
                <button 
                  onClick={() => { setFwMode('raw'); setFwStatus('idle'); setFwLogs([]); }} 
                  style={{
                    padding: '4px 12px', fontSize: '0.75rem', borderRadius: '4px', cursor: 'pointer',
                    background: fwMode === 'raw' ? '#ef4444' : '#334155',
                    color: '#fff', border: 'none', fontWeight: 'bold'
                  }}
                >
                  Direct Raw Transmit (Unsafe)
                </button>
                <button 
                  onClick={() => { setFwMode('federated'); setFwStatus('idle'); setFwLogs([]); }} 
                  style={{
                    padding: '4px 12px', fontSize: '0.75rem', borderRadius: '4px', cursor: 'pointer',
                    background: fwMode === 'federated' ? '#10b981' : '#334155',
                    color: '#fff', border: 'none', fontWeight: 'bold'
                  }}
                >
                  Federated Transmit (Secure)
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'stretch' }}>
              {/* Left Side: Edge Local Environment */}
              <div style={{ border: '1px solid #334155', borderRadius: '12px', padding: '1.2rem', background: '#1e293b', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#38bdf8' }}>🏥 EDGE NODE (LOCAL HOSPITAL CLINIC)</span>
                </div>
                
                <div>
                  <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Incoming EHR Intake Note:</label>
                  <div style={{
                    fontFamily: 'monospace', fontSize: '0.72rem', background: '#0f172a',
                    padding: '10px', borderRadius: '6px', border: '1px solid #334155',
                    color: '#f8fafc', minHeight: '80px', lineHeight: '1.4'
                  }}>
                    {casesData[fwCase].text}
                  </div>
                </div>

                {fwMode === 'federated' && (
                  <div>
                    <label style={{ fontSize: '0.75rem', color: '#10b981', display: 'block', marginBottom: '4px' }}>🛡️ Local De-identification Output:</label>
                    <div style={{
                      fontFamily: 'monospace', fontSize: '0.72rem', background: '#0f172a',
                      padding: '10px', borderRadius: '6px', border: '1px solid #10b981',
                      color: '#a7f3d0', minHeight: '80px', lineHeight: '1.4'
                    }}>
                      {casesData[fwCase].cleanText}
                    </div>
                  </div>
                )}

                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  <label style={{ display: 'block', marginBottom: '4px' }}>Fitted Local Numerical Features:</label>
                  <table style={{ width: '100%', fontSize: '0.7rem', background: '#0f172a', borderRadius: '6px', padding: '6px' }}>
                    <tbody>
                      <tr>
                        <td style={{ color: '#38bdf8', padding: '2px 4px' }}>ner_symptoms</td>
                        <td align="right" style={{ padding: '2px 4px' }}>{casesData[fwCase].features.symptoms} occurrences</td>
                      </tr>
                      <tr>
                        <td style={{ color: '#38bdf8', padding: '2px 4px' }}>temp_k</td>
                        <td align="right" style={{ padding: '2px 4px' }}>{casesData[fwCase].features.temp}</td>
                      </tr>
                      <tr>
                        <td style={{ color: '#38bdf8', padding: '2px 4px' }}>preci_mm</td>
                        <td align="right" style={{ padding: '2px 4px' }}>{casesData[fwCase].features.preci}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Secure local embedding preview */}
                {fwMode === 'federated' && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: '#38bdf8', marginBottom: '4px' }}>Generated Local 64-Dimensional Secure Embedding:</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '2px', height: '24px', background: '#0f172a', padding: '4px', borderRadius: '4px' }}>
                      {Array.from({ length: 64 }).map((_, i) => {
                        const val = Math.sin(i * 0.15 + fwCase) * 0.8;
                        const bg = val >= 0 ? `rgba(56, 189, 248, ${0.3 + val})` : `rgba(239, 68, 68, ${0.3 + Math.abs(val)})`;
                        return <div key={i} style={{ background: bg, borderRadius: '1px' }} title={`dim ${i}: ${val.toFixed(3)}`} />
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Right Side: Security Audit Console */}
              <div style={{ border: '1px solid #334155', borderRadius: '12px', padding: '1.2rem', background: '#090d16', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem' }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#e2e8f0', fontFamily: 'monospace' }}>🖥️ SECURITY FIREWALL TERMINAL</span>
                    <span style={{
                      fontSize: '0.7rem', borderRadius: 4, padding: '2px 8px', fontWeight: 'bold',
                      background: fwStatus === 'blocked' ? 'rgba(239, 68, 68, 0.2)' : fwStatus === 'passed' ? 'rgba(16, 185, 129, 0.2)' : '#1e293b',
                      color: fwStatus === 'blocked' ? '#ef4444' : fwStatus === 'passed' ? '#10b981' : '#94a3b8'
                    }}>
                      STATUS: {fwStatus.toUpperCase()}
                    </span>
                  </div>

                  {/* Terminal log window */}
                  <div style={{
                    flex: 1, background: '#020617', border: '1px solid #1e293b', borderRadius: '6px',
                    padding: '12px', fontFamily: 'monospace', fontSize: '0.7rem', overflowY: 'auto',
                    minHeight: '200px', maxHeight: '250px', color: '#38bdf8', display: 'flex', flexDirection: 'column', gap: '4px'
                  }}>
                    {fwLogs.length === 0 ? (
                      <div style={{ color: '#64748b', fontStyle: 'italic', textAlign: 'center', marginTop: '4rem' }}>
                        Ready to simulate transmission. Choose mode above and click "Run Transmission Audit".
                      </div>
                    ) : (
                      fwLogs.map((log, idx) => (
                        <div key={idx} style={{
                          color: log.includes('ERROR') || log.includes('WARNING') || log.includes('CRITICAL') ? '#ef4444' : log.includes('PASS') || log.includes('cleared') ? '#10b981' : '#38bdf8'
                        }}>
                          {log}
                        </div>
                      ))
                    )}
                  </div>

                  <div style={{ marginTop: '1rem', display: 'flex', gap: '0.8rem' }}>
                    <button 
                      className="btn btn-primary"
                      style={{ flex: 1, padding: '10px', fontSize: '0.8rem', background: '#38bdf8', color: '#0f172a', fontWeight: 'bold' }}
                      onClick={runSecurityAudit}
                      disabled={fwStatus === 'scanning'}
                    >
                      🛡️ Run Transmission Audit
                    </button>
                    <button 
                      className="btn btn-outline"
                      style={{ padding: '10px 16px', fontSize: '0.8rem', color: '#94a3b8', borderColor: '#334155' }}
                      onClick={() => { setFwStatus('idle'); setFwLogs([]); }}
                    >
                      Clear Logs
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Banner/Helper Info */}
            <div style={{ marginTop: '1.2rem', background: '#1e293b', padding: '12px', borderRadius: '8px', fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '1.5rem' }}>💡</span>
              <div>
                <strong>How the Federated Firewall protects patient records:</strong> The local hospital node uses NLP to extract features locally. In <strong>Federated Mode</strong>, only anonymous 64-dimensional float embeddings are transmitted. If you switch to <strong>Direct Mode</strong>, the firewall intercepts the raw data block immediately and triggers a HIPAA security alarm.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
