import { useState, useEffect } from 'react'
import Plot from '../PlotlyChart'

export default function SpatialGraph() {
  const [data, setData] = useState(null)
  const [timeIdx, setTimeIdx] = useState(0)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)
  const [nodeDetail, setNodeDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const fetchGraph = async (t) => {
    setLoading(true)
    try {
      const res = await fetch(`/api/graph?t=${t}`)
      const json = await res.json()
      setData(json)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  useEffect(() => { fetchGraph(-1) }, [])

  useEffect(() => {
    if (data && data.total_windows) setTimeIdx(data.t_idx)
  }, [data?.total_windows])

  const handleSlider = (e) => {
    const val = parseInt(e.target.value)
    setTimeIdx(val)
    fetchGraph(val)
  }

  const handleNodeClick = async (event) => {
    const pointIndex = event.points?.[0]?.pointIndex
    if (pointIndex === undefined || !data) return
    const node = data.nodes[pointIndex]
    if (!node) return
    setSelected(node)
    setDetailLoading(true)
    try {
      const res = await fetch(`/api/district-node/${node.censuscode}`)
      const json = await res.json()
      setNodeDetail(json)
    } catch (e) { console.error(e) }
    setDetailLoading(false)
  }

  if (loading && !data) return (
    <div className="page"><div className="loading"><div className="spinner" /><div className="loading-text">Loading spatial graph from trained model...</div></div></div>
  )

  const nodes = data?.nodes || []
  const edges = data?.edges || []

  // Build Plotly edge traces — thick and very visible
  const edgeTraces = []
  edges.forEach(e => {
    const s = nodes.find(n => n.id === e.source)
    const t = nodes.find(n => n.id === e.target)
    if (!s || !t) return
    const width = 0.8 + e.weight_norm * 3
    const opacity = 0.2 + e.weight_norm * 0.5
    edgeTraces.push({
      type: 'scattergeo',
      lon: [s.lon, t.lon], lat: [s.lat, t.lat],
      mode: 'lines',
      line: { width, color: `rgba(37, 99, 235, ${opacity})` },
      hoverinfo: 'text',
      text: `${s.name} ↔ ${t.name}<br>Border: ${e.weight_km} km`,
      showlegend: false,
    })
  })

  const probs = nodes.map(n => n.prob)
  const maxProb = Math.max(...probs, 0.01)
  const sizes = nodes.map(n => 5 + n.prob * 22)
  const highRisk = nodes.filter(n => n.prob > 0.3).sort((a, b) => b.prob - a.prob)

  const d = nodeDetail?.district

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Spatial Intelligence Graph</h1>
        <p className="page-subtitle">
          {nodes.length} districts · {edges.length} border connections · Click any node for detailed inference
        </p>
      </div>

      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-value">{nodes.length}</div>
          <div className="metric-label">Graph Nodes</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{edges.length}</div>
          <div className="metric-label">Spatial Edges</div>
        </div>
        <div className="metric-card">
          <div className="metric-value danger">{highRisk.length}</div>
          <div className="metric-label">High-Risk Alerts</div>
        </div>
        <div className="metric-card">
          <div className="metric-value success">{data?.year} W{data?.week}</div>
          <div className="metric-label">Current Window</div>
        </div>
      </div>

      {data && (
        <div className="card" style={{ marginBottom: '1.5rem', padding: '0.8rem 1.2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <span className="card-title" style={{ whiteSpace: 'nowrap' }}>Time Window</span>
            <input type="range" min={0} max={(data.total_windows || 100) - 1}
              value={timeIdx} onChange={handleSlider} style={{ flex: 1 }} />
            <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--blue-600)', whiteSpace: 'nowrap' }}>
              {data.year} Week {data.week}
            </span>
          </div>
        </div>
      )}

      <div className="grid-main">
        <div className="card" style={{ padding: '0.5rem' }}>
          <Plot
            data={[
              ...edgeTraces,
              {
                type: 'scattergeo',
                lon: nodes.map(n => n.lon),
                lat: nodes.map(n => n.lat),
                mode: 'markers',
                marker: {
                  size: sizes,
                  color: probs,
                  colorscale: [[0, '#10b981'], [0.2, '#34d399'], [0.4, '#fbbf24'], [0.6, '#f97316'], [1, '#ef4444']],
                  cmin: 0, cmax: Math.max(maxProb, 0.5),
                  colorbar: {
                    title: { text: 'Outbreak<br>Probability', font: { size: 10, color: '#64748b', family: 'DM Sans' } },
                    tickfont: { size: 9, color: '#94a3b8' },
                    thickness: 14, len: 0.5,
                    bgcolor: 'rgba(255,255,255,0.9)', borderwidth: 0, outlinewidth: 0,
                  },
                  line: { width: 1, color: 'rgba(255,255,255,0.8)' },
                },
                text: nodes.map(n =>
                  `<b>${n.name}</b>, ${n.state}<br>Risk: ${(n.prob * 100).toFixed(1)}%${n.truth ? '<br>⚠️ TRUE OUTBREAK' : ''}`
                ),
                hoverinfo: 'text',
                showlegend: false,
              },
            ]}
            layout={{
              geo: {
                scope: 'asia', projection: { type: 'mercator' },
                center: { lat: 22, lon: 80 },
                lonaxis: { range: [68, 98] }, lataxis: { range: [6, 36] },
                bgcolor: '#f8fafc',
                landcolor: '#f1f5f9',
                subunitcolor: '#e2e8f0',
                countrycolor: '#cbd5e1',
                coastlinecolor: '#94a3b8',
                showland: true, showocean: true,
                oceancolor: '#eff6ff',
                showlakes: false,
                framecolor: '#e2e8f0', framewidth: 1,
              },
              paper_bgcolor: '#ffffff',
              plot_bgcolor: '#ffffff',
              margin: { l: 0, r: 0, t: 10, b: 10 },
              height: 580,
              font: { family: 'DM Sans, sans-serif', color: '#334155' },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
            onClick={handleNodeClick}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Node detail panel — shows when a node is clicked */}
          {selected && (
            <div className="card" style={{ borderColor: 'var(--blue-400)' }}>
              <div className="card-title">Selected District</div>
              <h3 className="serif" style={{ fontSize: '1.4rem', marginTop: '0.3rem' }}>{selected.name}</h3>
              <div style={{ color: 'var(--slate-500)', fontSize: '0.82rem' }}>{selected.state} · Code {selected.censuscode}</div>
              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <div className="gauge-val" style={{ color: selected.prob > 0.5 ? 'var(--red-500)' : selected.prob > 0.1 ? '#d97706' : 'var(--emerald-500)' }}>
                  {(selected.prob * 100).toFixed(1)}%
                </div>
                <div className="metric-label">Outbreak Probability</div>
                <div className="prob-bar" style={{ marginTop: '0.5rem' }}>
                  <div className="prob-bar-fill" style={{
                    width: `${selected.prob * 100}%`,
                    background: selected.prob > 0.5 ? 'linear-gradient(90deg, #f97316, #ef4444)' : selected.prob > 0.1 ? 'linear-gradient(90deg, #fbbf24, #f97316)' : 'linear-gradient(90deg, #10b981, #34d399)'
                  }} />
                </div>
                {selected.truth === 1 && (
                  <div className="alert-box alert-risk" style={{ marginTop: '0.8rem', justifyContent: 'center' }}>
                    ⚠️ Confirmed outbreak this week
                  </div>
                )}
              </div>

              {detailLoading && <div className="loading" style={{ padding: '1rem' }}><div className="spinner" /></div>}

              {d && !detailLoading && (
                <div style={{ marginTop: '1rem' }}>
                  <div className="card-title" style={{ marginBottom: '0.5rem' }}>Model Internals</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--slate-500)', marginBottom: '0.3rem' }}>Client Embedding (32-dim)</div>
                  <div className="embedding-grid">
                    {d.client_embedding?.slice(0, 8).map((v, i) => {
                      const intensity = Math.min(Math.abs(v) * 3, 1)
                      const bg = v >= 0 ? `rgba(37, 99, 235, ${intensity * 0.9})` : `rgba(239, 68, 68, ${intensity * 0.9})`
                      return <div key={i} className="embedding-cell" style={{ background: bg }} title={`[${i}]=${v}`}>{v.toFixed(1)}</div>
                    })}
                  </div>
                  <div style={{ marginTop: '0.8rem', fontSize: '0.75rem', color: 'var(--slate-500)' }}>
                    Neighbors: <span style={{ color: 'var(--slate-700)' }}>{nodeDetail.neighbors?.length || 0}</span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--slate-400)', marginTop: '0.3rem' }}>
                    {nodeDetail.neighbors?.slice(0, 5).map(nb => nb.name).join(', ')}
                    {(nodeDetail.neighbors?.length || 0) > 5 && '...'}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="card">
            <div className="card-title" style={{ marginBottom: '0.8rem' }}>
              🔴 High-Risk Districts ({highRisk.length})
            </div>
            <div className="table-container" style={{ maxHeight: '320px' }}>
              <table>
                <thead><tr><th>District</th><th>Risk</th></tr></thead>
                <tbody>
                  {highRisk.slice(0, 20).map(n => (
                    <tr key={n.id} onClick={() => { setSelected(n); fetch(`/api/district-node/${n.censuscode}`).then(r => r.json()).then(setNodeDetail) }}>
                      <td>
                        <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{n.name}</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--slate-400)' }}>{n.state}</div>
                      </td>
                      <td>
                        <span className={n.prob > 0.5 ? 'badge badge-red' : 'badge badge-amber'}>
                          {(n.prob * 100).toFixed(1)}%
                        </span>
                        {n.truth === 1 && <span className="badge badge-red" style={{ marginLeft: 4 }}>TRUE</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <div className="card-title" style={{ marginBottom: '0.5rem' }}>Graph Info</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--slate-600)', lineHeight: 1.9 }}>
              <div><span className="mono" style={{ color: 'var(--blue-600)' }}>{nodes.length}</span> district nodes</div>
              <div><span className="mono" style={{ color: 'var(--blue-600)' }}>{edges.length}</span> undirected edges</div>
              <div>Weighted by <span style={{ fontWeight: 600 }}>shared border length (km)</span></div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--slate-400)' }}>
                Spatial DGAT (4-head) learns cross-district disease propagation via these border-weighted connections.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
