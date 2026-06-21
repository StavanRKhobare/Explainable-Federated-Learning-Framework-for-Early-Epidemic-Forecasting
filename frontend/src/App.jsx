import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import SpatialGraph from './pages/SpatialGraph'
import FederatedDemo from './pages/FederatedDemo'
import LivePredict from './pages/LivePredict'
import MultiNodeSimulation from './pages/MultiNodeSimulation'

function Navbar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="nav-brand">
        <div className="dot" />
        FedXGNN <span style={{ color: 'var(--blue-400)', fontWeight: 400 }}>Intelligence</span>
      </NavLink>
      <div className="nav-links">
        <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Spatial Graph
        </NavLink>
        <NavLink to="/federated" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Federated Demo
        </NavLink>
        <NavLink to="/predict" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Live Inference
        </NavLink>
        <NavLink to="/simulation" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Edge Simulation
        </NavLink>
      </div>
      <div className="nav-status">
        <div className="pulse-dot" />
        MODEL LOADED · 284 DISTRICTS
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<SpatialGraph />} />
        <Route path="/federated" element={<FederatedDemo />} />
        <Route path="/predict" element={<LivePredict />} />
        <Route path="/simulation" element={<MultiNodeSimulation />} />
      </Routes>
    </BrowserRouter>
  )
}
