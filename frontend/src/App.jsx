import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Welcome from './pages/Welcome'
import Demo from './pages/Demo'
import Results from './pages/Results'
import ProductionAPI from './pages/ProductionAPI'
import LiveDetection from './pages/LiveDetection'
import './App.css'

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/results" element={<Results />} />
          <Route path="/production" element={<ProductionAPI />} />
          <Route path="/live" element={<LiveDetection />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
