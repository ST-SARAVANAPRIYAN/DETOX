import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import io from 'socket.io-client'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Home, Play, Pause, SkipForward, SkipBack, 
  Terminal, Activity, CheckCircle2, AlertCircle,
  Loader2, ExternalLink, BarChart3 
} from 'lucide-react'
import StepViewer from '../components/StepViewer'
import TerminalOutput from '../components/TerminalOutput'
import SparkMonitor from '../components/SparkMonitor'
import ResultsView from '../components/ResultsView'
import './Demo.css'

const Demo = () => {
  const navigate = useNavigate()
  const [socket, setSocket] = useState(null)
  const [projectInfo, setProjectInfo] = useState(null)
  const [steps, setSteps] = useState([])
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [stepStatuses, setStepStatuses] = useState({})
  const [terminalOutputs, setTerminalOutputs] = useState({})
  const [showResults, setShowResults] = useState(false)
  const [results, setResults] = useState(null)
  const [activeView, setActiveView] = useState('step') // step, terminal, spark, results

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io('http://localhost:5000')
    setSocket(newSocket)

    // Fetch project info and steps
    fetchProjectInfo()
    fetchSteps()

    // Socket event listeners
    newSocket.on('step_started', (data) => {
      setStepStatuses(prev => ({
        ...prev,
        [data.step_id]: 'running'
      }))
      setCurrentStepIndex(data.step_id - 1)
    })

    newSocket.on('step_progress', (data) => {
      setTerminalOutputs(prev => ({
        ...prev,
        [data.step_id]: [...(prev[data.step_id] || []), data.message]
      }))
    })

    newSocket.on('step_completed', (data) => {
      setStepStatuses(prev => ({
        ...prev,
        [data.step_id]: data.success ? 'completed' : 'failed'
      }))
      
      if (data.output) {
        setTerminalOutputs(prev => ({
          ...prev,
          [data.step_id]: [...(prev[data.step_id] || []), data.output]
        }))
      }
      
      // Reset isRunning when step completes
      setIsRunning(false)
    })

    newSocket.on('pipeline_completed', (data) => {
      setIsRunning(false)
      if (data.success) {
        fetchResults()
      }
    })

    return () => {
      newSocket.disconnect()
    }
  }, [])

  const fetchProjectInfo = async () => {
    try {
      const response = await axios.get('/api/project-info')
      setProjectInfo(response.data)
    } catch (error) {
      console.error('Error fetching project info:', error)
    }
  }

  const fetchSteps = async () => {
    try {
      const response = await axios.get('/api/pipeline-steps')
      setSteps(response.data.steps)
      
      // Initialize all steps as pending
      const initialStatuses = {}
      response.data.steps.forEach(step => {
        initialStatuses[step.id] = 'pending'
      })
      setStepStatuses(initialStatuses)
    } catch (error) {
      console.error('Error fetching steps:', error)
    }
  }

  const fetchResults = async () => {
    try {
      const response = await axios.get('/api/results')
      setResults(response.data)
      setShowResults(true)
    } catch (error) {
      console.error('Error fetching results:', error)
    }
  }

  const executeStep = async (stepId) => {
    if (isRunning) return
    
    setIsRunning(true)
    try {
      await axios.post(`/api/execute-step/${stepId}`)
    } catch (error) {
      console.error('Error executing step:', error)
      setIsRunning(false)
    }
  }

  const executeAllSteps = async () => {
    if (isRunning) return
    
    setIsRunning(true)
    try {
      await axios.post('/api/execute-all')
    } catch (error) {
      console.error('Error executing all steps:', error)
      setIsRunning(false)
    }
  }

  const goToNextStep = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStepIndex(prev => prev + 1)
    }
  }

  const goToPreviousStep = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1)
    }
  }

  const getStepStatus = (stepId) => {
    return stepStatuses[stepId] || 'pending'
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 size={20} className="status-icon completed" />
      case 'running':
        return <Loader2 size={20} className="status-icon running spin" />
      case 'failed':
        return <AlertCircle size={20} className="status-icon failed" />
      default:
        return <div className="status-dot pending" />
    }
  }

  const currentStep = steps[currentStepIndex]

  return (
    <div className="demo-page">
      {/* Header */}
      <div className="demo-header">
        <div className="header-left">
          <button className="btn-icon" onClick={() => navigate('/')}>
            <Home size={20} />
          </button>
          <h1 className="demo-title">Interactive Demo</h1>
        </div>

        <div className="header-controls">
          <button 
            className="btn-control"
            onClick={goToPreviousStep}
            disabled={currentStepIndex === 0 || isRunning}
          >
            <SkipBack size={20} />
            Previous
          </button>

          <button 
            className="btn-control primary"
            onClick={() => executeStep(currentStep?.id)}
            disabled={isRunning || !currentStep}
          >
            {isRunning ? <Loader2 size={20} className="spin" /> : <Play size={20} />}
            Run Step
          </button>

          <button 
            className="btn-control"
            onClick={goToNextStep}
            disabled={currentStepIndex === steps.length - 1 || isRunning}
          >
            Next
            <SkipForward size={20} />
          </button>

          <button 
            className="btn-control success"
            onClick={() => executeAllSteps()}
            disabled={isRunning}
          >
            {isRunning ? <Loader2 size={20} className="spin" /> : <Play size={20} />}
            Run All
          </button>

          <button 
            className="btn-control results-btn"
            onClick={() => navigate('/results')}
          >
            <BarChart3 size={20} />
            View Results
          </button>

          <button 
            className="btn-control live-btn"
            onClick={() => navigate('/live')}
          >
            ⚡ Live Detection
          </button>
        </div>
      </div>

      <div className="demo-container">
        {/* Sidebar - Steps List */}
        <div className="demo-sidebar">
          <div className="sidebar-header">
            <h3>Pipeline Steps</h3>
            <span className="step-counter">
              {currentStepIndex + 1} / {steps.length}
            </span>
          </div>

          <div className="steps-list">
            {steps.map((step, index) => (
              <motion.div
                key={step.id}
                className={`step-item ${currentStepIndex === index ? 'active' : ''}`}
                onClick={() => !isRunning && setCurrentStepIndex(index)}
                whileHover={{ x: 5 }}
              >
                <div className="step-indicator">
                  {getStatusIcon(getStepStatus(step.id))}
                </div>
                <div className="step-info">
                  <span className="step-number">Step {step.id}</span>
                  <span className="step-name">{step.name}</span>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Progress Bar */}
          <div className="progress-section">
            <div className="progress-label">Overall Progress</div>
            <div className="progress-bar">
              <motion.div 
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ 
                  width: `${(Object.values(stepStatuses).filter(s => s === 'completed').length / steps.length) * 100}%` 
                }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="demo-main">
          {/* View Tabs */}
          <div className="view-tabs">
            <button 
              className={`tab ${activeView === 'step' ? 'active' : ''}`}
              onClick={() => setActiveView('step')}
            >
              <Activity size={18} />
              Step Details
            </button>
            <button 
              className={`tab ${activeView === 'terminal' ? 'active' : ''}`}
              onClick={() => setActiveView('terminal')}
            >
              <Terminal size={18} />
              Terminal Output
            </button>
            <button 
              className={`tab ${activeView === 'spark' ? 'active' : ''}`}
              onClick={() => setActiveView('spark')}
            >
              <ExternalLink size={18} />
              Spark Monitor
            </button>
            {showResults && (
              <button 
                className={`tab ${activeView === 'results' ? 'active' : ''}`}
                onClick={() => setActiveView('results')}
              >
                <BarChart3 size={18} />
                Results
              </button>
            )}
          </div>

          {/* Content Views */}
          <div className="view-content">
            <AnimatePresence mode="wait">
              {activeView === 'step' && currentStep && (
                <motion.div
                  key="step-view"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <StepViewer 
                    step={currentStep} 
                    status={getStepStatus(currentStep.id)}
                  />
                </motion.div>
              )}

              {activeView === 'terminal' && (
                <motion.div
                  key="terminal-view"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <TerminalOutput 
                    outputs={terminalOutputs[currentStep?.id] || []}
                    isRunning={getStepStatus(currentStep?.id) === 'running'}
                  />
                </motion.div>
              )}

              {activeView === 'spark' && (
                <motion.div
                  key="spark-view"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <SparkMonitor />
                </motion.div>
              )}

              {activeView === 'results' && results && (
                <motion.div
                  key="results-view"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <ResultsView results={results} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Demo
