import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Sparkles, 
  BarChart3, 
  Users, 
  Shield, 
  ArrowRight,
  Database,
  FileText,
  Filter,
  Settings,
  Brain,
  CheckCircle,
  TrendingUp,
  Save,
  Zap
} from 'lucide-react'
import './Welcome.css'

const Welcome = () => {
  const navigate = useNavigate()

  const pipelineSteps = [
    {
      step: 1,
      icon: <Database size={32} />,
      title: "Data Loading",
      description: "Load 589K+ records from CSV",
      tech: "PySpark DataFrame",
      color: "#3b82f6"
    },
    {
      step: 2,
      icon: <CheckCircle size={32} />,
      title: "Data Validation",
      description: "Validate & clean records",
      tech: "Schema Validation",
      color: "#8b5cf6"
    },
    {
      step: 3,
      icon: <FileText size={32} />,
      title: "Text Preprocessing",
      description: "Clean URLs, special chars, lowercase",
      tech: "UDF + Regex",
      color: "#ec4899"
    },
    {
      step: 4,
      icon: <Filter size={32} />,
      title: "Feature Engineering",
      description: "Tokenize → Stop Words → TF-IDF + Lexicon",
      tech: "HashingTF + IDF + 7 Lexicon Features",
      color: "#f59e0b"
    },
    {
      step: 5,
      icon: <Settings size={32} />,
      title: "Class Balancing",
      description: "Weight toxic samples 12.4x",
      tech: "Class Weights",
      color: "#10b981"
    },
    {
      step: 6,
      icon: <Brain size={32} />,
      title: "Model Training",
      description: "Train LogisticRegression with 10,007 features",
      tech: "Spark MLlib + PipelineModel",
      color: "#06b6d4"
    },
    {
      step: 7,
      icon: <TrendingUp size={32} />,
      title: "Model Evaluation",
      description: "Achieve 90.19% accuracy",
      tech: "Binary Classification Metrics",
      color: "#6366f1"
    },
    {
      step: 8,
      icon: <Zap size={32} />,
      title: "Predictions",
      description: "Generate toxicity scores for all messages",
      tech: "Batch Transform",
      color: "#8b5cf6"
    },
    {
      step: 9,
      icon: <Save size={32} />,
      title: "Export Results",
      description: "Save model + predictions",
      tech: "CSV + PipelineModel",
      color: "#10b981"
    }
  ]

  const features = [
    {
      icon: <Sparkles size={40} />,
      title: "Lexicon + TF-IDF",
      description: "Combined statistical & rule-based detection (10,007 features)"
    },
    {
      icon: <BarChart3 size={40} />,
      title: "90.19% Accuracy",
      description: "High-performance toxicity detection with class balancing"
    },
    {
      icon: <Users size={40} />,
      title: "User-Level Analysis",
      description: "183K+ users analyzed with severity levels"
    },
    {
      icon: <Shield size={40} />,
      title: "Real-time API",
      description: "Live predictions with <1s latency + model caching"
    }
  ]

  return (
    <div className="welcome-page">
      <div className="welcome-container">
        {/* Hero Section */}
        <motion.div
          className="hero-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="hero-badge">
            <Sparkles size={20} />
            <span>PySpark + ML + Lexicon Features</span>
          </div>
          
          <h1 className="hero-title">
            <span className="gradient-text">DETOX</span>
            <br />
            Chat Message Toxicity Detector
          </h1>
          
          <p className="hero-subtitle">
            An intelligent PySpark-powered system that analyzes chat messages to detect toxic content,
            leveraging Apache Spark MLlib's Logistic Regression with TF-IDF + Lexicon features.
          </p>

          <div className="hero-stats">
            <div className="stat-item">
              <span className="stat-number">589K+</span>
              <span className="stat-label">Messages Analyzed</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">183K+</span>
              <span className="stat-label">Users Evaluated</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">90.19%</span>
              <span className="stat-label">Model Accuracy</span>
            </div>
          </div>

          <motion.button
            className="btn btn-hero"
            onClick={() => navigate('/demo')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Start Interactive Demo
            <ArrowRight size={20} />
          </motion.button>
        </motion.div>

        {/* Pipeline Flow Diagram - Horizontal Timeline */}
        <motion.div
          className="pipeline-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          <h2 className="section-title">
            <Zap size={28} className="title-icon" />
            9-Step ML Pipeline Architecture
          </h2>
          <p className="section-subtitle">End-to-end toxicity detection workflow</p>
          
          <div className="pipeline-timeline">
            {pipelineSteps.map((step, index) => (
              <motion.div
                key={step.step}
                className="timeline-item"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ 
                  delay: 0.5 + index * 0.1, 
                  duration: 0.6,
                  type: "spring",
                  stiffness: 100
                }}
                whileHover={{ 
                  scale: 1.05,
                  y: -10,
                  transition: { duration: 0.2 }
                }}
              >
                <div className="timeline-card" style={{ borderTopColor: step.color }}>
                  <div className="timeline-step-number" style={{ backgroundColor: step.color }}>
                    {step.step}
                  </div>
                  
                  <div className="timeline-icon" style={{ color: step.color }}>
                    {step.icon}
                  </div>
                  
                  <h3 className="timeline-title">{step.title}</h3>
                  <p className="timeline-description">{step.description}</p>
                  
                  <div className="timeline-tech" style={{ backgroundColor: `${step.color}15` }}>
                    <span style={{ color: step.color }}>⚡</span>
                    {step.tech}
                  </div>
                </div>
                
                {index < pipelineSteps.length - 1 && (
                  <motion.div 
                    className="timeline-connector"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ delay: 0.5 + index * 0.1 + 0.3, duration: 0.4 }}
                  >
                    <ArrowRight size={24} className="connector-arrow" />
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Features Section */}
        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.8 }}
        >
          <h2 className="section-title">Key Features</h2>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 + index * 0.1, duration: 0.5 }}
                whileHover={{ y: -10, scale: 1.02 }}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack Section */}
        <motion.div
          className="tech-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.6, duration: 0.8 }}
        >
          <h2 className="section-title">Tech Stack</h2>
          
          <div className="tech-stack">
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: 5 }}>
              <span className="tech-badge">Apache Spark 3.5.3</span>
            </motion.div>
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: -5 }}>
              <span className="tech-badge">PySpark MLlib</span>
            </motion.div>
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: 5 }}>
              <span className="tech-badge">Python 3.13</span>
            </motion.div>
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: -5 }}>
              <span className="tech-badge">Logistic Regression</span>
            </motion.div>
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: 5 }}>
              <span className="tech-badge">TF-IDF + Lexicon</span>
            </motion.div>
            <motion.div className="tech-item" whileHover={{ scale: 1.1, rotate: -5 }}>
              <span className="tech-badge">Flask + React</span>
            </motion.div>
          </div>
        </motion.div>

        {/* Dataset Info */}
        <motion.div
          className="dataset-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.8, duration: 0.8 }}
        >
          <div className="dataset-card">
            <h3>Dataset: Jigsaw Toxic Comments</h3>
            <div className="dataset-details">
              <div className="dataset-item">
                <span className="dataset-label">Raw Records:</span>
                <span className="dataset-value">589,087</span>
              </div>
              <div className="dataset-item">
                <span className="dataset-label">Validated Records:</span>
                <span className="dataset-value">321,854</span>
              </div>
              <div className="dataset-item">
                <span className="dataset-label">Toxic Messages:</span>
                <span className="dataset-value">32,186 (10%)</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Welcome
