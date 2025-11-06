import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Sparkles, BarChart3, Users, Shield, ArrowRight } from 'lucide-react'
import './Welcome.css'

const Welcome = () => {
  const navigate = useNavigate()

  const features = [
    {
      icon: <Sparkles size={40} />,
      title: "Apache Spark MLlib",
      description: "Distributed machine learning with PySpark 3.5.3"
    },
    {
      icon: <BarChart3 size={40} />,
      title: "94.42% Accuracy",
      description: "High-performance toxicity detection model"
    },
    {
      icon: <Users size={40} />,
      title: "User-Level Analysis",
      description: "Aggregate toxicity scores per user with severity levels"
    },
    {
      icon: <Shield size={40} />,
      title: "Real-time Monitoring",
      description: "Integrated Spark Web UI for live job tracking"
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
            <span>PySpark + Machine Learning</span>
          </div>
          
          <h1 className="hero-title">
            <span className="gradient-text">DETOX</span>
            <br />
            Chat Message Toxicity Detector
          </h1>
          
          <p className="hero-subtitle">
            An intelligent PySpark-powered system that analyzes chat messages to detect toxic content,
            leveraging Apache Spark MLlib's Logistic Regression with TF-IDF features.
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
              <span className="stat-number">26.08s</span>
              <span className="stat-label">Execution Time</span>
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

        {/* Features Section */}
        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.8 }}
        >
          <h2 className="section-title">Key Features</h2>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + index * 0.1, duration: 0.5 }}
                whileHover={{ y: -10 }}
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
          transition={{ delay: 0.8, duration: 0.8 }}
        >
          <h2 className="section-title">Tech Stack</h2>
          
          <div className="tech-stack">
            <div className="tech-item">
              <span className="tech-badge">Apache Spark 3.5.3</span>
            </div>
            <div className="tech-item">
              <span className="tech-badge">PySpark MLlib</span>
            </div>
            <div className="tech-item">
              <span className="tech-badge">Python 3.13</span>
            </div>
            <div className="tech-item">
              <span className="tech-badge">Logistic Regression</span>
            </div>
            <div className="tech-item">
              <span className="tech-badge">TF-IDF Features</span>
            </div>
            <div className="tech-item">
              <span className="tech-badge">Flask + React</span>
            </div>
          </div>
        </motion.div>

        {/* Dataset Info */}
        <motion.div
          className="dataset-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0, duration: 0.8 }}
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
