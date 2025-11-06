import { useState } from 'react'
import { ExternalLink, Activity, AlertCircle } from 'lucide-react'
import './SparkMonitor.css'

const SparkMonitor = () => {
  const [sparkAvailable, setSparkAvailable] = useState(true)
  const sparkUrl = 'http://localhost:4040'

  return (
    <div className="spark-monitor">
      <div className="spark-header">
        <div className="spark-title">
          <Activity size={24} />
          <h2>Apache Spark Web UI</h2>
        </div>
        
        <a 
          href={sparkUrl} 
          target="_blank" 
          rel="noopener noreferrer"
          className="btn-external"
        >
          <ExternalLink size={18} />
          Open in New Tab
        </a>
      </div>

      <div className="spark-info">
        <div className="info-item">
          <span className="info-label">URL:</span>
          <span className="info-value">
            <a href={sparkUrl} target="_blank" rel="noopener noreferrer">
              {sparkUrl}
            </a>
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Port:</span>
          <span className="info-value">4040</span>
        </div>
        <div className="info-item">
          <span className="info-label">Status:</span>
          <span className={`status-badge ${sparkAvailable ? 'active' : 'inactive'}`}>
            {sparkAvailable ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      <div className="spark-description">
        <h3>What You Can Monitor:</h3>
        <ul>
          <li><strong>Jobs:</strong> Track Spark job execution progress and timing</li>
          <li><strong>Stages:</strong> View detailed stage information and task distribution</li>
          <li><strong>Storage:</strong> Monitor RDD/DataFrame caching and memory usage</li>
          <li><strong>Environment:</strong> Check Spark configuration and system properties</li>
          <li><strong>Executors:</strong> View executor status, memory, and task metrics</li>
          <li><strong>SQL:</strong> Analyze query plans and execution details</li>
        </ul>
      </div>

      <div className="spark-iframe-container">
        {sparkAvailable ? (
          <iframe
            src={sparkUrl}
            title="Spark Web UI"
            className="spark-iframe"
            onError={() => setSparkAvailable(false)}
          />
        ) : (
          <div className="spark-unavailable">
            <AlertCircle size={48} />
            <h3>Spark Web UI Not Available</h3>
            <p>
              The Spark Web UI is only available when a Spark application is running.
              Please run the pipeline to start a Spark session.
            </p>
            <div className="instructions">
              <h4>To Access Spark UI:</h4>
              <ol>
                <li>Click "Run All" or execute any step in the pipeline</li>
                <li>Wait for Spark session to initialize</li>
                <li>The UI will be available at <code>http://localhost:4040</code></li>
                <li>Refresh this page or click "Open in New Tab"</li>
              </ol>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default SparkMonitor
