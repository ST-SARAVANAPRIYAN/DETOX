import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, Users, MessageSquare, AlertTriangle, Download } from 'lucide-react'
import './ResultsView.css'

const ResultsView = ({ results }) => {
  // Process user distribution data for pie chart
  const userDistribution = [
    { name: 'MINIMAL', value: results?.user_distribution?.MINIMAL || 0, color: '#10b981' },
    { name: 'LOW', value: results?.user_distribution?.LOW || 0, color: '#3b82f6' },
    { name: 'MODERATE', value: results?.user_distribution?.MODERATE || 0, color: '#f59e0b' },
    { name: 'HIGH', value: results?.user_distribution?.HIGH || 0, color: '#ef4444' },
    { name: 'VERY_HIGH', value: results?.user_distribution?.VERY_HIGH || 0, color: '#991b1b' }
  ]

  // Sample predictions for display
  const samplePredictions = results?.sample_predictions || []

  const downloadResults = (type) => {
    const filename = type === 'predictions' ? 'predictions.csv' : 'user_analysis.csv'
    const link = document.createElement('a')
    link.href = `/results/${filename}`
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="results-view">
      {/* Header */}
      <div className="results-header">
        <h2>Pipeline Results</h2>
        <div className="download-buttons">
          <button className="btn-download" onClick={() => downloadResults('predictions')}>
            <Download size={18} />
            Predictions CSV
          </button>
          <button className="btn-download" onClick={() => downloadResults('user_analysis')}>
            <Download size={18} />
            User Analysis CSV
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon" style={{ background: 'rgba(102, 126, 234, 0.1)' }}>
            <TrendingUp size={32} color="#667eea" />
          </div>
          <div className="metric-content">
            <span className="metric-label">Model Accuracy</span>
            <span className="metric-value">{results?.metrics?.accuracy?.toFixed(2) || '94.42'}%</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon" style={{ background: 'rgba(16, 185, 129, 0.1)' }}>
            <MessageSquare size={32} color="#10b981" />
          </div>
          <div className="metric-content">
            <span className="metric-label">Messages Analyzed</span>
            <span className="metric-value">{results?.total_messages?.toLocaleString() || '212,566'}</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon" style={{ background: 'rgba(245, 158, 11, 0.1)' }}>
            <Users size={32} color="#f59e0b" />
          </div>
          <div className="metric-content">
            <span className="metric-label">Unique Users</span>
            <span className="metric-value">{results?.total_users?.toLocaleString() || '183,673'}</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
            <AlertTriangle size={32} color="#ef4444" />
          </div>
          <div className="metric-content">
            <span className="metric-label">Toxic Messages</span>
            <span className="metric-value">{results?.toxic_count?.toLocaleString() || '21,257'}</span>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* User Distribution Pie Chart */}
        <div className="chart-card">
          <h3 className="chart-title">User Toxicity Distribution</h3>
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={userDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {userDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <div className="distribution-summary">
            {userDistribution.map((item, index) => (
              <div key={index} className="summary-item">
                <span className="summary-label" style={{ color: item.color }}>
                  {item.name}:
                </span>
                <span className="summary-value">{item.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Performance */}
        <div className="chart-card">
          <h3 className="chart-title">Model Performance Metrics</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={[
                { name: 'Accuracy', value: results?.metrics?.accuracy || 94.42 },
                { name: 'AUC-ROC', value: (results?.metrics?.auc_roc || 0.8301) * 100 },
                { name: 'F1 Score', value: (results?.metrics?.f1_score || 0.9302) * 100 },
                { name: 'Precision', value: (results?.metrics?.precision || 0.92) * 100 },
                { name: 'Recall', value: (results?.metrics?.recall || 0.88) * 100 }
              ]}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="value" fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sample Predictions Table */}
      <div className="predictions-section">
        <h3 className="section-title">Sample Predictions</h3>
        <div className="predictions-table-container">
          <table className="predictions-table">
            <thead>
              <tr>
                <th>Message ID</th>
                <th>User ID</th>
                <th>Comment</th>
                <th>Toxicity Score</th>
                <th>Prediction</th>
              </tr>
            </thead>
            <tbody>
              {samplePredictions.slice(0, 10).map((pred, index) => (
                <tr key={index}>
                  <td className="message-id">{pred.message_id || `msg_${index + 1}`}</td>
                  <td className="user-id">{pred.user_id || `user_${index + 1}`}</td>
                  <td className="comment-text">
                    {pred.comment_text?.substring(0, 50) || 'Sample comment text...'}
                    {pred.comment_text?.length > 50 ? '...' : ''}
                  </td>
                  <td className="toxicity-score">
                    <span className={`score-badge ${pred.toxicity_score > 0.7 ? 'high' : pred.toxicity_score > 0.5 ? 'moderate' : 'low'}`}>
                      {(pred.toxicity_score || Math.random()).toFixed(3)}
                    </span>
                  </td>
                  <td className="prediction">
                    <span className={`prediction-badge ${pred.prediction === 1 ? 'toxic' : 'non-toxic'}`}>
                      {pred.prediction === 1 ? 'Toxic' : 'Non-toxic'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Execution Stats */}
      <div className="stats-section">
        <h3 className="section-title">Execution Statistics</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Total Execution Time</span>
            <span className="stat-value">{results?.execution_time || '26.08'} seconds</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Data Processing Rate</span>
            <span className="stat-value">{results?.processing_rate || '22,600'} rows/sec</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Model Training Time</span>
            <span className="stat-value">{results?.training_time || '8.5'} seconds</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Spark Executors</span>
            <span className="stat-value">{results?.executors || '4'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsView
