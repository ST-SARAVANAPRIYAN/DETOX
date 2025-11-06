import React, { useState } from 'react';
import './ProductionAPI.css';

const ProductionAPI = () => {
  const [singleText, setSingleText] = useState('');
  const [batchTexts, setBatchTexts] = useState('');
  const [singleResult, setSingleResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStats, setApiStats] = useState(null);

  // Fetch API stats on component mount
  React.useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/v1/stats');
      const data = await response.json();
      setApiStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const predictSingle = async () => {
    if (!singleText.trim()) {
      alert('Please enter some text');
      return;
    }

    setLoading(true);
    setSingleResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: singleText }),
      });

      const data = await response.json();
      setSingleResult(data);
    } catch (error) {
      setSingleResult({ error: error.message, success: false });
    } finally {
      setLoading(false);
    }
  };

  const predictBatch = async () => {
    if (!batchTexts.trim()) {
      alert('Please enter texts (one per line)');
      return;
    }

    setLoading(true);
    setBatchResult(null);

    const texts = batchTexts.split('\n').filter(t => t.trim());

    try {
      const response = await fetch('http://localhost:5000/api/v1/predict/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts }),
      });

      const data = await response.json();
      setBatchResult(data);
    } catch (error) {
      setBatchResult({ error: error.message, success: false });
    } finally {
      setLoading(false);
    }
  };

  const getLevelColor = (level) => {
    const colors = {
      'VERY_HIGH': '#ef4444',
      'HIGH': '#f59e0b',
      'MODERATE': '#fbbf24',
      'LOW': '#10b981',
      'MINIMAL': '#6366f1'
    };
    return colors[level] || '#6b7280';
  };

  return (
    <div className="production-api-container">
      <div className="api-header">
        <h1>🚀 Production API Testing</h1>
        <p>Fast, scalable toxicity detection with model caching and rate limiting</p>
      </div>

      {/* API Stats Card */}
      {apiStats && (
        <div className="stats-card">
          <h3>📊 API Information</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">API Version:</span>
              <span className="stat-value">{apiStats.api_version}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Model Loaded:</span>
              <span className={`stat-value ${apiStats.model_stats?.model_loaded ? 'success' : 'error'}`}>
                {apiStats.model_stats?.model_loaded ? '✓ Yes' : '✗ No'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Spark Active:</span>
              <span className={`stat-value ${apiStats.model_stats?.spark_active ? 'success' : 'error'}`}>
                {apiStats.model_stats?.spark_active ? '✓ Yes' : '✗ No'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Max Text Length:</span>
              <span className="stat-value">{apiStats.max_text_length} chars</span>
            </div>
          </div>
          <div className="rate-limits">
            <h4>⏱️ Rate Limits:</h4>
            <ul>
              <li><strong>Single Prediction:</strong> {apiStats.rate_limits?.predict_single}</li>
              <li><strong>Batch Prediction:</strong> {apiStats.rate_limits?.predict_batch}</li>
            </ul>
          </div>
        </div>
      )}

      <div className="api-sections">
        {/* Single Prediction Section */}
        <div className="api-section">
          <h2>💬 Single Message Prediction</h2>
          <p className="section-desc">Test individual messages with ultra-low latency</p>
          
          <div className="input-group">
            <label>Enter Message:</label>
            <textarea
              value={singleText}
              onChange={(e) => setSingleText(e.target.value)}
              placeholder="Type a message to check for toxicity..."
              rows={4}
            />
          </div>

          <div className="example-texts">
            <p>Quick examples:</p>
            <button onClick={() => setSingleText("You're an amazing person!")}>Positive</button>
            <button onClick={() => setSingleText("I disagree with your opinion")}>Neutral</button>
            <button onClick={() => setSingleText("You idiot, shut up!")}>Toxic</button>
          </div>

          <button 
            className="predict-btn primary" 
            onClick={predictSingle}
            disabled={loading}
          >
            {loading ? '⏳ Predicting...' : '🔍 Analyze'}
          </button>

          {singleResult && (
            <div className={`result-card ${singleResult.success ? 'success' : 'error'}`}>
              {singleResult.success ? (
                <>
                  <div className="result-header">
                    <h3>{singleResult.prediction}</h3>
                    <span 
                      className="level-badge"
                      style={{ backgroundColor: getLevelColor(singleResult.level) }}
                    >
                      {singleResult.level}
                    </span>
                  </div>
                  <div className="result-details">
                    <div className="detail-item">
                      <span className="detail-label">Toxicity Score:</span>
                      <span className="detail-value">{singleResult.toxicity_score}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Response Time:</span>
                      <span className="detail-value">{singleResult.latency_ms} ms</span>
                    </div>
                  </div>
                  <div className="performance-indicator">
                    {singleResult.latency_ms < 100 && <span className="fast">⚡ Lightning Fast!</span>}
                    {singleResult.latency_ms >= 100 && singleResult.latency_ms < 500 && <span className="good">✓ Good</span>}
                    {singleResult.latency_ms >= 500 && <span className="slow">⚠️ Slow</span>}
                  </div>
                </>
              ) : (
                <div className="error-message">
                  <h4>❌ Error</h4>
                  <p>{singleResult.error || singleResult.message}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Batch Prediction Section */}
        <div className="api-section">
          <h2>📦 Batch Prediction</h2>
          <p className="section-desc">Analyze multiple messages at once (much faster than individual requests)</p>
          
          <div className="input-group">
            <label>Enter Messages (one per line, max 100):</label>
            <textarea
              value={batchTexts}
              onChange={(e) => setBatchTexts(e.target.value)}
              placeholder="Message 1&#10;Message 2&#10;Message 3&#10;..."
              rows={8}
            />
          </div>

          <button 
            className="predict-btn secondary" 
            onClick={predictBatch}
            disabled={loading}
          >
            {loading ? '⏳ Processing Batch...' : '🔍 Analyze Batch'}
          </button>

          {batchResult && (
            <div className={`result-card ${batchResult.success ? 'success' : 'error'}`}>
              {batchResult.success ? (
                <>
                  <div className="batch-header">
                    <h3>✅ Batch Results</h3>
                    <div className="batch-stats">
                      <span><strong>{batchResult.total}</strong> messages</span>
                      <span><strong>{batchResult.total_latency_ms}</strong> ms total</span>
                      <span><strong>{batchResult.avg_latency_ms}</strong> ms average</span>
                    </div>
                  </div>

                  <div className="batch-table-wrapper">
                    <table className="batch-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Text</th>
                          <th>Prediction</th>
                          <th>Score</th>
                          <th>Level</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResult.predictions.map((pred, idx) => (
                          <tr key={idx}>
                            <td>{idx + 1}</td>
                            <td className="text-cell">{pred.text}</td>
                            <td className={pred.prediction === 'Toxic' ? 'toxic' : 'safe'}>
                              {pred.prediction}
                            </td>
                            <td>{pred.toxicity_score}</td>
                            <td>
                              <span 
                                className="level-badge-small"
                                style={{ backgroundColor: getLevelColor(pred.level) }}
                              >
                                {pred.level}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="performance-summary">
                    <p>
                      ⚡ Processed {batchResult.total} messages in {batchResult.total_latency_ms}ms
                      ({batchResult.avg_latency_ms}ms per message)
                    </p>
                    {batchResult.avg_latency_ms < 50 && (
                      <p className="highlight">🚀 Excellent performance! Ready for production.</p>
                    )}
                  </div>
                </>
              ) : (
                <div className="error-message">
                  <h4>❌ Error</h4>
                  <p>{batchResult.error || batchResult.message}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* API Documentation */}
      <div className="api-docs">
        <h2>📚 API Documentation</h2>
        <div className="endpoint-docs">
          <div className="endpoint-card">
            <h3>POST /api/v1/predict</h3>
            <p className="endpoint-desc">Predict toxicity for a single message</p>
            <div className="code-block">
              <code>
                {`curl -X POST http://localhost:5000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your message here"}'`}
              </code>
            </div>
          </div>

          <div className="endpoint-card">
            <h3>POST /api/v1/predict/batch</h3>
            <p className="endpoint-desc">Predict toxicity for multiple messages</p>
            <div className="code-block">
              <code>
                {`curl -X POST http://localhost:5000/api/v1/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Message 1", "Message 2"]}'`}
              </code>
            </div>
          </div>

          <div className="endpoint-card">
            <h3>GET /api/v1/health</h3>
            <p className="endpoint-desc">Check API health status</p>
            <div className="code-block">
              <code>curl http://localhost:5000/api/v1/health</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductionAPI;
