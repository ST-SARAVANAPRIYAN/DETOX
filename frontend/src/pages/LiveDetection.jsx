import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './LiveDetection.css';

const LiveDetection = () => {
  // Tab state
  const [activeTab, setActiveTab] = useState('manual'); // 'manual', 'telegram', 'csv'
  
  // Manual input states
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);
  const [debounceTimer, setDebounceTimer] = useState(null);
  
  // Telegram states
  const [telegramMessages, setTelegramMessages] = useState([]);
  const [telegramStats, setTelegramStats] = useState(null);
  const [telegramLoading, setTelegramLoading] = useState(false);
  const [botStatus, setBotStatus] = useState({ running: false });
  
  // CSV states
  const [csvFile, setCsvFile] = useState(null);
  const [csvResults, setCsvResults] = useState([]);
  const [csvProcessing, setCsvProcessing] = useState(false);
  const [csvProgress, setCsvProgress] = useState(0);

  // ============================================================
  // MANUAL INPUT FUNCTIONS
  // ============================================================
  
  const analyzeText = useCallback(async (text) => {
    if (!text.trim()) {
      setResult(null);
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      
      if (data.success) {
        setResult(data);
        
        setHistory(prev => [{
          text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
          result: data,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 9)]);
      } else {
        setResult({
          error: data.error || data.message || 'Analysis failed. Make sure the model is loaded.',
          success: false
        });
      }
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({
        error: 'Connection error. Make sure the backend is running.',
        success: false
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleTextChange = (text, immediate = false) => {
    setInputText(text);
    
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    
    if (immediate) {
      analyzeText(text);
      return;
    }
    
    const timer = setTimeout(() => {
      analyzeText(text);
    }, 300);
    
    setDebounceTimer(timer);
  };
  
  const handleAnalyzeClick = () => {
    if (inputText.trim()) {
      analyzeText(inputText);
    }
  };

  // ============================================================
  // TELEGRAM FUNCTIONS
  // ============================================================
  
  const fetchTelegramMessages = useCallback(async () => {
    try {
      setTelegramLoading(true);
      const response = await fetch('http://localhost:5000/api/telegram/messages?limit=50');
      const data = await response.json();
      
      if (data.success) {
        setTelegramMessages(data.messages);
      }
    } catch (error) {
      console.error('Error fetching Telegram messages:', error);
    } finally {
      setTelegramLoading(false);
    }
  }, []);

  const fetchTelegramStats = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5000/api/telegram/stats');
      const data = await response.json();
      
      if (data.success) {
        setTelegramStats(data.stats);
        setBotStatus(data.stats.bot_status || { running: false });
      }
    } catch (error) {
      console.error('Error fetching Telegram stats:', error);
    }
  }, []);

  const clearTelegramMessages = async () => {
    try {
      await fetch('http://localhost:5000/api/telegram/clear', { method: 'POST' });
      fetchTelegramMessages();
      fetchTelegramStats();
    } catch (error) {
      console.error('Error clearing messages:', error);
    }
  };

  const resetTelegramStats = async () => {
    try {
      await fetch('http://localhost:5000/api/telegram/reset-stats', { method: 'POST' });
      fetchTelegramStats();
    } catch (error) {
      console.error('Error resetting stats:', error);
    }
  };

  // Auto-refresh Telegram data every 5 seconds when on Telegram tab
  useEffect(() => {
    if (activeTab === 'telegram') {
      fetchTelegramMessages();
      fetchTelegramStats();
      
      const interval = setInterval(() => {
        fetchTelegramMessages();
        fetchTelegramStats();
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [activeTab, fetchTelegramMessages, fetchTelegramStats]);

  // ============================================================
  // CSV FUNCTIONS
  // ============================================================
  
  const handleCsvUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setCsvFile(file);
      setCsvResults([]);
    }
  };

  const processCsv = async () => {
    if (!csvFile) return;
    
    setCsvProcessing(true);
    setCsvProgress(0);
    
    try {
      const text = await csvFile.text();
      const lines = text.split('\n').filter(line => line.trim());
      const headers = lines[0].split(',');
      
      // Find text column (usually named 'text', 'message', 'comment', etc.)
      const textColIndex = headers.findIndex(h => 
        h.toLowerCase().includes('text') || 
        h.toLowerCase().includes('message') || 
        h.toLowerCase().includes('comment')
      );
      
      if (textColIndex === -1) {
        alert('Could not find text column in CSV. Expected column with "text", "message", or "comment" in name.');
        setCsvProcessing(false);
        return;
      }
      
      const messages = lines.slice(1).map(line => {
        const cols = line.split(',');
        return cols[textColIndex]?.replace(/^"|"$/g, '').trim();
      }).filter(Boolean);
      
      // Process in batches of 100
      const batchSize = 100;
      const results = [];
      
      for (let i = 0; i < messages.length; i += batchSize) {
        const batch = messages.slice(i, i + batchSize);
        
        const response = await fetch('http://localhost:5000/api/v1/predict/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texts: batch })
        });
        
        const data = await response.json();
        
        if (data.success) {
          results.push(...data.predictions);
        }
        
        setCsvProgress(Math.min(100, ((i + batch.length) / messages.length) * 100));
      }
      
      setCsvResults(results);
      setCsvProgress(100);
    } catch (error) {
      console.error('Error processing CSV:', error);
      alert('Error processing CSV: ' + error.message);
    } finally {
      setCsvProcessing(false);
    }
  };

  const downloadResults = () => {
    if (csvResults.length === 0) return;
    
    const csv = [
      ['Text', 'Prediction', 'Toxicity Score', 'Level'].join(','),
      ...csvResults.map(r => [
        `"${r.text.replace(/"/g, '""')}"`,
        r.prediction,
        r.toxicity_score,
        r.level
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'toxicity_results.csv';
    a.click();
  };

  // ============================================================
  // UTILITY FUNCTIONS
  // ============================================================

  useEffect(() => {
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  }, [debounceTimer]);

  const getLevelColor = (level) => {
    const colors = {
      'VERY_HIGH': '#ef4444',
      'HIGH': '#f59e0b',
      'MEDIUM': '#fbbf24',
      'LOW': '#10b981',
      'MINIMAL': '#6366f1'
    };
    return colors[level] || '#6b7280';
  };

  const getLevelEmoji = (level) => {
    const emojis = {
      'VERY_HIGH': '🚨',
      'HIGH': '⚠️',
      'MEDIUM': '😐',
      'LOW': '😊',
      'MINIMAL': '✅'
    };
    return emojis[level] || '❓';
  };

  const exampleMessages = [
    { text: "You're amazing! Great work!", label: "Positive" },
    { text: "I respectfully disagree with your opinion", label: "Neutral" },
    { text: "This is stupid and you're an idiot", label: "Toxic" },
    { text: "What a terrible idea, are you dumb?", label: "Very Toxic" },
    { text: "Hey, how are you doing today?", label: "Friendly" }
  ];

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="live-detection-container">
      {/* Header */}
      <div className="live-header">
        <h1>⚡ Live Toxicity Detection</h1>
        <p>Real-time analysis across multiple input methods</p>
      </div>

      {/* Tabs */}
      <div className="tabs-container">
        <button 
          className={`tab ${activeTab === 'manual' ? 'active' : ''}`}
          onClick={() => setActiveTab('manual')}
        >
          <span className="tab-icon">✍️</span>
          Manual Input
        </button>
        <button 
          className={`tab ${activeTab === 'telegram' ? 'active' : ''}`}
          onClick={() => setActiveTab('telegram')}
        >
          <span className="tab-icon">📱</span>
          Telegram Live Feed
          {telegramMessages.length > 0 && (
            <span className="tab-badge">{telegramMessages.length}</span>
          )}
        </button>
        <button 
          className={`tab ${activeTab === 'csv' ? 'active' : ''}`}
          onClick={() => setActiveTab('csv')}
        >
          <span className="tab-icon">📤</span>
          CSV Upload
          {csvResults.length > 0 && (
            <span className="tab-badge">{csvResults.length}</span>
          )}
        </button>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {/* ============================================================ */}
        {/* MANUAL INPUT TAB */}
        {/* ============================================================ */}
        {activeTab === 'manual' && (
          <motion.div
            key="manual"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="tab-content"
          >
            <div className="split-container">
              {/* LEFT SIDE - Input */}
              <div className="input-panel">
                <div className="panel-header">
                  <h2>💬 Your Message</h2>
                  <span className="char-count">{inputText.length} characters</span>
                </div>

                <div className="instruction-box">
                  <p>✍️ <strong>Type your message</strong> and it will be analyzed automatically after you stop typing (300ms delay)</p>
                  <p>⚡ Or click <strong>"Analyze Now"</strong> button for instant results</p>
                  <p>📝 Try the example buttons below for quick testing</p>
                </div>

                <textarea
                  className="input-textarea"
                  placeholder="Type or paste your message here..."
                  value={inputText}
                  onChange={(e) => handleTextChange(e.target.value)}
                  rows={8}
                />

                <button 
                  className="analyze-button"
                  onClick={handleAnalyzeClick}
                  disabled={!inputText.trim() || isAnalyzing}
                >
                  {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze Now'}
                </button>

                {/* Example Messages */}
                <div className="examples-section">
                  <h3>💡 Try These Examples:</h3>
                  <div className="examples-grid">
                    {exampleMessages.map((example, idx) => (
                      <button
                        key={idx}
                        className="example-button"
                        onClick={() => handleTextChange(example.text, true)}
                      >
                        <span className="example-label">{example.label}</span>
                        <span className="example-text">{example.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* RIGHT SIDE - Results */}
              <div className="results-panel">
                <div className="panel-header">
                  <h2>🔍 Live Analysis</h2>
                  {isAnalyzing && <span className="analyzing-badge">⏳ Analyzing...</span>}
                </div>

                <AnimatePresence mode="wait">
                  {!inputText.trim() ? (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="empty-state"
                    >
                      <div className="empty-icon">💭</div>
                      <h3>Waiting for input...</h3>
                      <p>Start typing on the left to see real-time toxicity detection in action!</p>
                    </motion.div>
                  ) : result && result.success ? (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="result-display"
                    >
                      {/* Main Result Card */}
                      <div className="main-result-card">
                        <div className="result-icon" style={{ color: getLevelColor(result.level) }}>
                          {getLevelEmoji(result.level)}
                        </div>
                        
                        <div className="result-info">
                          <h3 className="prediction-text">{result.prediction}</h3>
                          <div 
                            className="level-badge-large"
                            style={{ backgroundColor: getLevelColor(result.level) }}
                          >
                            {result.level}
                          </div>
                        </div>
                      </div>

                      {/* Detailed Metrics */}
                      <div className="metrics-grid">
                        <div className="metric-card">
                          <div className="metric-icon">📊</div>
                          <div className="metric-content">
                            <span className="metric-label">Toxicity Score</span>
                            <span className="metric-value">{(result.toxicity_score * 100).toFixed(2)}%</span>
                          </div>
                        </div>

                        <div className="metric-card">
                          <div className="metric-icon">⚡</div>
                          <div className="metric-content">
                            <span className="metric-label">Response Time</span>
                            <span className="metric-value">{result.processing_time_ms?.toFixed(2) || result.latency_ms} ms</span>
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      <div className="toxicity-bar-container">
                        <label>Toxicity Level</label>
                        <div className="toxicity-bar">
                          <motion.div 
                            className="toxicity-fill"
                            initial={{ width: 0 }}
                            animate={{ width: `${result.toxicity_score * 100}%` }}
                            transition={{ duration: 0.5, ease: "easeOut" }}
                            style={{ backgroundColor: getLevelColor(result.level) }}
                          />
                        </div>
                        <div className="bar-labels">
                          <span>0%</span>
                          <span>50%</span>
                          <span>100%</span>
                        </div>
                      </div>

                      {/* Interpretation */}
                      <div className="interpretation-box">
                        <h4>📖 What This Means:</h4>
                        {result.level === 'MINIMAL' && (
                          <p>✅ This message appears safe and non-toxic. It's unlikely to cause harm.</p>
                        )}
                        {result.level === 'LOW' && (
                          <p>😊 This message has minimal toxicity. Generally acceptable but monitor context.</p>
                        )}
                        {result.level === 'MEDIUM' && (
                          <p>😐 This message contains moderately toxic language. Review recommended.</p>
                        )}
                        {result.level === 'HIGH' && (
                          <p>⚠️ This message is highly toxic. Human moderation strongly recommended.</p>
                        )}
                        {result.level === 'VERY_HIGH' && (
                          <p>🚨 This message is extremely toxic. Immediate moderation required!</p>
                        )}
                      </div>
                    </motion.div>
                  ) : result && result.error ? (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="error-state"
                    >
                      <div className="error-icon">❌</div>
                      <h3>Error</h3>
                      <p>{result.error}</p>
                    </motion.div>
                  ) : null}
                </AnimatePresence>
              </div>
            </div>

            {/* History Section */}
            {history.length > 0 && (
              <div className="history-section">
                <h3>📜 Recent Analysis History</h3>
                <div className="history-grid">
                  {history.map((item, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="history-item"
                      onClick={() => handleTextChange(item.text)}
                    >
                      <div className="history-header">
                        <span 
                          className="history-level"
                          style={{ backgroundColor: getLevelColor(item.result.level) }}
                        >
                          {getLevelEmoji(item.result.level)} {item.result.level}
                        </span>
                        <span className="history-time">{item.timestamp}</span>
                      </div>
                      <p className="history-text">{item.text}</p>
                      <div className="history-score">
                        Score: {(item.result.toxicity_score * 100).toFixed(1)}%
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* ============================================================ */}
        {/* TELEGRAM TAB */}
        {/* ============================================================ */}
        {activeTab === 'telegram' && (
          <motion.div
            key="telegram"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="tab-content"
          >
            {/* Bot Status Header */}
            <div className="telegram-header">
              <div className="bot-status-card">
                <div className="status-indicator-container">
                  <div className={`status-indicator ${botStatus.running ? 'active' : 'inactive'}`}>
                    {botStatus.running ? '🟢' : '🔴'}
                  </div>
                  <div>
                    <h3>Bot Status: {botStatus.running ? 'Running' : 'Offline'}</h3>
                    <p className="status-subtitle">
                      {botStatus.running 
                        ? `Active • Last message: ${botStatus.last_message_at ? new Date(botStatus.last_message_at).toLocaleTimeString() : 'Never'}` 
                        : 'Start the bot with: python backend/telegram_bot.py'}
                    </p>
                  </div>
                </div>
                <div className="bot-controls">
                  <button className="control-btn refresh" onClick={() => { fetchTelegramMessages(); fetchTelegramStats(); }}>
                    🔄 Refresh
                  </button>
                  <button className="control-btn clear" onClick={clearTelegramMessages}>
                    🗑️ Clear Messages
                  </button>
                  <button className="control-btn reset" onClick={resetTelegramStats}>
                    ↩️ Reset Stats
                  </button>
                </div>
              </div>

              {/* Statistics Dashboard */}
              {telegramStats && (
                <div className="stats-dashboard">
                  <div className="stat-card">
                    <div className="stat-icon">📊</div>
                    <div className="stat-content">
                      <div className="stat-value">{telegramStats.total_analyzed || 0}</div>
                      <div className="stat-label">Total Analyzed</div>
                    </div>
                  </div>
                  <div className="stat-card toxic">
                    <div className="stat-icon">🚨</div>
                    <div className="stat-content">
                      <div className="stat-value">{telegramStats.total_toxic || 0}</div>
                      <div className="stat-label">Toxic Messages ({telegramStats.toxic_percentage?.toFixed(1) || 0}%)</div>
                    </div>
                  </div>
                  <div className="stat-card safe">
                    <div className="stat-icon">✅</div>
                    <div className="stat-content">
                      <div className="stat-value">{telegramStats.total_non_toxic || 0}</div>
                      <div className="stat-label">Non-toxic Messages ({telegramStats.non_toxic_percentage?.toFixed(1) || 0}%)</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Level Distribution */}
              {telegramStats && telegramStats.by_level && (
                <div className="level-distribution">
                  <h4>📈 Distribution by Severity Level</h4>
                  <div className="level-bars">
                    {Object.entries(telegramStats.by_level).map(([level, count]) => (
                      <div key={level} className="level-bar-item">
                        <span className="level-name">{getLevelEmoji(level)} {level}</span>
                        <div className="level-bar-track">
                          <div 
                            className="level-bar-fill"
                            style={{ 
                              width: `${(count / (telegramStats.total_analyzed || 1)) * 100}%`,
                              backgroundColor: getLevelColor(level)
                            }}
                          />
                        </div>
                        <span className="level-count">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Messages Feed */}
            <div className="telegram-messages-container">
              <div className="messages-header">
                <h3>📱 Real-time Message Feed</h3>
                <span className="auto-refresh">🔄 Auto-refreshing every 5s</span>
              </div>

              {telegramLoading && telegramMessages.length === 0 ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>Loading messages...</p>
                </div>
              ) : telegramMessages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">📭</div>
                  <h3>No messages yet</h3>
                  <p>Messages from your Telegram bot will appear here in real-time.</p>
                  <div className="setup-instructions">
                    <h4>Quick Setup:</h4>
                    <ol>
                      <li>Start backend: <code>python backend/app.py</code></li>
                      <li>Start bot: <code>python backend/telegram_bot.py</code></li>
                      <li>Open Telegram and message your bot: <code>@haki_filter_bot</code></li>
                    </ol>
                  </div>
                </div>
              ) : (
                <div className="messages-feed">
                  {telegramMessages.map((msg, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="telegram-message-card"
                      style={{ borderLeftColor: getLevelColor(msg.level) }}
                    >
                      <div className="message-header">
                        <div className="user-info">
                          <span className="user-icon">👤</span>
                          <span className="username">{msg.first_name || msg.username || 'User'}</span>
                          <span className="user-id">ID: {msg.user_id}</span>
                        </div>
                        <span className="message-time">{new Date(msg.timestamp).toLocaleString()}</span>
                      </div>
                      
                      <div className="message-text">
                        <p>{msg.text}</p>
                      </div>
                      
                      <div className="message-analysis">
                        <div className="analysis-row">
                          <span 
                            className="prediction-badge"
                            style={{ backgroundColor: getLevelColor(msg.level) }}
                          >
                            {getLevelEmoji(msg.level)} {msg.prediction}
                          </span>
                          <span className="score-badge">
                            {(msg.toxicity_score * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="analysis-details">
                          <span>Level: <strong>{msg.level}</strong></span>
                          <span>•</span>
                          <span>Lexicon: <strong>{msg.lexicon_severity}</strong></span>
                          <span>•</span>
                          <span>Time: <strong>{msg.processing_time_ms?.toFixed(2)}ms</strong></span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* ============================================================ */}
        {/* CSV UPLOAD TAB */}
        {/* ============================================================ */}
        {activeTab === 'csv' && (
          <motion.div
            key="csv"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="tab-content"
          >
            <div className="csv-container">
              <div className="csv-header">
                <h2>📤 Batch CSV Analysis</h2>
                <p>Upload a CSV file with messages to analyze in bulk</p>
              </div>

              {/* Upload Section */}
              <div className="csv-upload-section">
                <div className="upload-box">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleCsvUpload}
                    id="csv-upload"
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="csv-upload" className="upload-label">
                    <div className="upload-icon">📁</div>
                    <h3>{csvFile ? csvFile.name : 'Choose CSV File'}</h3>
                    <p>Click to browse or drag and drop</p>
                    <span className="file-requirements">CSV with 'text', 'message', or 'comment' column</span>
                  </label>
                </div>

                {csvFile && !csvProcessing && csvResults.length === 0 && (
                  <button className="process-btn" onClick={processCsv}>
                    🚀 Process CSV File
                  </button>
                )}
              </div>

              {/* Processing Progress */}
              {csvProcessing && (
                <div className="processing-section">
                  <h3>⏳ Processing CSV...</h3>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${csvProgress}%` }}
                    />
                  </div>
                  <p>{csvProgress.toFixed(0)}% complete</p>
                </div>
              )}

              {/* Results Table */}
              {csvResults.length > 0 && (
                <div className="csv-results-section">
                  <div className="results-header">
                    <h3>📊 Analysis Results ({csvResults.length} messages)</h3>
                    <button className="download-btn" onClick={downloadResults}>
                      💾 Download Results CSV
                    </button>
                  </div>

                  <div className="results-stats">
                    <div className="result-stat">
                      <span className="stat-label">Toxic:</span>
                      <span className="stat-value toxic">
                        {csvResults.filter(r => r.prediction !== 'Non-toxic').length}
                        ({((csvResults.filter(r => r.prediction !== 'Non-toxic').length / csvResults.length) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="result-stat">
                      <span className="stat-label">Non-toxic:</span>
                      <span className="stat-value safe">
                        {csvResults.filter(r => r.prediction === 'Non-toxic').length}
                        ({((csvResults.filter(r => r.prediction === 'Non-toxic').length / csvResults.length) * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>

                  <div className="results-table-container">
                    <table className="results-table">
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
                        {csvResults.map((result, idx) => (
                          <tr key={idx} style={{ borderLeftColor: getLevelColor(result.level) }}>
                            <td>{idx + 1}</td>
                            <td className="text-cell">
                              {result.text.substring(0, 100)}
                              {result.text.length > 100 && '...'}
                            </td>
                            <td>
                              <span 
                                className="prediction-badge-small"
                                style={{ backgroundColor: getLevelColor(result.level) }}
                              >
                                {result.prediction}
                              </span>
                            </td>
                            <td className="score-cell">
                              {(result.toxicity_score * 100).toFixed(2)}%
                            </td>
                            <td>
                              <span className="level-badge-small">
                                {getLevelEmoji(result.level)} {result.level}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer Info */}
      <div className="live-footer">
        <div className="footer-stat">
          <span className="stat-icon">⚡</span>
          <span>Real-time Detection</span>
        </div>
        <div className="footer-stat">
          <span className="stat-icon">🎯</span>
          <span>90.19% Accuracy</span>
        </div>
        <div className="footer-stat">
          <span className="stat-icon">🚀</span>
          <span>10,007 Features (TF-IDF + Lexicon)</span>
        </div>
      </div>
    </div>
  );
};

export default LiveDetection;
