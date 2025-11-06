import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './LiveDetection.css';

const LiveDetection = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);
  const [debounceTimer, setDebounceTimer] = useState(null);

  // Real-time analysis with debouncing
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
        
        // Add to history
        setHistory(prev => [{
          text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
          result: data,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev.slice(0, 9)]); // Keep last 10
      } else {
        // API returned error
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

  // Handle text input with debouncing (300ms delay)
  const handleTextChange = (text, immediate = false) => {
    setInputText(text);
    
    // Clear existing timer
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    
    // If immediate (from example button), analyze right away
    if (immediate) {
      analyzeText(text);
      return;
    }
    
    // Set new timer for real-time analysis
    const timer = setTimeout(() => {
      analyzeText(text);
    }, 300); // 300ms delay after user stops typing
    
    setDebounceTimer(timer);
  };
  
  // Manual analyze button
  const handleAnalyzeClick = () => {
    if (inputText.trim()) {
      analyzeText(inputText);
    }
  };

  // Cleanup timer on unmount
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
      'MODERATE': '#fbbf24',
      'LOW': '#10b981',
      'MINIMAL': '#6366f1'
    };
    return colors[level] || '#6b7280';
  };

  const getLevelEmoji = (level) => {
    const emojis = {
      'VERY_HIGH': '🚨',
      'HIGH': '⚠️',
      'MODERATE': '😐',
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

  return (
    <div className="live-detection-container">
      {/* Header */}
      <div className="live-header">
        <h1>⚡ Live Toxicity Detection</h1>
        <p>Type anything and watch real-time analysis happen instantly!</p>
      </div>

      {/* Main Split View */}
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
            className="live-input"
            placeholder="Start typing your message here...
            
The analysis will appear in real-time on the right side as you type!"
            value={inputText}
            onChange={(e) => handleTextChange(e.target.value)}
            autoFocus
          />

          {/* Quick Examples */}
          <div className="examples-section">
            <h3>📝 Try These Examples:</h3>
            <div className="example-buttons">
              {exampleMessages.map((example, idx) => (
                <button
                  key={idx}
                  className="example-btn"
                  onClick={() => handleTextChange(example.text, true)}
                >
                  <span className="example-label">{example.label}</span>
                  <span className="example-text">{example.text}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="action-buttons">
            <button 
              className="analyze-btn"
              onClick={handleAnalyzeClick}
              disabled={!inputText.trim() || isAnalyzing}
            >
              {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze Now'}
            </button>
            
            {inputText && (
              <button 
                className="clear-btn"
                onClick={() => handleTextChange('', false)}
              >
                🗑️ Clear
              </button>
            )}
          </div>
        </div>

        {/* RIGHT SIDE - Real-time Results */}
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
                      <span className="metric-value">{result.latency_ms} ms</span>
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
                  {result.level === 'MODERATE' && (
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

      {/* Footer Info */}
      <div className="live-footer">
        <div className="footer-stat">
          <span className="stat-icon">⚡</span>
          <span>Real-time Detection</span>
        </div>
        <div className="footer-stat">
          <span className="stat-icon">🔄</span>
          <span>Auto-updates as you type</span>
        </div>
        <div className="footer-stat">
          <span className="stat-icon">📊</span>
          <span>{history.length} analyses performed</span>
        </div>
      </div>
    </div>
  );
};

export default LiveDetection;
