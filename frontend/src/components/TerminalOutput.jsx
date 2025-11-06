import { useEffect, useRef } from 'react'
import { Terminal as TerminalIcon } from 'lucide-react'
import './TerminalOutput.css'

const TerminalOutput = ({ outputs, isRunning }) => {
  const terminalRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom when new output arrives
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [outputs])

  return (
    <div className="terminal-output">
      <div className="terminal-header">
        <div className="terminal-title">
          <TerminalIcon size={20} />
          <span>Terminal Output</span>
        </div>
        <div className="terminal-buttons">
          <span className="terminal-dot red"></span>
          <span className="terminal-dot yellow"></span>
          <span className="terminal-dot green"></span>
        </div>
      </div>

      <div className="terminal-body" ref={terminalRef}>
        {outputs.length === 0 ? (
          <div className="terminal-empty">
            <TerminalIcon size={48} className="empty-icon" />
            <p>No output yet. Run the step to see the output here.</p>
          </div>
        ) : (
          <div className="terminal-content">
            {outputs.map((output, index) => (
              <div key={index} className="terminal-line">
                <span className="line-number">{index + 1}</span>
                <span className="line-content">{output}</span>
              </div>
            ))}
            
            {isRunning && (
              <div className="terminal-cursor">
                <span className="cursor-blink">▊</span>
              </div>
            )}
          </div>
        )}
      </div>

      {isRunning && (
        <div className="terminal-footer">
          <div className="status-indicator">
            <span className="status-dot running"></span>
            <span>Process running...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default TerminalOutput
