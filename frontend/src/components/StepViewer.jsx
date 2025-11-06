import { CheckCircle2, Loader2, AlertCircle, Clock } from 'lucide-react'
import './StepViewer.css'

const StepViewer = ({ step, status }) => {
  const getStatusBadge = () => {
    switch (status) {
      case 'completed':
        return (
          <div className="status-badge completed">
            <CheckCircle2 size={20} />
            Completed
          </div>
        )
      case 'running':
        return (
          <div className="status-badge running">
            <Loader2 size={20} className="spin" />
            Running...
          </div>
        )
      case 'failed':
        return (
          <div className="status-badge failed">
            <AlertCircle size={20} />
            Failed
          </div>
        )
      default:
        return (
          <div className="status-badge pending">
            <Clock size={20} />
            Pending
          </div>
        )
    }
  }

  return (
    <div className="step-viewer">
      <div className="step-header-section">
        <div className="step-title-row">
          <h2 className="step-title">
            Step {step.id}: {step.name}
          </h2>
          {getStatusBadge()}
        </div>
        
        <p className="step-description">{step.description}</p>
      </div>

      <div className="step-content-section">
        <div className="info-card">
          <h3 className="info-title">What This Step Does</h3>
          <div className="info-content">
            {step.id === 1 && (
              <>
                <p>This step loads the Jigsaw Toxic Comments dataset from the CSV file and performs initial validation:</p>
                <ul>
                  <li>Reads the CSV with 11 columns including comment text and toxicity labels</li>
                  <li>Validates the schema and data types</li>
                  <li>Calculates basic statistics (row count, column count)</li>
                  <li>Identifies missing values and data quality issues</li>
                </ul>
                <div className="code-block">
                  <code>spark.read.csv("data/chat_data.csv", header=True, schema=defined_schema)</code>
                </div>
              </>
            )}
            
            {step.id === 2 && (
              <>
                <p>Text preprocessing is crucial for NLP tasks. This step cleans and prepares the text data:</p>
                <ul>
                  <li>Removes URLs, special characters, and extra whitespace</li>
                  <li>Converts text to lowercase for consistency</li>
                  <li>Creates binary labels (0 = non-toxic, 1 = toxic) from multiple toxicity columns</li>
                  <li>Filters out null values and invalid entries</li>
                </ul>
                <div className="code-block">
                  <code>clean_text_udf = udf(clean_text, StringType())<br/>
                  df = df.withColumn("cleaned_text", clean_text_udf(col("comment_text")))</code>
                </div>
              </>
            )}

            {step.id === 3 && (
              <>
                <p>Feature extraction converts text into numerical vectors using TF-IDF:</p>
                <ul>
                  <li><strong>Tokenization:</strong> Splits text into individual words</li>
                  <li><strong>HashingTF:</strong> Converts tokens to term frequency vectors (10,000 features)</li>
                  <li><strong>IDF:</strong> Applies inverse document frequency to weight important terms</li>
                  <li>Generates sparse vectors that represent text semantically</li>
                </ul>
                <div className="code-block">
                  <code>tokenizer → HashingTF(10000) → IDF → features vector</code>
                </div>
              </>
            )}

            {step.id === 4 && (
              <>
                <p>Splits the dataset into training and testing sets:</p>
                <ul>
                  <li><strong>Training Set:</strong> 80% of data for model learning</li>
                  <li><strong>Test Set:</strong> 20% of data for unbiased evaluation</li>
                  <li>Random seed ensures reproducible splits</li>
                  <li>Stratified sampling maintains label distribution</li>
                </ul>
              </>
            )}

            {step.id === 5 && (
              <>
                <p>Trains a Logistic Regression model for binary classification:</p>
                <ul>
                  <li><strong>Algorithm:</strong> Logistic Regression with L2 regularization</li>
                  <li><strong>Max Iterations:</strong> 100 (configured in config.py)</li>
                  <li><strong>Regularization:</strong> 0.01 to prevent overfitting</li>
                  <li>Uses Spark MLlib's distributed training</li>
                  <li>Outputs probability scores and binary predictions</li>
                </ul>
              </>
            )}

            {step.id === 6 && (
              <>
                <p>Evaluates model performance using multiple metrics:</p>
                <ul>
                  <li><strong>AUC-ROC:</strong> 83.01% - Area Under ROC Curve</li>
                  <li><strong>Accuracy:</strong> 94.42% - Overall prediction correctness</li>
                  <li><strong>F1 Score:</strong> 93.02% - Harmonic mean of precision and recall</li>
                  <li>Confusion matrix for detailed error analysis</li>
                </ul>
                <div className="metrics-highlight">
                  <div className="metric-item">
                    <span className="metric-label">AUC-ROC</span>
                    <span className="metric-value">83.01%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Accuracy</span>
                    <span className="metric-value">94.42%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">F1 Score</span>
                    <span className="metric-value">93.02%</span>
                  </div>
                </div>
              </>
            )}

            {step.id === 7 && (
              <>
                <p>Generates predictions on new data:</p>
                <ul>
                  <li>Applies trained model to all messages</li>
                  <li>Extracts probability scores from sparse vectors</li>
                  <li>Creates binary predictions (toxic vs non-toxic)</li>
                  <li>Adds message IDs for traceability</li>
                  <li>Processes 212,566 messages in this batch</li>
                </ul>
              </>
            )}

            {step.id === 8 && (
              <>
                <p>Aggregates toxicity at the user level:</p>
                <ul>
                  <li>Groups messages by user_id</li>
                  <li>Calculates average, max, and min toxicity scores per user</li>
                  <li>Counts toxic and non-toxic messages per user</li>
                  <li>Assigns severity levels: VERY_HIGH, HIGH, MODERATE, LOW, MINIMAL</li>
                  <li>Analyzes 183,673 unique users</li>
                </ul>
                <div className="code-block">
                  <code>severity = CASE<br/>
                  &nbsp;&nbsp;WHEN avg_score ≥ 0.9 THEN 'VERY_HIGH'<br/>
                  &nbsp;&nbsp;WHEN avg_score ≥ 0.7 THEN 'HIGH'<br/>
                  &nbsp;&nbsp;WHEN avg_score ≥ 0.5 THEN 'MODERATE'<br/>
                  &nbsp;&nbsp;...</code>
                </div>
              </>
            )}

            {step.id === 9 && (
              <>
                <p>Exports all results to CSV files:</p>
                <ul>
                  <li><strong>predictions.csv:</strong> Message-level predictions with scores</li>
                  <li><strong>user_analysis.csv:</strong> User-level aggregated statistics</li>
                  <li><strong>model/:</strong> Saved model for future use</li>
                  <li>Files saved to results/ directory</li>
                  <li>Total execution time: 26.08 seconds</li>
                </ul>
              </>
            )}
          </div>
        </div>

        <div className="tech-info-card">
          <h3 className="info-title">Technical Details</h3>
          <div className="tech-details">
            <div className="tech-item">
              <span className="tech-label">Module:</span>
              <span className="tech-value">
                {step.id <= 2 ? 'data_ingestion.py & preprocessing.py' :
                 step.id <= 4 ? 'preprocessing.py' :
                 step.id <= 7 ? 'model.py' :
                 step.id === 8 ? 'user_analysis.py' : 'main.py'}
              </span>
            </div>
            <div className="tech-item">
              <span className="tech-label">Spark Component:</span>
              <span className="tech-value">
                {step.id === 1 ? 'SparkSession, DataFrameReader' :
                 step.id === 2 ? 'DataFrame transformations, UDFs' :
                 step.id === 3 ? 'ML Pipeline, Tokenizer, HashingTF, IDF' :
                 step.id <= 7 ? 'MLlib - LogisticRegression' :
                 step.id === 8 ? 'DataFrame aggregations' : 'DataFrameWriter'}
              </span>
            </div>
            <div className="tech-item">
              <span className="tech-label">Estimated Time:</span>
              <span className="tech-value">
                {step.id === 1 ? '~2-3 seconds' :
                 step.id === 2 ? '~3-4 seconds' :
                 step.id === 3 ? '~4-5 seconds' :
                 step.id === 5 ? '~8-10 seconds' :
                 '~1-2 seconds'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StepViewer
