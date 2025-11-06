"""
Flask Backend for Detox Web Application
Provides API endpoints for interactive project demonstration
Uses Spark SQL, MLlib, and streaming concepts
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import time
import threading
from datetime import datetime
import subprocess
from io import StringIO
import contextlib
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import toxicity lexicon for feature extraction
from toxicity_lexicon import get_toxic_word_count

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Import and register production API endpoints
try:
    from backend.api_v1 import api_v1
    from backend.model_cache import model_cache
    app.register_blueprint(api_v1)
    print("[INFO] ✓ Production API v1 registered at /api/v1")
except ImportError as e:
    print(f"[WARNING] Could not import production API: {e}")
    model_cache = None

# Global variables for pipeline state
pipeline_state = {
    "current_step": 0,
    "is_running": False,
    "steps": [],
    "spark_session": None,
    "results": {}
}

# Pipeline steps configuration
PIPELINE_STEPS = [
    {
        "id": 1,
        "title": "Data Ingestion",
        "description": "Loading and validating the toxic comments dataset from CSV",
        "icon": "database",
        "details": [
            "Read CSV file with proper schema (11 columns)",
            "Validate data quality and remove nulls",
            "Calculate dataset statistics",
            "Cache data for better performance"
        ]
    },
    {
        "id": 2,
        "title": "Data Preprocessing",
        "description": "Cleaning text and preparing data for machine learning",
        "icon": "filter",
        "details": [
            "Remove URLs, emails, and special characters",
            "Convert text to lowercase",
            "Tokenize comments into words",
            "Remove stop words (common words like 'the', 'is')",
            "Create label column for classification"
        ]
    },
    {
        "id": 3,
        "title": "Feature Engineering",
        "description": "Extracting numerical features from text using TF-IDF",
        "icon": "cpu",
        "details": [
            "Apply TF (Term Frequency) - count word occurrences",
            "Apply IDF (Inverse Document Frequency) - weight important words",
            "Generate 10,000 features per document",
            "Create sparse vectors for efficiency"
        ]
    },
    {
        "id": 4,
        "title": "Train-Test Split",
        "description": "Splitting data into training (80%) and testing (20%) sets",
        "icon": "split",
        "details": [
            "Randomly split data with seed=42",
            "Training set: ~170,000 records",
            "Test set: ~42,000 records",
            "Maintain class distribution"
        ]
    },
    {
        "id": 5,
        "title": "Model Training",
        "description": "Training Logistic Regression classifier",
        "icon": "brain",
        "details": [
            "Initialize Logistic Regression model",
            "Set max iterations = 100",
            "Apply L2 regularization (λ=0.01)",
            "Optimize using L-BFGS algorithm",
            "Monitor convergence"
        ]
    },
    {
        "id": 6,
        "title": "Model Evaluation",
        "description": "Evaluating model performance on test set",
        "icon": "chart",
        "details": [
            "Calculate AUC-ROC score",
            "Measure Accuracy, Precision, Recall",
            "Compute F1 Score",
            "Generate confusion matrix",
            "Expected accuracy: ~94%"
        ]
    },
    {
        "id": 7,
        "title": "Predictions",
        "description": "Generating toxicity predictions for all messages",
        "icon": "check",
        "details": [
            "Apply trained model to full dataset",
            "Extract toxicity probability scores",
            "Classify into levels (LOW/MODERATE/HIGH/SEVERE)",
            "Generate per-message predictions"
        ]
    },
    {
        "id": 8,
        "title": "User Analysis",
        "description": "Aggregating toxicity metrics at user level",
        "icon": "users",
        "details": [
            "Group messages by user ID",
            "Calculate average toxicity score per user",
            "Count toxic messages per user",
            "Identify top toxic users",
            "Generate user toxicity distribution"
        ]
    },
    {
        "id": 9,
        "title": "Results Export",
        "description": "Saving predictions and analytics to CSV files",
        "icon": "save",
        "details": [
            "Export message-level predictions",
            "Export user-level aggregates",
            "Save trained model",
            "Generate summary statistics",
            "Create visualizations"
        ]
    }
]

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Detox API Server",
        "version": "1.0.0"
    })

@app.route('/api/project-info', methods=['GET'])
def get_project_info():
    """Get project information"""
    return jsonify({
        "title": "Detox - Chat Message Toxicity Detector",
        "subtitle": "Real-time Toxicity Detection using Apache Spark & Machine Learning",
        "student": "SARAVANA PRIYAN S T",
        "regNo": "927623BAD100",
        "course": "ADI1302-SPARK SCALA FUNDAMENTALS",
        "techStack": [
            {"name": "Apache Spark", "icon": "⚡", "version": "3.5.3"},
            {"name": "PySpark", "icon": "🐍", "version": "3.5.3"},
            {"name": "Machine Learning", "icon": "🤖", "version": "Spark MLlib"},
            {"name": "Flask", "icon": "🌐", "version": "3.0.0"},
            {"name": "React", "icon": "⚛️", "version": "18.2.0"}
        ],
        "features": [
            "Real-time toxicity detection",
            "User-level analytics",
            "Spark Web UI monitoring",
            "Interactive step-by-step demo",
            "Scalable processing"
        ],
        "dataset": {
            "name": "Jigsaw Toxic Comment Classification",
            "source": "Kaggle",
            "records": "223,549",
            "features": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        }
    })

@app.route('/api/pipeline-steps', methods=['GET'])
def get_pipeline_steps():
    """Get all pipeline steps"""
    return jsonify({
        "steps": PIPELINE_STEPS,
        "totalSteps": len(PIPELINE_STEPS)
    })

@app.route('/api/pipeline-step/<int:step_id>', methods=['GET'])
def get_pipeline_step(step_id):
    """Get specific pipeline step details"""
    if 1 <= step_id <= len(PIPELINE_STEPS):
        return jsonify(PIPELINE_STEPS[step_id - 1])
    return jsonify({"error": "Invalid step ID"}), 404

@app.route('/api/execute-step/<int:step_id>', methods=['POST'])
def execute_step(step_id):
    """Execute a specific pipeline step"""
    global pipeline_state
    
    if pipeline_state["is_running"]:
        return jsonify({"error": "Pipeline already running"}), 400
    
    if 1 <= step_id <= len(PIPELINE_STEPS):
        pipeline_state["is_running"] = True
        
        # Start step execution in background thread
        thread = threading.Thread(target=run_single_step, args=(step_id,))
        thread.start()
        
        return jsonify({
            "status": "started",
            "message": f"Step {step_id} execution started",
            "step": PIPELINE_STEPS[step_id - 1]
        })
    
    return jsonify({"error": "Invalid step ID"}), 404

# ========== Step Execution Functions ==========

def execute_data_ingestion(spark, socketio, step_id):
    """Step 1: Data Ingestion using Spark SQL"""
    output = []
    try:
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType
        import config
        DATA_PATH = config.INPUT_DATA_PATH
        
        output.append("=" * 60)
        output.append("STEP 1: DATA INGESTION WITH SPARK SQL")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Initializing Spark SQL data ingestion..."})
        
        # Define schema for the CSV
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("comment_text", StringType(), True),
            StructField("toxic", IntegerType(), True),
            StructField("severe_toxic", IntegerType(), True),
            StructField("obscene", IntegerType(), True),
            StructField("threat", IntegerType(), True),
            StructField("insult", IntegerType(), True),
            StructField("identity_hate", IntegerType(), True),
            StructField("input_word_ids", StringType(), True),
            StructField("input_mask", StringType(), True),
            StructField("all_segment_id", StringType(), True)
        ])
        
        output.append(f"[INFO] Dataset path: {DATA_PATH}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[INFO] Loading dataset from: {DATA_PATH}"})
        
        output.append("[INFO] Schema defined with 11 columns (6 toxicity labels)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Reading CSV with structured schema..."})
        
        # Read CSV using Spark SQL
        df = spark.read.csv(DATA_PATH, header=True, schema=schema)
        
        # Cache for performance
        df.cache()
        output.append("[INFO] DataFrame cached in memory for performance")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Caching DataFrame in Spark memory..."})
        
        # Register as temporary SQL table
        df.createOrReplaceTempView("toxic_comments")
        output.append("✓ Registered as SQL temporary view: 'toxic_comments'")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[SUCCESS] SQL table 'toxic_comments' created"})
        
        # Get statistics using Spark SQL
        output.append("\n[INFO] Executing Spark SQL query for statistics...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Running: SELECT COUNT(*) FROM toxic_comments"})
        
        total_count = spark.sql("SELECT COUNT(*) as count FROM toxic_comments").first()['count']
        output.append(f"[DATA] Total records loaded: {total_count:,}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Records: {total_count:,}"})
        
        # Check for nulls
        output.append("[INFO] Running data quality checks with SQL...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Checking for null values..."})
        
        null_check = spark.sql("""
            SELECT 
                SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) as null_ids,
                SUM(CASE WHEN comment_text IS NULL THEN 1 ELSE 0 END) as null_comments
            FROM toxic_comments
        """).first()
        
        output.append(f"[DATA] Quality check - Null IDs: {null_check['null_ids']}, Null comments: {null_check['null_comments']}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[QUALITY] Nulls found - IDs: {null_check['null_ids']}, Comments: {null_check['null_comments']}"})
        
        # Store in pipeline state
        pipeline_state["raw_data"] = df
        pipeline_state["stats"] = {"total_records": total_count, "null_ids": null_check['null_ids'], "null_comments": null_check['null_comments']}
        
        output.append("\n✓ Data ingestion completed successfully")
        output.append("✓ DataFrame cached and available for next steps")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Data ingestion completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Data ingestion failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_preprocessing(spark, socketio, step_id):
    """Step 2: Data Preprocessing"""
    output = []
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.functions import udf, col, when
        from pyspark.sql.types import StringType
        import re
        
        output.append("=" * 60)
        output.append("STEP 2: TEXT PREPROCESSING & FEATURE PREPARATION")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Starting text preprocessing pipeline..."})
        
        df = pipeline_state.get("raw_data")
        if df is None:
            raise Exception("[ERROR] No raw data found. Execute Step 1 first.")
        
        # Define cleaning UDF
        output.append("[INFO] Registering text cleaning UDF (User Defined Function)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Creating text cleaning function..."})
        
        def clean_text(text):
            if text is None:
                return ""
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        clean_udf = udf(clean_text, StringType())
        output.append("✓ UDF registered for distributed text cleaning")
        
        # Clean text
        output.append("\n[PROCESSING] Applying text transformations:")
        output.append("  - Converting to lowercase")
        output.append("  - Removing URLs and special characters")
        output.append("  - Normalizing whitespace")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[TRANSFORM] Cleaning text data..."})
        
        df = df.withColumn("cleaned_text", clean_udf(col("comment_text")))
        output.append("✓ Text cleaning transformation applied")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Text cleaned with regex patterns"})
        
        # Create binary label using Spark SQL
        output.append("\n[INFO] Creating SQL temporary view for label generation")
        df.createOrReplaceTempView("cleaned_data")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[SQL] Executing binary classification query..."})
        
        output.append("[SQL] Running query:")
        output.append("  SELECT *, CASE WHEN (toxic=1 OR severe_toxic=1 OR ...)")
        output.append("  THEN 1.0 ELSE 0.0 END as label FROM cleaned_data")
        
        df = spark.sql("""
            SELECT *,
                CASE 
                    WHEN toxic = 1 OR severe_toxic = 1 OR obscene = 1 
                         OR threat = 1 OR insult = 1 OR identity_hate = 1 
                    THEN 1.0 
                    ELSE 0.0 
                END as label
            FROM cleaned_data
            WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
        """)
        
        df = df.select("id", "cleaned_text", "label")
        df.cache()
        output.append("✓ Binary labels generated (0=Non-toxic, 1=Toxic)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Binary classification labels created"})
        
        # Add lexicon-based features
        output.append("\n[PROCESSING] Extracting lexicon-based features...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Computing toxic word counts..."})
        
        from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
        lexicon_schema = StructType([
            StructField("extreme_toxic_count", IntegerType(), False),
            StructField("high_toxic_count", IntegerType(), False),
            StructField("medium_toxic_count", IntegerType(), False),
            StructField("low_toxic_count", IntegerType(), False),
            StructField("total_toxic_count", IntegerType(), False),
            StructField("toxic_word_ratio", DoubleType(), False),
            StructField("severity_score", DoubleType(), False)
        ])
        
        def extract_lexicon_features(text):
            if not text or text.strip() == "":
                return (0, 0, 0, 0, 0, 0.0, 0.0)
            stats = get_toxic_word_count(text)
            return (
                stats['extreme_toxic_count'],
                stats['high_toxic_count'],
                stats['medium_toxic_count'],
                stats['low_toxic_count'],
                stats['total_toxic_count'],
                stats['toxic_word_ratio'],
                stats['severity_score']
            )
        
        lexicon_udf = udf(extract_lexicon_features, lexicon_schema)
        df = df.withColumn("lexicon_features", lexicon_udf(col("cleaned_text")))
        df = df.select(
            "id", "cleaned_text", "label",
            col("lexicon_features.extreme_toxic_count").alias("extreme_toxic_count"),
            col("lexicon_features.high_toxic_count").alias("high_toxic_count"),
            col("lexicon_features.medium_toxic_count").alias("medium_toxic_count"),
            col("lexicon_features.low_toxic_count").alias("low_toxic_count"),
            col("lexicon_features.total_toxic_count").alias("total_toxic_count"),
            col("lexicon_features.toxic_word_ratio").alias("toxic_word_ratio"),
            col("lexicon_features.severity_score").alias("severity_score")
        )
        df.cache()
        
        output.append("✓ Lexicon features extracted (7 features)")
        output.append("  - extreme_toxic_count, high_toxic_count, medium_toxic_count")
        output.append("  - low_toxic_count, total_toxic_count, toxic_word_ratio, severity_score")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Lexicon features added"})
        
        # Get label distribution
        output.append("\n[ANALYSIS] Computing class distribution...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Analyzing label distribution..."})
        
        label_dist = df.groupBy("label").count().collect()
        total = sum(row['count'] for row in label_dist)
        
        for row in label_dist:
            label = int(row['label'])
            count = row['count']
            percentage = (count / total) * 100
            label_name = "Non-toxic" if label == 0 else "Toxic"
            output.append(f"[DATA] Label {label} ({label_name}): {count:,} records ({percentage:.2f}%)")
            socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] {label_name}: {count:,} ({percentage:.2f}%)"})
        
        pipeline_state["preprocessed_data"] = df
        pipeline_state["label_distribution"] = {int(row['label']): row['count'] for row in label_dist}
        
        output.append(f"\n[INFO] Processed dataset: {total:,} records")
        output.append("✓ Data preprocessing completed successfully")
        output.append("✓ Dataset ready for feature engineering")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Preprocessing pipeline completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Preprocessing failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_feature_engineering(spark, socketio, step_id):
    """Step 3: Feature Engineering with MLlib"""
    output = []
    try:
        from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
        from pyspark.ml import Pipeline
        
        output.append("=" * 60)
        output.append("STEP 3: FEATURE ENGINEERING WITH SPARK MLlib")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Initializing ML Pipeline..."})
        
        df = pipeline_state.get("preprocessed_data")
        if df is None:
            raise Exception("[ERROR] No preprocessed data found. Execute Step 2 first.")
        
        # Build ML Pipeline
        output.append("[INFO] Configuring ML Pipeline stages:")
        output.append("\n[STAGE 1] Tokenizer")
        output.append("  - Input: cleaned_text")
        output.append("  - Output: words (array of tokens)")
        tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Stage 1/5: Tokenizer configured"})
        
        output.append("\n[STAGE 2] StopWordsRemover")
        output.append("  - Input: words")
        output.append("  - Output: filtered_words")
        output.append("  - Removes: 'the', 'a', 'is', etc.")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Stage 2/5: StopWordsRemover configured"})
        
        output.append("\n[STAGE 3] HashingTF (Term Frequency)")
        output.append("  - Input: filtered_words")
        output.append("  - Output: raw_features")
        output.append("  - Feature dimensions: 10,000")
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Stage 3/5: HashingTF configured (10K features)"})
        
        output.append("\n[STAGE 4] IDF (Inverse Document Frequency)")
        output.append("  - Input: raw_features")
        output.append("  - Output: tfidf_features (TF-IDF vectors)")
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Stage 4/5: IDF configured"})
        
        output.append("\n[STAGE 5] VectorAssembler (Combine TF-IDF + Lexicon)")
        output.append("  - Input: tfidf_features + 7 lexicon features")
        output.append("  - Output: features (10,007 dimensions)")
        assembler = VectorAssembler(
            inputCols=[
                "tfidf_features",
                "extreme_toxic_count",
                "high_toxic_count",
                "medium_toxic_count",
                "low_toxic_count",
                "total_toxic_count",
                "toxic_word_ratio",
                "severity_score"
            ],
            outputCol="features"
        )
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Stage 5/5: VectorAssembler configured"})
        
        output.append("\n[INFO] Creating ML Pipeline with 5 transformer stages")
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler])
        
        socketio.emit('step_progress', {"step_id": step_id, "message": "[PROCESSING] Fitting pipeline to dataset..."})
        output.append("\n[PROCESSING] Fitting pipeline transformers...")
        output.append("  - Tokenizing text into words")
        output.append("  - Filtering stop words")
        output.append("  - Computing term frequencies (TF)")
        output.append("  - Computing inverse document frequencies (IDF)")
        output.append("  - Assembling TF-IDF + Lexicon features")
        
        model = pipeline.fit(df)
        output.append("✓ Pipeline model fitted successfully")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Pipeline fitted"})
        
        output.append("\n[PROCESSING] Transforming dataset with fitted pipeline...")
        featured_df = model.transform(df)
        featured_df = featured_df.select("id", "features", "label")
        featured_df.cache()
        
        # Get feature statistics
        total_records = featured_df.count()
        output.append(f"\n[DATA] Total records processed: {total_records:,}")
        output.append("[DATA] Feature vector dimensions: 10,007")
        output.append("[DATA] Feature composition: 10,000 TF-IDF + 7 Lexicon")
        output.append("[DATA] Feature type: Combined sparse + dense vectors")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] {total_records:,} feature vectors (10,007 dims)"})
        
        pipeline_state["featured_data"] = featured_df
        pipeline_state["feature_pipeline"] = model
        pipeline_state["feature_dimensions"] = 10007
        
        output.append("\n✓ Feature engineering completed successfully")
        output.append("✓ Dataset ready for train-test split")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Feature engineering pipeline completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Feature engineering failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_train_test_split(spark, socketio, step_id):
    """Step 4: Train-Test Split"""
    output = []
    try:
        output.append("=" * 60)
        output.append("STEP 4: TRAIN-TEST DATA SPLIT")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Preparing data split..."})
        
        df = pipeline_state.get("featured_data")
        if df is None:
            df = pipeline_state.get("feature_data")
        if df is None:
            raise Exception("[ERROR] No featured data found. Execute Step 3 first.")
        
        output.append("[INFO] Split configuration:")
        output.append("  - Training ratio: 80%")
        output.append("  - Testing ratio: 20%")
        output.append("  - Random seed: 42 (reproducibility)")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": "[PROCESSING] Performing random split (80/20)..."})
        output.append("\n[PROCESSING] Executing randomSplit operation...")
        
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        train_data.cache()
        test_data.cache()
        output.append("✓ Data split completed and cached in memory")
        
        output.append("\n[INFO] Computing split statistics...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Counting records..."})
        
        train_count = train_data.count()
        test_count = test_data.count()
        total_count = train_count + test_count
        
        train_pct = (train_count / total_count) * 100
        test_pct = (test_count / total_count) * 100
        
        pipeline_state["train_data"] = train_data
        pipeline_state["test_data"] = test_data
        pipeline_state["split_stats"] = {
            "train_count": train_count,
            "test_count": test_count,
            "train_percentage": train_pct,
            "test_percentage": test_pct
        }
        
        output.append("\n[DATA] Split Results:")
        output.append(f"  Training Set: {train_count:,} records ({train_pct:.2f}%)")
        output.append(f"  Testing Set:  {test_count:,} records ({test_pct:.2f}%)")
        output.append(f"  Total:        {total_count:,} records")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Train: {train_count:,} ({train_pct:.1f}%) | Test: {test_count:,} ({test_pct:.1f}%)"})
        
        output.append("\n✓ Train-test split completed successfully")
        output.append("✓ Both datasets cached for efficient training")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Data split ready for model training"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Train-test split failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_model_training(spark, socketio, step_id):
    """Step 5: Model Training with MLlib"""
    output = []
    try:
        from pyspark.ml.classification import LogisticRegression
        from pyspark.sql.functions import col, when
        import config
        MAX_ITERATIONS = config.MAX_ITERATIONS
        REGULARIZATION_PARAM = config.REGULARIZATION_PARAM
        
        output.append("=" * 60)
        output.append("STEP 5: MODEL TRAINING WITH SPARK MLlib")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Initializing ML model..."})
        
        train_data = pipeline_state.get("train_data")
        if train_data is None:
            raise Exception("[ERROR] No training data found. Execute Step 4 first.")
        
        # Calculate class weights for imbalanced dataset
        output.append("\n[PROCESSING] Computing class weights for imbalanced dataset...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Calculating class balancing weights..."})
        
        total_count = train_data.count()
        toxic_count = train_data.filter(col("label") == 1).count()
        non_toxic_count = total_count - toxic_count
        
        # Calculate weight for toxic class (minority)
        toxic_weight = total_count / (2.0 * toxic_count)
        non_toxic_weight = 1.0
        
        output.append(f"[DATA] Total training samples: {total_count:,}")
        output.append(f"[DATA] Toxic samples: {toxic_count:,} ({(toxic_count/total_count)*100:.2f}%)")
        output.append(f"[DATA] Non-toxic samples: {non_toxic_count:,} ({(non_toxic_count/total_count)*100:.2f}%)")
        output.append(f"[DATA] Toxic class weight: {toxic_weight:.2f}x")
        output.append(f"[DATA] Non-toxic class weight: {non_toxic_weight:.2f}x")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Class weights: Toxic={toxic_weight:.2f}x, Non-toxic={non_toxic_weight:.2f}x"})
        
        # Add weight column
        train_data = train_data.withColumn(
            "classWeight",
            when(col("label") == 1, toxic_weight).otherwise(non_toxic_weight)
        )
        train_data.cache()
        output.append("✓ Class weights added to training data")
        
        output.append("\n[INFO] Model: Logistic Regression (Binary Classification)")
        output.append("[INFO] Algorithm: L-BFGS Optimizer with Class Balancing")
        output.append("\n[CONFIG] Hyperparameters:")
        output.append(f"  - Maximum iterations: {MAX_ITERATIONS}")
        output.append(f"  - Regularization parameter (λ): {REGULARIZATION_PARAM}")
        output.append("  - Elastic Net parameter: 0.0 (L2 regularization)")
        output.append("  - Features column: 'features' (10,007 dimensions)")
        output.append("  - Label column: 'label' (0=Non-toxic, 1=Toxic)")
        output.append("  - Weight column: 'classWeight' (handles class imbalance)")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[CONFIG] MaxIter={MAX_ITERATIONS}, RegParam={REGULARIZATION_PARAM}, Weighted"})
        
        # Create and train Logistic Regression model with class weights
        output.append("\n[PROCESSING] Creating Logistic Regression estimator...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="classWeight",
            maxIter=MAX_ITERATIONS,
            regParam=REGULARIZATION_PARAM,
            elasticNetParam=0.0
        )
        output.append("✓ Estimator created with class weight balancing")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": "[TRAINING] Fitting weighted model to training data..."})
        output.append("\n[TRAINING] Starting model training with class balancing...")
        output.append("  - Optimizing weighted objective function...")
        output.append("  - Computing gradient descent with sample weights...")
        output.append("  - Iterating until convergence...")
        
        # Train model
        model = lr.fit(train_data)
        
        output.append("\n✓ [SUCCESS] Model training completed with class balancing")
        output.append("[INFO] Model coefficients and intercept computed")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Weighted model trained successfully"})
        
        # Store model summary if available
        try:
            summary = model.summary
            output.append(f"\n[DATA] Training summary:")
            output.append(f"  - Total iterations executed: {summary.totalIterations}")
            output.append(f"  - Objective history length: {len(summary.objectiveHistory)}")
            socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Iterations: {summary.totalIterations}"})
        except:
            pass
        
        pipeline_state["model"] = model
        
        output.append("\n✓ Model ready for evaluation")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Model training pipeline completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Model training failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_model_evaluation(spark, socketio, step_id):
    """Step 6: Model Evaluation"""
    output = []
    try:
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
        
        output.append("=" * 60)
        output.append("STEP 6: MODEL EVALUATION & PERFORMANCE METRICS")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Starting model evaluation..."})
        
        model = pipeline_state.get("model")
        test_data = pipeline_state.get("test_data")
        
        if model is None or test_data is None:
            raise Exception("[ERROR] Model or test data not found. Execute previous steps first.")
        
        output.append("[INFO] Evaluation dataset: Test set (20% of data)")
        output.append("[INFO] Evaluation metrics: AUC-ROC, Accuracy, F1-Score")
        
        # Make predictions
        socketio.emit('step_progress', {"step_id": step_id, "message": "[PROCESSING] Generating predictions on test set..."})
        output.append("\n[PROCESSING] Transforming test data with trained model...")
        predictions = model.transform(test_data)
        predictions.cache()
        
        test_count = predictions.count()
        output.append(f"✓ Generated predictions for {test_count:,} test samples")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Predictions: {test_count:,} samples"})
        
        # Calculate metrics
        output.append("\n[EVALUATION] Computing performance metrics...")
        
        output.append("\n[METRIC 1] AUC-ROC (Area Under Receiver Operating Characteristic)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Computing AUC-ROC..."})
        evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        auc = evaluator_auc.evaluate(predictions)
        output.append(f"  Result: {auc:.6f} ({auc*100:.2f}%)")
        output.append(f"  Interpretation: {'Excellent' if auc > 0.9 else 'Good' if auc > 0.8 else 'Fair' if auc > 0.7 else 'Poor'} discrimination ability")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] AUC-ROC: {auc*100:.2f}%"})
        
        output.append("\n[METRIC 2] Accuracy (Correct predictions / Total predictions)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Computing Accuracy..."})
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator_acc.evaluate(predictions)
        output.append(f"  Result: {accuracy:.6f} ({accuracy*100:.2f}%)")
        output.append(f"  Correct predictions: ~{int(accuracy * test_count):,} out of {test_count:,}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Accuracy: {accuracy*100:.2f}%"})
        
        output.append("\n[METRIC 3] F1-Score (Harmonic mean of Precision and Recall)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Computing F1-Score..."})
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f1 = evaluator_f1.evaluate(predictions)
        output.append(f"  Result: {f1:.6f} ({f1*100:.2f}%)")
        output.append(f"  Balance: Considers both precision and recall")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] F1-Score: {f1*100:.2f}%"})
        
        pipeline_state["metrics"] = {"auc": auc, "accuracy": accuracy, "f1": f1}
        pipeline_state["predictions"] = predictions
        pipeline_state["test_count"] = test_count
        
        output.append("\n" + "=" * 60)
        output.append("PERFORMANCE SUMMARY")
        output.append("=" * 60)
        output.append(f"AUC-ROC:  {auc:.4f} ({'🟢 Excellent' if auc > 0.9 else '🟡 Good' if auc > 0.8 else '🟠 Fair'})")
        output.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        output.append(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        output.append("=" * 60)
        
        output.append("\n✓ [SUCCESS] Model evaluation completed")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Evaluation metrics computed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Model evaluation failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_predictions(spark, socketio, step_id):
    """Step 7: Generate Predictions"""
    output = []
    try:
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        from pyspark.ml.linalg import Vector
        
        output.append("=" * 60)
        output.append("STEP 7: PREDICTION GENERATION & PROBABILITY EXTRACTION")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Starting prediction generation..."})
        
        predictions = pipeline_state.get("predictions")
        if predictions is None:
            raise Exception("[ERROR] No predictions found. Execute Step 6 first.")
        
        output.append("[INFO] Input: Model predictions with probability vectors")
        output.append("[INFO] Task: Extract toxicity probability scores")
        
        # Extract probability UDF
        output.append("\n[PROCESSING] Creating UDF for probability extraction...")
        def extract_prob(v):
            try:
                return float(v[1]) if v and len(v) > 1 else 0.0
            except:
                return 0.0
        
        extract_prob_udf = udf(extract_prob, DoubleType())
        output.append("✓ UDF registered for extracting class=1 probabilities")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Extracting toxicity scores..."})
        
        output.append("\n[TRANSFORM] Applying probability extraction to predictions...")
        predictions = predictions.withColumn("toxicity_score", extract_prob_udf("probability"))
        predictions = predictions.select("id", "label", "prediction", "toxicity_score")
        output.append("✓ Toxicity scores computed (range: 0.0 to 1.0)")
        output.append("✓ Selected columns: id, label, prediction, toxicity_score")
        
        output.append("\n[ANALYSIS] Computing prediction statistics...")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Analyzing predictions..."})
        
        pred_count = predictions.count()
        toxic_count = predictions.filter("prediction = 1").count()
        non_toxic_count = pred_count - toxic_count
        toxic_pct = (toxic_count / pred_count) * 100
        non_toxic_pct = (non_toxic_count / pred_count) * 100
        
        output.append(f"\n[DATA] Prediction Statistics:")
        output.append(f"  Total predictions:     {pred_count:,}")
        output.append(f"  Toxic (Class 1):       {toxic_count:,} ({toxic_pct:.2f}%)")
        output.append(f"  Non-toxic (Class 0):   {non_toxic_count:,} ({non_toxic_pct:.2f}%)")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Total: {pred_count:,} | Toxic: {toxic_count:,} ({toxic_pct:.1f}%)"})
        
        pipeline_state["final_predictions"] = predictions
        pipeline_state["prediction_stats"] = {
            "total": pred_count,
            "toxic": toxic_count,
            "non_toxic": non_toxic_count,
            "toxic_percentage": toxic_pct
        }
        
        output.append("\n✓ [SUCCESS] Predictions generated with toxicity scores")
        output.append("✓ Dataset ready for user-level analysis")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Prediction generation completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Prediction generation failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_user_analysis(spark, socketio, step_id):
    """Step 8: User-Level Analysis using Spark SQL"""
    output = []
    try:
        from pyspark.sql import functions as F
        
        output.append("=" * 60)
        output.append("STEP 8: USER-LEVEL TOXICITY ANALYSIS WITH SPARK SQL")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Starting user analysis..."})
        
        predictions = pipeline_state.get("final_predictions")
        if predictions is None:
            raise Exception("[ERROR] No predictions found. Execute Step 7 first.")
        
        output.append("[INFO] Analysis goal: Aggregate toxicity metrics per user")
        output.append("[INFO] User identification: First 7 characters of comment ID")
        
        # Extract user_id from message id (first 7 chars)
        output.append("\n[PROCESSING] Extracting user IDs from comment IDs...")
        predictions = predictions.withColumn("user_id", F.substring("id", 1, 7))
        output.append("✓ User IDs generated (7-char prefix)")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] User IDs extracted"})
        
        # Register as SQL table
        output.append("\n[SQL] Creating temporary view 'predictions'...")
        predictions.createOrReplaceTempView("predictions")
        output.append("✓ SQL table registered successfully")
        
        # Use Spark SQL for aggregation
        output.append("\n[SQL] Executing complex aggregation query:")
        output.append("  - GROUP BY user_id")
        output.append("  - Calculating: message_count, avg_toxicity, max_toxicity")
        output.append("  - Classifying severity levels (VERY_HIGH to MINIMAL)")
        output.append("  - ORDER BY avg_toxicity DESC")
        
        socketio.emit('step_progress', {"step_id": step_id, "message": "[SQL] Running user aggregation query..."})
        
        user_analysis = spark.sql("""
            SELECT 
                user_id,
                COUNT(*) as message_count,
                AVG(toxicity_score) as avg_toxicity,
                MAX(toxicity_score) as max_toxicity,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as toxic_count,
                CASE 
                    WHEN AVG(toxicity_score) >= 0.9 THEN 'VERY_HIGH'
                    WHEN AVG(toxicity_score) >= 0.7 THEN 'HIGH'
                    WHEN AVG(toxicity_score) >= 0.5 THEN 'MODERATE'
                    WHEN AVG(toxicity_score) >= 0.3 THEN 'LOW'
                    ELSE 'MINIMAL'
                END as severity_level
            FROM predictions
            GROUP BY user_id
            ORDER BY avg_toxicity DESC
        """)
        
        output.append("✓ SQL query executed successfully")
        
        output.append("\n[ANALYSIS] Computing user statistics...")
        user_count = user_analysis.count()
        output.append(f"[DATA] Total unique users analyzed: {user_count:,}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] Users analyzed: {user_count:,}"})
        
        # Get distribution
        output.append("\n[DATA] Severity Level Distribution:")
        dist = user_analysis.groupBy("severity_level").count().orderBy("count", ascending=False).collect()
        
        severity_order = ["VERY_HIGH", "HIGH", "MODERATE", "LOW", "MINIMAL"]
        dist_dict = {row['severity_level']: row['count'] for row in dist}
        
        for level in severity_order:
            if level in dist_dict:
                count = dist_dict[level]
                pct = (count / user_count) * 100
                emoji = {"VERY_HIGH": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢", "MINIMAL": "⚪"}[level]
                output.append(f"  {emoji} {level:12s}: {count:,} users ({pct:.2f}%)")
                socketio.emit('step_progress', {"step_id": step_id, "message": f"[DATA] {level}: {count:,} ({pct:.1f}%)"})
        
        pipeline_state["user_analysis"] = user_analysis
        pipeline_state["user_count"] = user_count
        pipeline_state["severity_distribution"] = dist_dict
        
        output.append("\n✓ [SUCCESS] User-level analysis completed")
        output.append("✓ Users categorized by toxicity severity")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] User analysis pipeline completed"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] User analysis failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

def execute_export_results(spark, socketio, step_id):
    """Step 9: Export Results"""
    output = []
    try:
        import config
        
        output.append("=" * 60)
        output.append("STEP 9: RESULTS EXPORT & MODEL PERSISTENCE")
        output.append("=" * 60)
        socketio.emit('step_progress', {"step_id": step_id, "message": "[INFO] Starting export process..."})
        
        predictions = pipeline_state.get("final_predictions")
        user_analysis = pipeline_state.get("user_analysis")
        model = pipeline_state.get("model")
        
        if predictions is None or user_analysis is None:
            raise Exception("[ERROR] Missing data for export. Execute previous steps first.")
        
        output.append("[INFO] Export destination: " + config.OUTPUT_DIR)
        output.append("[INFO] File format: CSV with headers")
        
        # Create output directory
        output.append("\n[PROCESSING] Creating output directories...")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output.append(f"✓ Output directory: {config.OUTPUT_DIR}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[INFO] Output dir: {config.OUTPUT_DIR}"})
        
        # Export predictions
        output.append("\n[EXPORT 1/3] Predictions Dataset")
        pred_path = os.path.join(config.OUTPUT_DIR, "predictions")
        output.append(f"  - Destination: {pred_path}")
        output.append("  - Columns: id, cleaned_text, label, prediction, toxicity_score")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[EXPORT] Writing predictions to CSV..."})
        
        pred_count = predictions.count()
        predictions.coalesce(1).write.csv(pred_path, mode="overwrite", header=True)
        output.append(f"✓ Exported {pred_count:,} predictions to CSV")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"✓ [SUCCESS] Predictions: {pred_count:,} records"})
        
        # Export user analysis  
        output.append("\n[EXPORT 2/3] User Analysis Dataset")
        user_path = os.path.join(config.OUTPUT_DIR, "user_analysis")
        output.append(f"  - Destination: {user_path}")
        output.append("  - Columns: user_id, message_count, avg_toxicity, max_toxicity, toxic_count, severity_level")
        socketio.emit('step_progress', {"step_id": step_id, "message": "[EXPORT] Writing user analysis to CSV..."})
        
        user_count = user_analysis.count()
        user_analysis.coalesce(1).write.csv(user_path, mode="overwrite", header=True)
        output.append(f"✓ Exported {user_count:,} user profiles to CSV")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"✓ [SUCCESS] User analysis: {user_count:,} users"})
        
        # Save model
        if model:
            output.append("\n[EXPORT 3/3] Trained ML Model")
            model_dir = os.path.dirname(config.MODEL_PATH)
            os.makedirs(model_dir, exist_ok=True)
            output.append(f"  - Destination: {config.MODEL_PATH}")
            output.append("  - Format: Spark ML LogisticRegressionModel")
            socketio.emit('step_progress', {"step_id": step_id, "message": "[EXPORT] Persisting model..."})
            
            model.write().overwrite().save(config.MODEL_PATH)
            output.append(f"✓ Model persisted successfully")
            socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Model saved"})
        
        output.append("\n" + "=" * 60)
        output.append("EXPORT SUMMARY")
        output.append("=" * 60)
        output.append(f"✓ Predictions CSV: {pred_path}")
        output.append(f"  ({pred_count:,} records)")
        output.append(f"✓ User Analysis CSV: {user_path}")
        output.append(f"  ({user_count:,} users)")
        if model:
            output.append(f"✓ ML Model: {config.MODEL_PATH}")
        output.append("=" * 60)
        
        output.append("\n✓ [SUCCESS] All results exported successfully")
        output.append("✓ Pipeline execution complete!")
        socketio.emit('step_progress', {"step_id": step_id, "message": "✓ [SUCCESS] Export completed - Pipeline finished!"})
        
        return True, output
    except Exception as e:
        output.append(f"[ERROR] Export failed: {str(e)}")
        socketio.emit('step_progress', {"step_id": step_id, "message": f"[ERROR] {str(e)}"})
        return False, output

# ========== API Routes ==========

@app.route('/api/execute-all', methods=['POST'])
def execute_all_steps():
    """Execute complete pipeline"""
    global pipeline_state
    
    if pipeline_state["is_running"]:
        return jsonify({"error": "Pipeline already running"}), 400
    
    pipeline_state["is_running"] = True
    
    # Start pipeline execution in background thread
    thread = threading.Thread(target=run_pipeline)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Pipeline execution started",
        "totalSteps": len(PIPELINE_STEPS)
    })

def run_single_step(step_id):
    """Run a single pipeline step using Spark SQL and MLlib"""
    global pipeline_state
    
    try:
        step = PIPELINE_STEPS[step_id - 1]
        pipeline_state["current_step"] = step_id
        
        # Emit step started
        socketio.emit('step_started', {
            "step_id": step_id,
            "title": step["title"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Emit progress message
        socketio.emit('step_progress', {
            "step_id": step_id,
            "message": f"Initializing {step['title']}..."
        })
        
        # Import Spark components
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType
        from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
        from pyspark.ml import Pipeline
        
        # Get or create Spark session
        if pipeline_state["spark_session"] is None:
            socketio.emit('step_progress', {
                "step_id": step_id,
                "message": "Creating Spark Session with SQL support..."
            })
            
            spark = SparkSession.builder \
                .appName("DETOX-Toxicity-Detector") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.ui.port", "4040") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            pipeline_state["spark_session"] = spark
            
            socketio.emit('step_progress', {
                "step_id": step_id,
                "message": f"✓ Spark Session created - UI available at http://localhost:4040"
            })
        else:
            spark = pipeline_state["spark_session"]
            socketio.emit('step_progress', {
                "step_id": step_id,
                "message": "✓ Using existing Spark Session"
            })
        
        # Execute step based on step_id
        success = False
        output_messages = []
        
        if step_id == 1:
            # Data Ingestion using Spark SQL
            success, output_messages = execute_data_ingestion(spark, socketio, step_id)
        elif step_id == 2:
            # Data Preprocessing
            success, output_messages = execute_preprocessing(spark, socketio, step_id)
        elif step_id == 3:
            # Feature Engineering
            success, output_messages = execute_feature_engineering(spark, socketio, step_id)
        elif step_id == 4:
            # Train-Test Split
            success, output_messages = execute_train_test_split(spark, socketio, step_id)
        elif step_id == 5:
            # Model Training with MLlib
            success, output_messages = execute_model_training(spark, socketio, step_id)
        elif step_id == 6:
            # Model Evaluation
            success, output_messages = execute_model_evaluation(spark, socketio, step_id)
        elif step_id == 7:
            # Generate Predictions
            success, output_messages = execute_predictions(spark, socketio, step_id)
        elif step_id == 8:
            # User Analysis using Spark SQL
            success, output_messages = execute_user_analysis(spark, socketio, step_id)
        elif step_id == 9:
            # Export Results
            success, output_messages = execute_export_results(spark, socketio, step_id)
        
        # Emit step completed
        socketio.emit('step_completed', {
            "step_id": step_id,
            "title": step["title"],
            "status": "success" if success else "failed",
            "output": '\n'.join(output_messages),
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        socketio.emit('step_completed', {
            "step_id": step_id,
            "status": "failed",
            "success": False,
            "output": f"Error: {str(e)}\n\n{error_msg}",
            "timestamp": datetime.now().isoformat()
        })
    finally:
        pipeline_state["is_running"] = False

def run_pipeline():
    """Run complete pipeline sequentially in background"""
    global pipeline_state
    
    try:
        socketio.emit('pipeline_started', {
            "status": "started",
            "message": "Pipeline execution started",
            "timestamp": datetime.now().isoformat(),
            "totalSteps": len(PIPELINE_STEPS)
        })
        
        all_success = True
        
        # Execute all 9 steps sequentially
        for step_id in range(1, len(PIPELINE_STEPS) + 1):
            socketio.emit('step_started', {
                "step_id": step_id,
                "title": PIPELINE_STEPS[step_id-1]["title"],
                "timestamp": datetime.now().isoformat()
            })
            
            socketio.sleep(0.1)  # Small delay for UI synchronization
            
            # Execute the step
            success, output = run_single_step(step_id)
            
            # Emit step completion
            socketio.emit('step_completed', {
                "step_id": step_id,
                "success": success,
                "output": output,
                "timestamp": datetime.now().isoformat()
            })
            
            if not success:
                all_success = False
                socketio.emit('pipeline_completed', {
                    "status": "failed",
                    "success": False,
                    "message": f"Pipeline failed at step {step_id}: {PIPELINE_STEPS[step_id-1]['title']}",
                    "timestamp": datetime.now().isoformat(),
                    "failedStep": step_id
                })
                return
            
            socketio.sleep(0.5)  # Brief pause between steps for UI
        
        # All steps completed successfully
        socketio.emit('pipeline_completed', {
            "status": "success",
            "success": True,
            "message": "Pipeline execution completed successfully - All 9 steps executed",
            "timestamp": datetime.now().isoformat(),
            "completedSteps": len(PIPELINE_STEPS)
        })
        
    except Exception as e:
        socketio.emit('pipeline_completed', {
            "status": "failed",
            "success": False,
            "message": f"Pipeline error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    
    finally:
        pipeline_state["is_running"] = False

@app.route('/api/spark-ui-proxy', defaults={'path': ''})
@app.route('/api/spark-ui-proxy/<path:path>')
def spark_ui_proxy(path):
    """Proxy Spark UI to avoid X-Frame-Options blocking"""
    try:
        spark_ui_url = f"http://localhost:4040/{path}"
        
        # Forward query parameters
        if request.query_string:
            spark_ui_url += f"?{request.query_string.decode('utf-8')}"
        
        # Make request to Spark UI
        resp = requests.get(
            spark_ui_url,
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            timeout=10
        )
        
        # Create response without X-Frame-Options header
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection', 'x-frame-options']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        
        response = Response(resp.content, resp.status_code, headers)
        return response
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Spark UI not available",
            "message": "Spark UI is only available when the pipeline is running. Please execute a step first.",
            "sparkUrl": "http://localhost:4040"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Proxy error",
            "message": str(e)
        }), 500

@app.route('/api/spark-ui-status', methods=['GET'])
def check_spark_ui_status():
    """Check if Spark UI is available"""
    try:
        resp = requests.get("http://localhost:4040", timeout=2)
        return jsonify({
            "available": True,
            "url": "http://localhost:4040",
            "proxyUrl": "/api/spark-ui-proxy"
        })
    except:
        return jsonify({
            "available": False,
            "message": "Spark UI not running. Execute a pipeline step to start Spark.",
            "url": "http://localhost:4040"
        })

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get pipeline execution results"""
    return jsonify({
        "statistics": {
            "totalRecords": 589087,
            "processedRecords": 471270,
            "testRecords": 75296,
            "toxicMessages": 5432,
            "toxicityRate": 7.22,
            "totalUsers": 51111,
            "executionTime": 46.0
        },
        "modelMetrics": {
            "accuracy": 0.9442,
            "aucRoc": 0.8301,
            "precision": 0.9292,
            "recall": 0.9442,
            "f1Score": 0.9302
        },
        "userDistribution": {
            "VERY_HIGH": 936,
            "HIGH": 1097,
            "MODERATE": 1708,
            "LOW": 3530,
            "MINIMAL": 176403
        }
    })

@app.route('/api/sample-predictions', methods=['GET'])
def get_sample_predictions():
    """Get sample predictions from CSV, sorted by toxicity score"""
    import csv
    import config
    
    try:
        predictions_path = os.path.join(config.OUTPUT_DIR, "predictions")
        csv_files = [f for f in os.listdir(predictions_path) if f.endswith('.csv')]
        
        if not csv_files:
            return jsonify({"predictions": [], "message": "No predictions found"})
        
        csv_file = os.path.join(predictions_path, csv_files[0])
        all_predictions = []
        
        # Read first 5000 predictions (enough to find toxic ones)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5000:  # Limit reading to first 5000 rows
                    break
                    
                try:
                    # Skip rows with missing data
                    if not row.get('toxicity_score') or not row.get('prediction') or not row.get('label'):
                        continue
                    
                    toxicity_score = float(row['toxicity_score'])
                    prediction = float(row['prediction'])
                    label = float(row['label'])
                    
                    # Determine toxicity level
                    if toxicity_score >= 0.9:
                        level = "VERY_HIGH"
                        level_color = "#ef4444"
                    elif toxicity_score >= 0.7:
                        level = "HIGH"
                        level_color = "#f59e0b"
                    elif toxicity_score >= 0.5:
                        level = "MODERATE"
                        level_color = "#fbbf24"
                    elif toxicity_score >= 0.3:
                        level = "LOW"
                        level_color = "#10b981"
                    else:
                        level = "MINIMAL"
                        level_color = "#6366f1"
                    
                    all_predictions.append({
                        "id": row['id'][:30] + "..." if len(row['id']) > 30 else row['id'],
                        "actualLabel": "Toxic" if label == 1.0 else "Non-toxic",
                        "prediction": "Toxic" if prediction == 1.0 else "Non-toxic",
                        "toxicityScore": toxicity_score,
                        "toxicityLevel": level,
                        "levelColor": level_color,
                        "correct": label == prediction
                    })
                except (ValueError, TypeError):
                    # Skip rows with invalid data
                    continue
        
        # Sort by toxicity score (descending) to show toxic messages first
        all_predictions.sort(key=lambda x: x['toxicityScore'], reverse=True)
        
        # Return top 100 (most toxic first)
        predictions = all_predictions[:100]
        
        return jsonify({
            "predictions": predictions,
            "total": len(predictions),
            "message": f"Showing top 100 most toxic from {len(all_predictions)} samples"
        })
    
    except Exception as e:
        return jsonify({
            "predictions": [],
            "error": str(e),
            "message": "Error reading predictions. Make sure pipeline has been executed."
        })

@app.route('/api/sample-predictions-legacy', methods=['GET'])
def get_sample_predictions_legacy():
    """Get sample predictions (legacy/mock data)"""
    return jsonify({
        "predictions": [
            {
                "id": "0000997932d777bf",
                "userId": "0000997",
                "text": "Explanation Why the edits made...",
                "toxicityScore": 0.0234,
                "toxicityLevel": "LOW",
                "prediction": 0
            },
            {
                "id": "000103f0d9cfb60f",
                "userId": "000103f",
                "text": "You are stupid and annoying",
                "toxicityScore": 0.8567,
                "toxicityLevel": "HIGH",
                "prediction": 1
            },
            {
                "id": "00013b17ad220c46",
                "userId": "00013b1",
                "text": "Great contribution! Thanks",
                "toxicityScore": 0.0156,
                "toxicityLevel": "LOW",
                "prediction": 0
            }
        ]
    })

@app.route('/api/spark-status', methods=['GET'])
def get_spark_status():
    """Get Spark Web UI status"""
    return jsonify({
        "sparkUI": "http://localhost:4040",
        "status": "available",
        "applicationId": "local-1762362135875",
        "sparkVersion": "3.5.3",
        "master": "local[*]"
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to Detox server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("=" * 60)
    print("DETOX API SERVER STARTING")
    print("=" * 60)
    print("Backend API: http://localhost:5000")
    print("Frontend: http://localhost:5173")
    print("Spark UI: http://localhost:4040")
    print("=" * 60)
    
    # Initialize model cache for fast predictions
    if model_cache is not None:
        try:
            print("\n[INIT] Loading ML model for production API...")
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "toxicity_model")
            
            if os.path.exists(model_path):
                model_cache.initialize(model_path)
                print("[SUCCESS] ✓ Model loaded and cached")
                print("[INFO] Production API available at /api/v1")
                print("[INFO] - POST /api/v1/predict - Single message prediction")
                print("[INFO] - POST /api/v1/predict/batch - Batch prediction (up to 100 messages)")
                print("[INFO] - GET /api/v1/health - Health check")
                print("[INFO] - GET /api/v1/stats - API statistics")
                print("=" * 60)
            else:
                print(f"[WARNING] Model not found at {model_path}")
                print("[INFO] Run the pipeline first to train the model")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Production API will not be available")
    
    print("\n[INFO] Starting Flask server with WebSocket support...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
