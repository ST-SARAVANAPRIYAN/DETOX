# Detox - Chat Message Toxicity Detector

**ADI1302-SPARK SCALA FUNDAMENTALS**  
**Name:** SARAVANA PRIYAN S T  
**Registration Number:** 927623BAD100

---

## 📋 Project Overview

Detox is a scalable, real-time chat message toxicity detection system built with Apache Spark and PySpark. The system identifies toxic, abusive, and harmful content in chat messages, providing user-level toxicity analytics and insights.

### 🎯 Problem Statement

**Real-Time Processing & Scalability**
- Toxic messages can emerge at high volume in chat applications, especially in gaming or streaming contexts
- Systems must process messages in milliseconds and scale to handle thousands per second

**Context Sensitivity & Subtlety**
- Many messages rely on subtle language, sarcasm, or insider slang
- Systems must distinguish harmless banter from harmful content

**Bias & Fairness Concerns**
- Toxicity is subjectively defined, varying with culture, identity, and language
- Models must be fair and unbiased across demographic groups

---

## 🏗️ Tech Stack

| Layer | Components |
|-------|-----------|
| **Data Source** | CSV (Jigsaw Toxic Comments Dataset) |
| **Data Ingestion** | PySpark (read CSV from local/HDFS) |
| **Processing Engine** | PySpark (Spark Core, DataFrame API) |
| **NLP & Model** | Spark MLlib (TF-IDF, Logistic Regression) |
| **Serving/Output** | CSV files with toxicity labels & scores |
| **Infrastructure** | Local Spark / Jupyter Notebook |
| **Monitoring** | Apache Spark Web UI (Port 4040) |

---

## 📊 Dataset

**Dataset Name:** Jigsaw Multilingual Toxic Comment Classification  
**Source:** Kaggle  
**Total Records:** 223,549 comments

### Dataset Schema

| Column | Description | Values | Example |
|--------|-------------|--------|---------|
| `id` | Unique identifier | 223,549 unique | 0000997932d777bf |
| `comment_text` | Raw text of comment | 223,549 unique | "Explanation Why the edits..." |
| `toxic` | Binary toxicity label | {0, 1} | 0 |
| `severe_toxic` | Severe toxicity label | {0, 1} | 0 |
| `obscene` | Obscenity label | {0, 1} | 0 |
| `threat` | Threat label | {0, 1} | 0 |
| `insult` | Insult label | {0, 1} | 0 |
| `identity_hate` | Identity hate label | {0, 1} | 0 |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Java 8 or 11 (required for Spark)
- 8GB RAM minimum (recommended)

### Step 1: Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Java Installation

```bash
java -version
```

If Java is not installed:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11
```

### Step 3: Prepare Dataset

Place your `chat_data.csv` file in the `data/` directory:

```bash
# Dataset should be at:
data/chat_data.csv
```

**Note:** If you don't have the dataset, download it from [Kaggle - Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

---

## 📁 Project Structure

```
ssfproject/
├── data/
│   └── chat_data.csv              # Input dataset
├── output/
│   ├── toxicity_predictions.csv   # Message-level predictions
│   └── user_toxicity_levels.csv   # User-level aggregates
├── models/
│   └── toxicity_model/            # Saved ML model
├── config.py                      # Configuration settings
├── data_ingestion.py              # Data loading & validation
├── preprocessing.py               # Text preprocessing & features
├── model.py                       # ML model training & prediction
├── user_analysis.py               # User-level analytics
├── main.py                        # Main application entry point
├── detox_analysis.ipynb           # Jupyter notebook for analysis
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🎮 Usage

### Option 1: Run Main Application (with Spark Web UI)

```bash
# Activate virtual environment
source venv/bin/activate

# Run application
python main.py

# Or use the run script
./run.sh
```

This will:
1. Load and preprocess the dataset
2. Train the toxicity detection model
3. Generate predictions for all messages
4. Aggregate user-level toxicity metrics
5. Save results to `output/` directory
6. Keep Spark Web UI running for monitoring

**Access Spark Web UI:** http://localhost:4040

### Option 2: Interactive Jupyter Notebook

```bash
# Activate virtual environment
source venv/bin/activate

# Run Jupyter notebook
jupyter notebook detox_analysis.ipynb

# Or use the run script
./run_notebook.sh
```

The notebook provides:
- Step-by-step execution
- Data visualizations
- Interactive model training
- Real-time monitoring

---

## 📈 Expected Output

### 1. Message-Level Predictions

**File:** `output/toxicity_predictions.csv`

Contains predictions for each message:

| Column | Description |
|--------|-------------|
| `id` | Message ID |
| `user_id` | User ID (extracted from message ID) |
| `comment_text` | Original message text |
| `toxicity_score` | Probability score (0.0 - 1.0) |
| `toxicity_level` | Category: LOW, MODERATE, HIGH, SEVERE |
| `prediction` | Binary prediction (0 or 1) |
| `toxic`, `severe_toxic`, etc. | Original labels |

### 2. User-Level Toxicity Aggregates

**File:** `output/user_toxicity_levels.csv`

Contains aggregated metrics per user:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `total_messages` | Total messages sent by user |
| `avg_toxicity_score` | Average toxicity score |
| `max_toxicity_score` | Highest toxicity score |
| `min_toxicity_score` | Lowest toxicity score |
| `severe_toxic_count` | Count of severe toxic messages |
| `high_toxic_count` | Count of high toxic messages |
| `moderate_toxic_count` | Count of moderate toxic messages |
| `low_toxic_count` | Count of low toxic messages |
| `toxic_messages_count` | Total toxic messages |
| `toxicity_percentage` | Percentage of toxic messages |
| `user_toxicity_level` | Overall user level: MINIMAL, LOW, MODERATE, HIGH, VERY_HIGH |
| `message_ids_list` | Comma-separated list of message IDs |

### 3. Model Performance

Expected metrics:
- **Accuracy:** ~95%+
- **AUC-ROC:** ~97%+
- **Precision:** ~94%+
- **Recall:** ~92%+
- **F1 Score:** ~93%+

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Spark Configuration
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "4g"
SPARK_UI_PORT = 4040

# Model Parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
MAX_ITERATIONS = 100
REGULARIZATION_PARAM = 0.01

# Feature Engineering
MAX_FEATURES = 10000
IDF_MIN_DOC_FREQ = 5

# Toxicity Thresholds
TOXICITY_THRESHOLD = 0.5
HIGH_TOXICITY_THRESHOLD = 0.7
SEVERE_TOXICITY_THRESHOLD = 0.9
```

---

## 🌐 Apache Spark Web UI

The Spark Web UI provides real-time monitoring:

**URL:** http://localhost:4040

### Features:
- **Jobs:** View running and completed jobs
- **Stages:** Monitor stage progress and tasks
- **Storage:** Check cached RDDs and DataFrames
- **Environment:** View Spark configuration
- **Executors:** Monitor executor metrics
- **SQL:** View DataFrame operations

### Screenshots Guide:

1. **Jobs Tab:** Shows all Spark jobs executed
2. **Stages Tab:** Displays detailed stage information
3. **Storage Tab:** Shows cached data in memory
4. **SQL Tab:** Visualizes DataFrame query plans

---

## 🧪 Testing

### Test with Sample Data

```bash
# Create small test dataset
head -n 1000 data/chat_data.csv > data/test_sample.csv

# Update config.py to use test_sample.csv
# Then run:
python main.py
```

---

## 📊 Model Pipeline

```
Input CSV
    ↓
Data Validation & Cleaning
    ↓
Text Preprocessing (lowercase, remove special chars)
    ↓
Tokenization
    ↓
Stop Words Removal
    ↓
TF-IDF Feature Extraction
    ↓
Logistic Regression Model
    ↓
Toxicity Predictions
    ↓
User-Level Aggregation
    ↓
Export Results
```

---

## 🎓 Key Features

### 1. Scalable Processing
- Handles large datasets efficiently with Spark
- Distributed processing across multiple cores
- Optimized for real-time inference

### 2. Comprehensive Analytics
- Message-level toxicity scores
- User-level aggregation
- Multiple toxicity categories
- Confidence scores

### 3. Flexible Deployment
- Local execution for development
- Easy migration to cluster mode
- Supports HDFS and cloud storage

### 4. Monitoring & Debugging
- Real-time Spark Web UI
- Detailed logging
- Performance metrics

---

## 🔍 Example Use Cases

1. **Gaming Platforms:** Monitor in-game chat for toxic behavior
2. **Social Media:** Filter harmful comments in real-time
3. **Customer Support:** Identify aggressive customer messages
4. **Content Moderation:** Automate first-pass content review
5. **User Analytics:** Track user behavior trends

---

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time streaming with Spark Structured Streaming
- [ ] Deep Learning models (BERT, RoBERTa)
- [ ] REST API for real-time predictions
- [ ] Dashboard for monitoring
- [ ] A/B testing framework
- [ ] Bias detection and mitigation
- [ ] Context-aware toxicity detection

---

## 🐛 Troubleshooting

### Issue: Spark Web UI not accessible

**Solution:**
```bash
# Check if port 4040 is in use
netstat -an | grep 4040

# Change port in config.py if needed
SPARK_UI_PORT = 4041
```

### Issue: Java not found

**Solution:**
```bash
# Set JAVA_HOME environment variable
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### Issue: Out of memory error

**Solution:**
```python
# Reduce memory in config.py
SPARK_EXECUTOR_MEMORY = "2g"
SPARK_DRIVER_MEMORY = "2g"
```

### Issue: Dataset too large

**Solution:**
```python
# Sample the dataset
df = df.sample(fraction=0.1, seed=42)
```

---

## 📚 References

1. **PySpark Documentation:** https://spark.apache.org/docs/latest/api/python/
2. **Spark MLlib Guide:** https://spark.apache.org/docs/latest/ml-guide.html
3. **Jigsaw Dataset:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
4. **Apache Spark Web UI:** https://spark.apache.org/docs/latest/web-ui.html

---

## 📝 License

This project is developed for academic purposes as part of ADI1302-SPARK SCALA FUNDAMENTALS course.

---

## 👨‍💻 Author

**SARAVANA PRIYAN S T**  
Registration Number: 927623BAD100  
Course: ADI1302-SPARK SCALA FUNDAMENTALS

---

## 🎉 Acknowledgments

- Apache Spark Community
- Kaggle for providing the dataset
- Course instructors and mentors

---

**For questions or support, please contact: saravana.priyan@example.com**
