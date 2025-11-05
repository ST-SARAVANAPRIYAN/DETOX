# Detox Project - Complete File Structure

## 📁 Project Organization

```
ssfproject/
│
├── 📄 README.md                    # Comprehensive project documentation
├── 📄 QUICKSTART.md                # 5-minute quick start guide
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 🔧 Configuration & Setup
│   ├── config.py                   # Central configuration file
│   ├── setup.sh                    # Automated setup script
│   └── check_system.py             # System verification script
│
├── 🧠 Core Application Modules
│   ├── main.py                     # Main application entry point
│   ├── data_ingestion.py          # Data loading & validation
│   ├── preprocessing.py           # Text preprocessing & features
│   ├── model.py                   # ML model training & prediction
│   ├── user_analysis.py           # User-level analytics
│   └── utils.py                   # Utility functions
│
├── 📊 Analysis & Visualization
│   └── detox_analysis.ipynb       # Jupyter notebook for interactive analysis
│
├── 📂 data/                        # Input data directory
│   ├── chat_data.csv              # Main dataset (user-provided)
│   └── .gitkeep                   # Keep directory in git
│
├── 📂 output/                      # Results directory
│   ├── toxicity_predictions.csv   # Message-level predictions
│   ├── user_toxicity_levels.csv   # User-level aggregates
│   └── .gitkeep
│
└── 📂 models/                      # Trained models directory
    ├── toxicity_model/            # Saved Spark ML model
    └── .gitkeep
```

---

## 📚 File Descriptions

### Configuration Files

#### `config.py`
Central configuration for all project settings:
- Paths (data, output, models)
- Spark configuration (memory, ports)
- Model parameters (iterations, regularization)
- Toxicity thresholds

#### `requirements.txt`
Python dependencies:
- pyspark==3.5.0
- pandas, numpy
- matplotlib, seaborn
- jupyter

---

### Core Modules

#### `main.py` (Main Application)
**Purpose:** Complete pipeline execution with Spark Web UI
**Functions:**
- `create_spark_session()` - Initialize Spark with Web UI
- `main()` - Execute full pipeline

**Pipeline Steps:**
1. Data Ingestion
2. Data Preprocessing
3. Train-Test Split
4. Model Training
5. Model Evaluation
6. Predictions
7. User Analysis
8. Save Results

**Usage:**
```bash
python main.py
```

---

#### `data_ingestion.py`
**Purpose:** Load and validate toxic comments dataset

**Class:** `DataIngestion`

**Methods:**
- `read_csv_data(file_path)` - Read CSV with schema
- `validate_data(df)` - Clean and validate data
- `get_data_statistics(df)` - Calculate dataset statistics

**Features:**
- Schema enforcement
- Null value handling
- Data quality checks
- Statistical analysis

---

#### `preprocessing.py`
**Purpose:** Text preprocessing and feature engineering

**Class:** `TextPreprocessor`

**Methods:**
- `clean_text_udf()` - UDF for text cleaning
- `preprocess_text(df)` - Clean and normalize text
- `build_feature_pipeline()` - Create ML pipeline
- `fit_transform_features(df)` - Extract TF-IDF features
- `create_label_column(df)` - Prepare labels

**Pipeline Stages:**
1. Tokenization
2. Stop words removal
3. HashingTF (Term Frequency)
4. IDF (Inverse Document Frequency)

---

#### `model.py`
**Purpose:** Train and evaluate toxicity classification model

**Class:** `ToxicityClassifier`

**Methods:**
- `train_model(train_df)` - Train Logistic Regression
- `evaluate_model(test_df)` - Calculate metrics
- `predict(df)` - Generate predictions
- `save_model(path)` - Save trained model
- `load_model(path)` - Load saved model

**Metrics:**
- AUC-ROC
- AUC-PR
- Accuracy
- Precision
- Recall
- F1 Score

---

#### `user_analysis.py`
**Purpose:** Aggregate toxicity metrics at user level

**Class:** `UserToxicityAnalyzer`

**Methods:**
- `aggregate_user_toxicity(predictions_df)` - User aggregation
- `get_top_toxic_users(user_aggregates, n)` - Top N users
- `get_user_statistics(user_aggregates)` - Overall stats
- `create_detailed_message_report(predictions_df)` - Detailed report

**User Metrics:**
- Total messages
- Average toxicity score
- Max/min toxicity score
- Toxic message counts by severity
- User toxicity level
- Message ID list

---

#### `utils.py`
**Purpose:** Helper functions for analysis and visualization

**Functions:**
- `display_data_sample()` - Show data samples
- `get_toxicity_distribution()` - Calculate label distribution
- `plot_toxicity_distribution()` - Visualize distribution
- `calculate_class_weights()` - Handle imbalanced data
- `print_confusion_matrix()` - Display confusion matrix
- `print_spark_config()` - Show Spark configuration
- `create_summary_report()` - Generate report
- `export_to_excel()` - Export to Excel

---

### Analysis & Visualization

#### `detox_analysis.ipynb`
**Purpose:** Interactive analysis with Jupyter Notebook

**Sections:**
1. Setup & Initialization
2. Data Loading & Exploration
3. Data Preprocessing
4. Feature Engineering
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Predictions
9. User-Level Analysis
10. Visualizations
11. Export Results
12. Summary

**Visualizations:**
- Toxicity level distribution
- Average toxicity score histogram
- Top toxic users bar chart
- Messages vs toxicity scatter plot
- Model performance metrics

---

### Setup Scripts

#### `setup.sh`
**Purpose:** Automated project setup

**Steps:**
1. Check Python version
2. Verify Java installation
3. Create directories
4. Install dependencies
5. Verify PySpark

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

---

#### `check_system.py`
**Purpose:** Verify system requirements and configuration

**Checks:**
1. Python version (3.8+)
2. Java installation
3. Python packages
4. Project directories
5. Dataset presence
6. Port availability (4040)

**Usage:**
```bash
python check_system.py
```

---

## 🎯 Execution Flow

### Standard Flow (main.py)
```
Start
  ↓
Initialize Spark Session (with Web UI)
  ↓
Load CSV Data → Validate → Statistics
  ↓
Clean Text → Tokenize → Remove Stop Words
  ↓
Extract Features (TF-IDF)
  ↓
Split Data (80/20)
  ↓
Train Logistic Regression Model
  ↓
Evaluate on Test Set
  ↓
Predict on Full Dataset
  ↓
Aggregate User-Level Metrics
  ↓
Export Results (CSV)
  ↓
Save Model
  ↓
Display Summary
  ↓
Keep Spark UI Running
  ↓
End
```

---

## 📊 Output Files

### 1. toxicity_predictions.csv
Contains prediction for each message:

**Columns:**
- id, user_id, comment_text
- toxicity_score (0.0-1.0)
- toxicity_level (LOW/MODERATE/HIGH/SEVERE)
- prediction (0 or 1)
- Original labels (toxic, severe_toxic, etc.)

**Sample:**
```csv
id,user_id,comment_text,toxicity_score,toxicity_level,prediction
abc123,abc12345,"Great game!",0.0234,LOW,0
def456,def45678,"You're terrible",0.8567,HIGH,1
```

---

### 2. user_toxicity_levels.csv
Contains aggregated user metrics:

**Columns:**
- user_id
- total_messages
- avg_toxicity_score
- max/min_toxicity_score
- severe/high/moderate/low_toxic_count
- toxic_messages_count
- toxicity_percentage
- user_toxicity_level
- message_ids_list

**Sample:**
```csv
user_id,total_messages,avg_toxicity_score,user_toxicity_level,toxic_messages_count
abc12345,25,0.1234,MINIMAL,2
def45678,50,0.7891,HIGH,35
```

---

## 🌐 Apache Spark Web UI

**Access:** http://localhost:4040

**Key Pages:**

1. **Jobs Tab**
   - View all Spark jobs
   - Job duration and status
   - Stage breakdown

2. **Stages Tab**
   - Detailed stage information
   - Task metrics
   - Shuffle read/write

3. **Storage Tab**
   - Cached RDDs/DataFrames
   - Memory usage
   - Disk usage

4. **Environment Tab**
   - Spark properties
   - System properties
   - Classpath entries

5. **Executors Tab**
   - Executor metrics
   - Memory usage per executor
   - Task statistics

6. **SQL Tab**
   - DataFrame operations
   - Query execution plans
   - Physical plans

---

## 🔧 Customization Guide

### Change Memory Settings
Edit `config.py`:
```python
SPARK_EXECUTOR_MEMORY = "2g"  # Reduce for smaller systems
SPARK_DRIVER_MEMORY = "2g"
```

### Adjust Toxicity Thresholds
Edit `config.py`:
```python
TOXICITY_THRESHOLD = 0.4        # More sensitive
HIGH_TOXICITY_THRESHOLD = 0.6
SEVERE_TOXICITY_THRESHOLD = 0.8
```

### Modify Model Parameters
Edit `config.py`:
```python
MAX_ITERATIONS = 150           # More training iterations
REGULARIZATION_PARAM = 0.001   # Less regularization
MAX_FEATURES = 20000           # More features
```

### Change Ports
Edit `config.py`:
```python
SPARK_UI_PORT = 4041          # Use different port
```

---

## 📝 Best Practices

### For Development
1. Use `detox_analysis.ipynb` for exploration
2. Test with small dataset first
3. Monitor Spark Web UI for performance
4. Cache DataFrames for repeated use

### For Production
1. Use `main.py` for batch processing
2. Adjust memory based on data size
3. Enable logging for debugging
4. Save models for reuse

### For Performance
1. Partition data appropriately
2. Use `.cache()` on frequently used DataFrames
3. Reduce shuffle operations
4. Monitor executor memory

---

## 🎓 Learning Resources

### Included in Project
- README.md - Full documentation
- QUICKSTART.md - Quick setup guide
- Code comments - Inline documentation
- Jupyter notebook - Interactive tutorial

### External Resources
- PySpark Documentation
- Spark MLlib Guide
- Kaggle Dataset
- Apache Spark Web UI Guide

---

**Project Complete! Ready for deployment! 🚀**

*For questions: See README.md or contact project maintainer*
