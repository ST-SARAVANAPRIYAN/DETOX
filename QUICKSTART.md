# Quick Start Guide - Detox

## 🚀 5-Minute Quick Start

### Step 1: Setup (First Time Only)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### Step 2: Prepare Data

Place your dataset in the `data/` directory:
```bash
# Your file should be named:
data/chat_data.csv
```

**Required CSV columns:**
- `id` - Message ID
- `comment_text` - Message content
- `toxic` - Binary label (0 or 1)
- `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` - Additional labels

### Step 3: Run the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run application
python main.py

# Or simply use:
./run.sh
```

The application will:
1. ✅ Load and validate data
2. ✅ Preprocess text
3. ✅ Train the model
4. ✅ Generate predictions
5. ✅ Create user analytics
6. ✅ Save results

### Step 4: Access Spark Web UI

Open your browser:
```
http://localhost:4040
```

### Step 5: View Results

Check the `output/` directory:
- `toxicity_predictions.csv` - Message-level predictions
- `user_toxicity_levels.csv` - User-level aggregates

---

## 🎓 Using Jupyter Notebook

For interactive analysis:

```bash
# Activate virtual environment first
source venv/bin/activate

# Then run Jupyter
jupyter notebook detox_analysis.ipynb

# Or simply use:
./run_notebook.sh
```

Then run cells sequentially (Shift+Enter).

---

## 📊 Expected Results

### Console Output Example:

```
============================================================
DETOX - CHAT MESSAGE TOXICITY DETECTOR
============================================================
Student: SARAVANA PRIYAN S T
Reg No: 927623BAD100
============================================================

[INFO] Creating Spark Session...
[INFO] Spark Web UI: http://localhost:4040

============================================================
STEP 1: DATA INGESTION
============================================================
[INFO] Successfully loaded 223549 records

============================================================
DATASET STATISTICS
============================================================
Total Records: 223549
Toxicity Distribution:
  toxic: 21384 (9.57%)
  severe_toxic: 1595 (0.71%)
  ...

============================================================
MODEL EVALUATION METRICS
============================================================
AUC-ROC:          0.9726
Accuracy:         0.9543
Precision:        0.9412
Recall:           0.9201
F1 Score:         0.9305
============================================================

✓ DETOX PIPELINE COMPLETED SUCCESSFULLY!
```

---

## 🔍 Sample Output Data

### toxicity_predictions.csv
```csv
id,user_id,comment_text,toxicity_score,toxicity_level,prediction
0000997932d777bf,0000997,Why the edits...,0.0234,LOW,0
000103f0d9cfb60f,000103f,"You are stupid",0.8567,HIGH,1
```

### user_toxicity_levels.csv
```csv
user_id,total_messages,avg_toxicity_score,user_toxicity_level,toxic_messages_count
0000997,15,0.1245,MINIMAL,1
000103f,8,0.7823,HIGH,6
```

---

## ⚙️ Configuration Options

Edit `config.py` before running:

```python
# Adjust memory for your system
SPARK_EXECUTOR_MEMORY = "4g"  # Reduce if needed
SPARK_DRIVER_MEMORY = "4g"

# Change thresholds
TOXICITY_THRESHOLD = 0.5       # Default: 0.5
HIGH_TOXICITY_THRESHOLD = 0.7   # Default: 0.7

# Model tuning
MAX_ITERATIONS = 100            # More = better accuracy
REGULARIZATION_PARAM = 0.01     # Prevent overfitting
```

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'pyspark'"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install pyspark
```

### Issue: "JAVA_HOME is not set"
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### Issue: "FileNotFoundError: data/chat_data.csv"
- Ensure your CSV file is in the `data/` directory
- Check the filename matches exactly

### Issue: Out of Memory
- Reduce memory in `config.py`
- Sample your dataset: `df.sample(fraction=0.1)`

---

## 📈 Monitoring with Spark Web UI

### Key Tabs:

1. **Jobs** - View completed/running jobs
2. **Stages** - See task progress
3. **Storage** - Check cached data
4. **SQL** - View query plans
5. **Environment** - Check configuration

### What to Monitor:

- ✅ Job completion status
- ✅ Stage duration
- ✅ Memory usage
- ✅ Task failures (should be 0)
- ✅ Data shuffle size

---

## 🎯 Testing with Sample Data

Create a small test file:

```bash
head -n 1000 data/chat_data.csv > data/test_sample.csv
```

Update `config.py`:
```python
INPUT_DATA_PATH = os.path.join(DATA_DIR, "test_sample.csv")
```

Run:
```bash
python main.py
```

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Review error messages in console
3. Check Spark Web UI for job failures
4. Verify Java and Python versions
5. Ensure dataset format is correct

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Java 8/11 installed
- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset in `data/chat_data.csv`
- [ ] Directories created (data/, output/, models/)
- [ ] Port 4040 available for Spark UI

---

**Ready to go! Activate venv and run `python main.py` or use `./run.sh` to start detecting toxicity! 🚀**
