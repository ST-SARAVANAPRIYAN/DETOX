"""
Configuration file for Detox - Chat Message Toxicity Detector
ADI1302-SPARK SCALA FUNDAMENTALS
NAME : SARAVANA PRIYAN S T
REG NO : 927623BAD100
"""

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data Paths
INPUT_DATA_PATH = os.path.join(DATA_DIR, "chat_data.csv")
OUTPUT_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "toxicity_predictions.csv")
OUTPUT_USER_AGGREGATES_PATH = os.path.join(OUTPUT_DIR, "user_toxicity_levels.csv")
OUTPUT_STATISTICS_PATH = os.path.join(OUTPUT_DIR, "toxicity_statistics.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "toxicity_model")

# Spark Configuration
SPARK_APP_NAME = "Detox-Toxicity-Detector"
SPARK_MASTER = "local[*]"  # Use all available cores
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_DRIVER_MEMORY = "4g"
SPARK_UI_PORT = 4040

# Model Parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
MAX_ITERATIONS = 100
REGULARIZATION_PARAM = 0.01
TOXICITY_THRESHOLD = 0.35  # Lower threshold for better toxic detection

# Feature Engineering
MIN_TOKEN_LENGTH = 2
MAX_FEATURES = 10000
IDF_MIN_DOC_FREQ = 5

# Toxicity Labels
TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Toxicity Thresholds
TOXICITY_THRESHOLD = 0.5
HIGH_TOXICITY_THRESHOLD = 0.7
SEVERE_TOXICITY_THRESHOLD = 0.9
