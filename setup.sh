#!/bin/bash

# Detox Setup Script
# Quick setup for Chat Message Toxicity Detector

echo "========================================"
echo "Detox - Chat Message Toxicity Detector"
echo "Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Check Java installation
echo "[2/6] Checking Java installation..."
if command -v java &> /dev/null
then
    java_version=$(java -version 2>&1 | head -n 1)
    echo "✓ Java is installed: $java_version"
else
    echo "✗ Java is not installed"
    echo "Please install Java 8 or 11:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install openjdk-11-jdk"
    exit 1
fi
echo ""

# Create virtual environment
echo "[3/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Create directories
echo "[4/6] Creating project directories..."
mkdir -p data
mkdir -p output
mkdir -p models
echo "✓ Directories created"
echo ""

# Activate virtual environment and install dependencies
echo "[5/6] Installing Python dependencies in virtual environment..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify Spark installation
echo "[6/6] Verifying PySpark installation..."
python -c "from pyspark.sql import SparkSession; print('✓ PySpark is working correctly')"
echo ""

echo "========================================"
echo "✓ Setup Complete!"
echo "========================================"
echo ""
echo "Next Steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Place your dataset in: data/chat_data.csv"
echo "3. Run the application: python main.py"
echo "4. Or open Jupyter notebook: jupyter notebook detox_analysis.ipynb"
echo ""
echo "For more information, see README.md"
echo ""
