#!/bin/bash

# Run Jupyter Notebook with virtual environment

echo "Starting Jupyter Notebook..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run Jupyter Notebook
jupyter notebook detox_analysis.ipynb
