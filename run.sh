#!/bin/bash

# Run Detox Application
# This script activates the virtual environment and runs the main application

echo "Starting Detox - Chat Message Toxicity Detector..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run main application
echo "Running main application..."
echo ""
python main.py

# Keep virtual environment activated
exec bash
