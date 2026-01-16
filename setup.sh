#!/bin/bash
# UIDAI Aadhaar Intelligence System - Setup Script
# Usage: bash setup.sh

set -e

echo "================================================"
echo "🆔 UIDAI Aadhaar Intelligence System Setup"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

echo "📌 Checking Python version..."
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Python $PYTHON_VERSION detected"
else
    echo "❌ Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "📌 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "📌 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📌 Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "📌 Creating project directories..."
mkdir -p models data config utils

# Check if init files exist
touch config/__init__.py
touch utils/__init__.py

# Generate sample data
echo ""
echo "📌 Generating sample data..."
python -c "from utils.data_generator import save_sample_data; save_sample_data()"

# Train model
echo ""
echo "📌 Training LSTM model..."
python train_model.py

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "To start the application, run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "The dashboard will be available at:"
echo "  http://localhost:8501"
echo ""
