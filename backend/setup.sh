#!/bin/bash

# Setup script for Insider Threat Detection API

echo "Setting up the Insider Threat Detection API..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create model directory if it doesn't exist
MODEL_DIR="../finetune/data/insider-threat-ner_model/model-4n4ekgnp"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Warning: Model directory $MODEL_DIR not found!"
    
    # Try to find the model in other locations
    ALT_MODEL_DIR=$(python -c "
import os
base_dir = os.path.dirname(os.path.abspath('__file__'))
model_locations = [
    os.path.join(base_dir, '..', 'finetune', 'data', 'insider-threat-ner_model', 'model-4n4ekgnp'),
    os.path.join(base_dir, 'models', 'model-4n4ekgnp'),
    os.path.join(base_dir, '..', 'models', 'model-4n4ekgnp'),
    os.path.join(base_dir, '..', 'finetune', 'data', 'insider-threat-ner_model')
]
for loc in model_locations:
    if os.path.exists(loc) and (os.path.exists(os.path.join(loc, 'config.json')) or os.path.exists(os.path.join(loc, 'model.safetensors'))):
        print(loc)
        break
")
    
    if [ -n "$ALT_MODEL_DIR" ]; then
        echo "Found model in alternative location: $ALT_MODEL_DIR"
    else
        echo "Model not found in any expected location."
        echo "Please ensure the model files are correctly placed in one of the expected directories."
    fi
else
    echo "Found model directory at $MODEL_DIR"
fi

# Make run script executable
chmod +x run.sh

# Instructions
echo
echo "Setup completed!"
echo
echo "To run the API server, use: ./run.sh"
echo "To test model loading, use: ./run.sh --test-model"
echo "To run the server without auto-reload, use: ./run.sh --no-reload"
echo
echo "API documentation will be available at http://localhost:8000/docs" 