#!/bin/bash

# Script to run the Insider Threat Detection API server
# Usage: ./run.sh [--test-model] [--no-reload]

# Parse command line arguments
TEST_MODEL=false
RELOAD="--reload"

while [[ $# -gt 0 ]]; do
  case $1 in
    --test-model)
      TEST_MODEL=true
      shift
      ;;
    --no-reload)
      RELOAD=""
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run.sh [--test-model] [--no-reload]"
      exit 1
      ;;
  esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    bash setup.sh
fi

# Activate the virtual environment
source venv/bin/activate

# Run the model test if requested
if [ "$TEST_MODEL" = true ]; then
    echo "Running model loading test..."
    python test_model_loading.py
    exit $?
fi

# Set optimized CPU parameters if on MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected MacOS, setting optimized parameters for Apple Silicon..."
    export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
    export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
fi

# Start the server
echo "Starting Insider Threat Detection API server..."
echo "Server will be available at http://localhost:8000"
echo "API docs will be available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 $RELOAD

# Note: The --reload flag is for development only and should be removed in production 