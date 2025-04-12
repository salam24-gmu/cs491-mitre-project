# Insider Threat Detection

This project uses machine learning models to detect potential insider threats through tweet analysis, incorporating both Named Entity Recognition (NER) and sentiment analysis.

## Environment Setup

1. Install Miniconda (if not already installed):
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. Create and activate the conda environment:
   ```bash
   # Create environment from yml file
   conda env create -f environment.yml

   # Activate environment
   conda activate insider-threat-detection
   ```

3. Verify Installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import transformers; print(transformers.__version__)"
   ```

## Project Structure

```
finetune/
├── data/                     # Dataset directory
│   └── synthetic_insider_threat.csv
├── notebooks/               # Jupyter notebooks
│   └── train_evaluate.ipynb  # Training and evaluation notebook
├── src/                    # Source code
│   └── models.py           # Model implementations
├── environment.yml         # Conda environment file
└── README.md              # This file
```

## Usage

1. Activate the environment:
   ```bash
   conda activate insider-threat-detection
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `notebooks/train_evaluate.ipynb` and follow the instructions in the notebook.

## Models

1. **Corporate NER Model**
   - Based on DeBERTa-v3
   - Identifies entities like ROLE, FACILITY, ACCESS_CODE, SENSITIVE_DATA

2. **Sentiment Analysis Model**
   - Based on RoBERTa
   - Classifies tweets into negative, neutral, positive sentiments
   - Helps identify potentially malicious behavior

## Models Implementation (models.py)

The `models.py` file contains the core model implementations for insider threat detection. It provides two specialized models built on top of a common base architecture:

### Key Components

1. **Base Model**
   - Provides shared functionality for all models
   - Handles data preparation and training pipeline
   - Manages device allocation (CPU/GPU)

2. **Corporate NER Model**
   - Purpose: Identify sensitive entities in text
   - Base Model: DeBERTa-v3
   - Entity Types:
     - ROLE (e.g., "CEO", "System Admin")
     - FACILITY (e.g., "Server Room", "R&D Lab")
     - ACCESS_CODE
     - SENSITIVE_DATA

3. **Sentiment Analysis Model**
   - Purpose: Detect concerning behavior through sentiment
   - Base Model: RoBERTa
   - Classifications:
     - Negative (potential risk indicators)
     - Neutral (normal communication)
     - Positive (positive engagement)

### Dependencies

The implementation relies on the following key libraries:

```
transformers>=4.30.0    # Hugging Face Transformers for pre-trained models
torch>=2.0.0           # PyTorch for deep learning operations
numpy>=1.24.0          # Numerical operations
scikit-learn>=1.2.0    # Metrics calculation
```

Key Transformers Components Used:
- `AutoModelForTokenClassification`: NER model backbone
- `AutoModelForSequenceClassification`: Sentiment analysis backbone
- `AutoTokenizer`: Text tokenization
- `Trainer`: Training pipeline
- `DataCollatorForTokenClassification`: NER data batching

### Usage Example

```python
from src.models import NERModel, SentimentAnalysisModel

# Initialize NER model
ner_model = NERModel()
entities = ner_model.predict(["Employee accessed server room at midnight"])

# Initialize Sentiment model
sentiment_model = SentimentAnalysisModel()
sentiments = sentiment_model.predict(["I hate this company's security policies"])
```

### Custom Dataset

The implementation includes a custom `InsiderThreatDataset` class that:
- Handles both NER and sentiment analysis data formats
- Integrates with PyTorch's DataLoader
- Manages text and label pairs efficiently

### Performance Metrics

The models track multiple performance metrics:
- Accuracy: Overall prediction accuracy
- F1 Score: Weighted F1 score
- Precision: Weighted precision
- Recall: Weighted recall

These metrics are computed using scikit-learn's implementation to ensure reliability.

