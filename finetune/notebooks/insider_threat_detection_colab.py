# !pip install transformers torch pandas numpy scikit-learn seaborn matplotlib tqdm

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    pipeline
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import random
import gc
from tqdm import tqdm

# GPU Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up overall seed for reproducibility
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

## 2. Model Implementations

# Constants
NER_LABELS = ['O', 'B-ROLE', 'I-ROLE', 'B-FACILITY', 'I-FACILITY', 'B-ACCESS_CODE', 'I-ACCESS_CODE']

def compute_ner_metrics(true_labels, predicted_labels):
    """
    Compute NER-specific metrics for model evaluation.
    
    Design Decision: Metric Selection
    -------------------------------
    1. Token-level metrics:
       - Precision: Accuracy of positive predictions
       - Recall: Coverage of actual positive cases
       - F1: Harmonic mean of precision and recall
    
    2. Label-wise analysis:
       - Per-class metrics for detailed analysis
       - Support values for class balance assessment
    
    Args:
        true_labels (List[List[str]]): Ground truth NER labels
        predicted_labels (List[List[str]]): Predicted NER labels
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Convert labels to flat lists for evaluation
    true_flat = [label for seq in true_labels for label in seq]
    pred_flat = [label for seq in predicted_labels for label in seq]

    # Ensure both lists have the same length
    min_len = min(len(true_flat), len(pred_flat))
    true_flat = true_flat[:min_len]
    pred_flat = pred_flat[:min_len]

    # Calculate metrics using sklearn's classification_report
    report = classification_report(
        true_flat,
        pred_flat,
        output_dict=True,
        zero_division=0
    )

    return report

class InsiderThreatDataset:
    """
    Custom dataset class for insider threat detection.
    
    Design Decision: Dataset Structure
    ------------------------------
    1. Flexible Label Handling:
       - Optional labels for inference-only use
       - Support for both NER and classification tasks
    
    2. Memory Efficiency:
       - Lazy tokenization during __getitem__
       - No redundant data storage
    
    Architecture:
    ```
    [Text Data] ─┬─► [Tokenization] ─► [Token IDs]
                 └─► [Labels] ───────► [Label IDs]
    ```
    """
    def __init__(self, texts: List[str], labels: Optional[List] = None, tokenizer=None):
        """
        Initialize dataset with texts and optional labels.
        
        Args:
            texts (List[str]): Input texts
            labels (Optional[List]): Optional labels for supervised learning
            tokenizer: HuggingFace tokenizer instance
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Design Decision: Return Format
        --------------------------
        - Dictionary format compatible with HuggingFace Trainer
        - Automatic padding and truncation handled by tokenizer
        - Labels included only if available
        """
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        
        return item

class BaseModel:
    """
    Base model class implementing common functionality.
    
    Design Decision: Model Architecture
    ------------------------------
    1. Model Loading:
       - Automatic device selection (GPU/CPU)
       - Pre-trained model loading with error handling
    
    2. Training Pipeline:
       - Configurable hyperparameters
       - Progress tracking
       - Validation during training
    
    Architecture:
    ```
    ┌─────────────┐
    │ Base Model  │
    ├─────────────┤
    │ - Tokenizer │
    │ - Device    │
    │ - Training  │
    └─────────┬───┘
              │
        ┌─────┴─────┐
        │           │
    ┌───▼───┐ ┌────▼───┐
    │  NER  │ │Logistic│
    └───────┘ └────────┘
    """
    def __init__(self, model_name: str, num_labels: int):
        """
        Initialize base model with specified architecture.
        
        Args:
            model_name (str): HuggingFace model identifier
            num_labels (int): Number of output labels
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def prepare_data(self, texts: List[str], labels: Optional[List] = None):
        """
        Prepare data for training or inference.
        
        Design Decision: Data Preparation
        -----------------------------
        - Tokenization with truncation and padding
        - Label encoding (if applicable)
        - Device placement for GPU acceleration
        
        Args:
            texts (List[str]): Input texts
            labels (Optional[List]): Optional labels for supervised learning
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move tensors to device
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        if labels is not None:
            encodings['labels'] = torch.tensor(labels).to(self.device)

        return encodings

    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir,
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the model with specified hyperparameters.
        
        Design Decision: Training Pipeline
        -----------------------------
        - Configurable hyperparameters
        - Progress tracking
        - Validation during training
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Output directory for model checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        train_encodings = self.prepare_data(train_texts, train_labels)
        val_encodings = self.prepare_data(val_texts, val_labels)

        train_dataset = InsiderThreatDataset(train_encodings, train_labels, self.tokenizer)
        val_dataset = InsiderThreatDataset(val_encodings, val_labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator if hasattr(self, 'data_collator') else None
        )

        trainer.train()
        self.model.save_pretrained(f"{output_dir}/best_model")
        self.tokenizer.save_pretrained(f"{output_dir}/best_model")

class NERModel(BaseModel):
    """
    NER model class inheriting from base model.
    
    Design Decision: NER Model Architecture
    -----------------------------------
    - Token classification with pre-trained model
    - Custom data collator for token classification
    """
    def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-base",
            num_labels: int = len(NER_LABELS)
        ):
        super().__init__(model_name, num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Make predictions on input texts.
        
        Design Decision: Prediction Pipeline
        -----------------------------
        - Model evaluation mode
        - Batch processing for efficiency
        - Token-level predictions
        
        Args:
            texts (List[str]): Input texts
        
        Returns:
            List[List[str]]: Predicted NER labels
        """
        self.model.eval()
        encodings = self.prepare_data(texts)

        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = outputs.logits.argmax(dim=-1)

        return [[NER_LABELS[p.item()] for p in pred_seq] for pred_seq in predictions]

class InsiderThreatLogisticModel(BaseModel):
    """
    Logistic model class inheriting from base model.
    
    Design Decision: Logistic Model Architecture
    ----------------------------------------
    - Sequence classification with pre-trained model
    - Custom classification layer with sigmoid activation
    """
    def __init__(
            self,
            model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ):
        super().__init__(model_name, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        ).to(self.device)

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1),
            torch.nn.Sigmoid()
        )

## 3. Base Model Evaluation

def create_empty_ner_labels(text):
    """Create empty NER labels for a given text"""
    tokens = text.split()
    return ['O'] * len(tokens)  # 'O' represents "Outside" in NER tagging

def evaluate_ner_base_model(test_texts, test_labels, batch_size=8):
    """    
    This handles cases where the ground truth for a tweet
    is empty. In such cases, we generate a ground truth 
    label list consisting of "O" for each token produced by the model's tokenizer.
    
    Args:
        test_texts (List[str]): List of tweet texts.
        test_labels (List[List[str]]): List of token-level ground truth labels per tweet.
        batch_size (int): Batch size for processing.
        
    Returns:
        dict: A dictionary containing predicted labels and evaluation metrics.
    """
    # Instantiate your NERModel (make sure to use your updated class)
    model = NERModel()  # Use your current NERModel class
    model.model = model.model.half() # Optional, convert model to half precision for faster inference
    model.model.eval()

    all_predictions = []
    adjusted_true_labels = []

    total_batches = (len(test_texts) + batch_size - 1) // batch_size

    # Process the test data in batches
    for i in tqdm(range(0, len(test_texts), batch_size), desc="Evaluating NER"):
        # Clear cache at the start of each batch to manage GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        batch_texts = test_texts[i:i + batch_size]
        # For each tweet in the batch, adjust the ground truth:
        # If the corresponding ground truth is empty, generate a list of "O" for each token.
        batch_true_labels = []
        for idx, text in enumerate(batch_texts):
            tokens = model.tokenizer.tokenize(text)
            gt = test_labels[i + idx]
            if not gt or len(gt) == 0:
                # If no ground truth exists, assume all tokens are "O"
                batch_true_labels.append(["O"] * len(tokens))
            else:
                # Otherwise, use the provided ground truth
                # (Make sure these lists align with your model's tokenization!)
                batch_true_labels.append(gt)
        
        # Get model predictions for the batch
        with torch.no_grad():
            try:
                batch_predictions = model.predict(batch_texts)
                all_predictions.extend(batch_predictions)
                adjusted_true_labels.extend(batch_true_labels)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If running out of memory, process tweets individually
                    for text in batch_texts:
                        pred = model.predict([text])
                        all_predictions.append(pred[0])
                    for idx, text in enumerate(batch_texts):
                        tokens = model.tokenizer.tokenize(text)
                        gt = test_labels[i + idx]
                        if not gt or len(gt) == 0:
                            adjusted_true_labels.append(["O"] * len(tokens))
                        else:
                            adjusted_true_labels.append(gt)
                else:
                    raise e

    # Compute NER metrics using your compute_ner_metrics helper.
    metrics = compute_ner_metrics(adjusted_true_labels, all_predictions)
    return {
        'predictions': all_predictions,
        'metrics': metrics
    }

# Example of using the evaluation function:
# Assume df is your dataframe loaded from train_with_ner.csv,
# and that its 'ner_labels' column contains the ground truth NER token lists
# (which might be empty if no entities were auto-detected).
test_texts = df['Tweet'].tolist()
# Here, we assume df['ner_labels'] already contains a list of token labels for each tweet.
# If the column is stored as JSON strings, you may need to parse them:
if isinstance(df['ner_labels'].iloc[0], str):
    test_labels = df['ner_labels'].apply(json.loads).tolist()
else:
    test_labels = df['ner_labels'].tolist()

# Evaluate the NER model
ner_results = evaluate_ner_base_model(test_texts, test_labels)
print("\nNER Model Results:")
print(pd.DataFrame(ner_results['metrics']).T)


def evaluate_sentiment_base_model(test_data, test_labels, batch_size=16):
    """Evaluate base sentiment model performance with batch processing"""
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        model_kwargs={"torch_dtype": torch.float16}  # Use fp16
    )

    all_predictions = []

    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size)):
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        batch_texts = test_data[i:i + batch_size]
        try:
            batch_predictions = sentiment_pipe(batch_texts)
            all_predictions.extend(batch_predictions)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs, try with even smaller batch
                torch.cuda.empty_cache()
                gc.collect()
                batch_predictions = []
                for text in batch_texts:
                    pred = sentiment_pipe([text])
                    batch_predictions.extend(pred)
                all_predictions.extend(batch_predictions)
            else:
                raise e

    pred_labels = [1 if pred['label'] == 'POSITIVE' else 0 for pred in all_predictions]

    report = classification_report(test_labels, pred_labels, output_dict=True)
    cm = confusion_matrix(test_labels, pred_labels)

    return {
        'predictions': pred_labels,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

## 4. Load and Prepare Data

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set your dataset path here
DATASET_PATH = '/content/drive/MyDrive/cs491_DataAnalyzer/Dataset/train_with_ner.csv'
MODEL_OUTPUT_DIR = '/content/drive/MyDrive/cs491_DataAnalyzer/Dataset/models'

# Load data
df = pd.read_csv(DATASET_PATH)

# Display sample data
print("Dataset shape:", df.shape)
print("\nSample data:")
display(df.head())

## 5. Run Base Model Evaluation

# Debug: Print DataFrame information
print("DataFrame columns:", df.columns.tolist())
print("\nDataFrame head:\n", df.head())

# First, prepare sentiment labels and initialize NER labels
df['sentiment'] = df['Is Insider Threat'].map({'Yes': 1, 'No': 0})

# Initialize empty NER labels with proper structure
if 'ner_labels' not in df.columns:
    df['ner_labels'] = df['Tweet'].apply(create_empty_ner_labels)

print("Sample NER labels:", df['ner_labels'].iloc[0])
print("Length of first text:", len(df['Tweet'].iloc[0].split()))
print("Length of first NER labels:", len(df['ner_labels'].iloc[0]))

# Evaluate NER model
print("\nEvaluating Base NER Model...")
ner_results = evaluate_ner_base_model(
    df['Tweet'].values.tolist(),
    df['ner_labels'].values.tolist()
)
print("\nNER Model Results:")
print(pd.DataFrame(ner_results['metrics']).T)

print("\nEvaluating Base Sentiment Model...")
sentiment_results = evaluate_sentiment_base_model(
    df['Tweet'].values.tolist(),
    df['sentiment'].values.tolist()
)
print("\nSentiment Model Results:")
print(pd.DataFrame(sentiment_results['classification_report']).T)

plot_confusion_matrix(
    sentiment_results['confusion_matrix'],
    labels=['Negative', 'Positive'],
    title='Base Sentiment Model Confusion Matrix'
)