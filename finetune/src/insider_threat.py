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

# --------------------------------------------------------------------------------
# Data Flow Explanation:
#
# 1. Data Ingestion:
#    - The dataset (e.g., CSV file) contains tweets along with their corresponding
#      token-level NER labels (e.g., ["B-ROLE", "O", "O", "O", "O", "B-FACILITY", "O"])
#      and a sentiment/malicious label (e.g., "Yes"/"No" or 1/0).
#
# 2. Preprocessing:
#    - Each tweet is tokenized using a pretrained tokenizer.
#    - The tokenization aligns with the provided NER labels.
#
# 3. Model Training:
#    - For NER: A token classification model (e.g., based on DeBERTa) is fine-tuned
#      using the tokens and their entity labels.
#    - For Sentiment: A sequence classification model is fine-tuned on the tweet-level
#      sentiment labels.
#
# 4. Evaluation:
#    - The evaluate_ner_base_model() function computes precision, recall, and F1
#      scores for each entity type.
#    - Similarly, evaluate_sentiment_base_model() provides a classification report
#      and confusion matrix for the sentiment model.
#
# --------------------------------------------------------------------------------

# GPU Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set overall seed for reproducibility
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Constants for NER labels – update these according to your synthetic dataset annotations.
NER_LABELS = ['O', 'B-ROLE', 'I-ROLE', 'B-FACILITY', 'I-FACILITY', 'B-ACCESS_CODE', 'I-ACCESS_CODE']

def compute_ner_metrics(true_labels, predicted_labels):
    """
    Compute NER-specific metrics.

    Args:
        true_labels (List[List[str]]): Ground truth NER labels for each token in each tweet.
        predicted_labels (List[List[str]]): Predicted NER labels from the model.

    Returns:
        dict: A dictionary containing precision, recall, and F1-score for each entity type.
    """
    # Flatten label lists for evaluation
    true_flat = [label for seq in true_labels for label in seq]
    pred_flat = [label for seq in predicted_labels for label in seq]

    # Align lengths if necessary
    min_len = min(len(true_flat), len(pred_flat))
    true_flat = true_flat[:min_len]
    pred_flat = pred_flat[:min_len]

    report = classification_report(true_flat, pred_flat, output_dict=True, zero_division=0)
    return report

class InsiderThreatDataset:
    """
    Custom dataset for insider threat detection.
    
    Attributes:
        texts (List[str]): List of tweet texts.
        labels (Optional[List]): Corresponding labels (NER or sentiment).
        tokenizer: The tokenizer used for encoding texts.
    """
    def __init__(self, texts: List[str], labels: Optional[List] = None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx] if self.labels is not None else None

def compute_metrics(pred):
    """
    Compute classification metrics for sequence-level tasks (e.g., sentiment).
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) if len(pred.predictions.shape) > 1 else (pred.predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

class BaseModel:
    """
    Base model class to handle common tasks: tokenization, data preparation, and training.
    """
    def __init__(self, model_name: str, num_labels: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def prepare_data(self, texts: List[str], labels: Optional[List] = None):
        """
        Tokenize the texts and convert to tensors. If labels are provided, include them.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        # Move tensors to the selected device
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        if labels is not None:
            # For token-level labels, ensure correct shape (e.g., list of lists of label ids)
            encodings['labels'] = torch.tensor(labels).to(self.device)
        return encodings

    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir,
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Fine-tune the model using the Trainer API.
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
    Model for corporate NER (token classification). Fine-tunes a base transformer for entity detection.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", num_labels: int = len(NER_LABELS)):
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
        Run prediction on a list of texts. Returns a list of lists, where each inner list
        contains the predicted entity labels for each token in the corresponding text.
        """
        self.model.eval()
        encodings = self.prepare_data(texts)
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = outputs.logits.argmax(dim=-1)
        # Convert predictions (indices) back to label strings using NER_LABELS
        return [[NER_LABELS[p.item()] for p in pred_seq] for pred_seq in predictions]

class InsiderThreatLogisticModel(BaseModel):
    """
    Model for sentiment analysis (sequence classification). Fine-tunes a transformer for
    predicting the overall malicious (or sentiment) label of a tweet.
    """
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        super().__init__(model_name, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        ).to(self.device)
        # Replace the classifier with a simple feed-forward layer and Sigmoid for binary classification.
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1),
            torch.nn.Sigmoid()
        )

# --------------------------------------------------------------------------------
# Sample Data Instance
# --------------------------------------------------------------------------------
# Imagine your synthetic dataset (CSV) contains the following row:
#
# Tweet: "CEO announces merger plans at headquarters."
#
# After tokenization (simple split for illustration), tokens might be:
# ["CEO", "announces", "merger", "plans", "at", "headquarters", "."]
#
# And the corresponding NER labels (real entity annotations) should be:
# ["B-ROLE", "O", "O", "O", "O", "B-FACILITY", "O"]
#
# In your CSV file, the 'ner_labels' column should contain these lists (possibly as JSON strings).
#
# Similarly, your sentiment label (e.g., "Is Insider Threat") could be mapped to 1 (malicious) or 0 (non-malicious).

# For demonstration, here's a manual example:
sample_tweet = "CEO announces merger plans at headquarters."
sample_tokens = sample_tweet.split()  # Simple tokenization; actual tokenization will depend on your tokenizer.
sample_ner_labels = ["B-ROLE", "O", "O", "O", "O", "B-FACILITY", "O"]

print("Sample Tokens:", sample_tokens)
print("Sample NER Labels:", sample_ner_labels)

# --------------------------------------------------------------------------------
# Evaluation Functions (same as before)
# --------------------------------------------------------------------------------
def evaluate_ner_base_model(test_data, test_labels, batch_size=8):
    """
    Evaluate the NER model on the test set, processing data in batches.
    """
    # Use the fine-tuned NERModel (replace CorporateNERModel with NERModel if that's your class name)
    model = NERModel()
    # Convert model to half precision for faster inference if using GPU
    model.model = model.model.half()
    model.model.eval()

    all_predictions = []
    for i in tqdm(range(0, len(test_data), batch_size)):
        torch.cuda.empty_cache()
        gc.collect()
        batch_texts = test_data[i:i + batch_size]
        try:
            batch_predictions = model.predict(batch_texts)
            all_predictions.extend(batch_predictions)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                batch_predictions = []
                for text in batch_texts:
                    pred = model.predict([text])
                    batch_predictions.extend(pred)
                all_predictions.extend(batch_predictions)
            else:
                raise e

    return {
        'predictions': all_predictions,
        'metrics': compute_ner_metrics(test_labels, all_predictions)
    }

def evaluate_sentiment_base_model(test_data, test_labels, batch_size=16):
    """
    Evaluate the sentiment model on the test set, processing data in batches.
    """
    device_num = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device_num,
        model_kwargs={"torch_dtype": torch.float16}
    )
    all_predictions = []
    for i in tqdm(range(0, len(test_data), batch_size)):
        torch.cuda.empty_cache()
        gc.collect()
        batch_texts = test_data[i:i + batch_size]
        try:
            batch_predictions = sentiment_pipe(batch_texts)
            all_predictions.extend(batch_predictions)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                batch_predictions = []
                for text in batch_texts:
                    pred = sentiment_pipe([text])
                    batch_predictions.extend(pred)
                all_predictions.extend(batch_predictions)
            else:
                raise e
    # Map sentiment labels to binary (example: 'POSITIVE' -> 1, others -> 0)
    pred_labels = [1 if pred['label'] == 'POSITIVE' else 0 for pred in all_predictions]
    report = classification_report(test_labels, pred_labels, output_dict=True)
    cm = confusion_matrix(test_labels, pred_labels)
    return {
        'predictions': pred_labels,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Plot the confusion matrix using seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --------------------------------------------------------------------------------
# Data Loading and Preparation (example using CSV from Google Drive)
# --------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = '/content/drive/MyDrive/cs491_DataAnalyzer/Dataset/synthetic_insider_threat.csv'
MODEL_OUTPUT_DIR = '/content/drive/MyDrive/cs491_DataAnalyzer/Dataset/models'

# Load dataset with pandas. Assume 'Tweet', 'ner_labels', and 'Is Insider Threat' columns exist.
df = pd.read_csv(DATASET_PATH)
print("Dataset shape:", df.shape)
print("\nSample data:")
display(df.head())

# Convert sentiment column: map "Yes" to 1 (malicious) and "No" to 0 (non-malicious)
df['sentiment'] = df['Is Insider Threat'].map({'Yes': 1, 'No': 0})

# ---------------------------------------------------------------------------
# IMPORTANT:
# Replace the dummy "O" labels with real NER labels from your synthetic dataset.
# Example:
# Instead of generating empty labels, your CSV should contain real labels.
# For instance, one row could be:
# Tweet: "CEO announces merger plans at headquarters."
# ner_labels: '["B-ROLE", "O", "O", "O", "O", "B-FACILITY", "O"]'
#
# Here, we assume that df['ner_labels'] is already populated with such lists.
# If it's stored as a JSON string, you might need to convert it using:
# import json
# df['ner_labels'] = df['ner_labels'].apply(json.loads)
# ---------------------------------------------------------------------------
if 'ner_labels' in df.columns:
    # Uncomment the following line if your 'ner_labels' column is stored as JSON strings:
    # df['ner_labels'] = df['ner_labels'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    print("Using real NER labels from the dataset.")
else:
    # This block is for testing purposes only; replace it with actual labels.
    df['ner_labels'] = df['Tweet'].apply(lambda text: ["O"] * len(text.split()))

# Display sample labels for the first tweet
print("Sample NER labels for first tweet:", df['ner_labels'].iloc[0])
print("Length of first tweet tokens:", len(df['Tweet'].iloc[0].split()))
print("Length of first NER labels:", len(df['ner_labels'].iloc[0]))

# --------------------------------------------------------------------------------
# Evaluate Models
# --------------------------------------------------------------------------------

print("\nEvaluating Fine-Tuned NER Model...")
ner_results = evaluate_ner_base_model(
    df['Tweet'].values.tolist(),
    df['ner_labels'].values.tolist()
)
print("\nNER Model Results (Metrics):")
print(pd.DataFrame(ner_results['metrics']).T)

print("\nEvaluating Fine-Tuned Sentiment Model...")
sentiment_results = evaluate_sentiment_base_model(
    df['Tweet'].values.tolist(),
    df['sentiment'].values.tolist()
)
print("\nSentiment Model Results (Classification Report):")
print(pd.DataFrame(sentiment_results['classification_report']).T)

plot_confusion_matrix(
    sentiment_results['confusion_matrix'],
    labels=['Negative', 'Positive'],
    title='Sentiment Model Confusion Matrix'
)
