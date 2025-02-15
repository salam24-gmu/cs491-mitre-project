"""
Models for Insider Threat Detection

This module implements a multi-task learning approach combining:
1. Named Entity Recognition (NER) for identifying sensitive entities
2. Logistic Regression for insider threat detection
"""

from transformers import (
    AutoModelForTokenClassification, #  For NER tasks
    AutoTokenizer, # For tokenization
    AutoModelForSequenceClassification, # For sentiment analysis tasks
    TrainingArguments,
    Trainer, # 
    DataCollatorForTokenClassification # for Batched NER data
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Entity labels for NER
NER_LABELS = [
    "O",  # Outside of named entity
    "B-ROLE", "I-ROLE",  # Role entities (B: Beginning, I: Inside)
    "B-FACILITY", "I-FACILITY",  # Facility entities
    "B-ACCESS_CODE", "I-ACCESS_CODE",  # Access code entities
    "B-SENSITIVE_DATA", "I-SENSITIVE_DATA",  # Sensitive data entities
]

class InsiderThreatDataset(Dataset):
    """
    Custom dataset for insider threat detection tasks.

    This dataset handles both NER and sentiment analysis data formats.

    Args:
        texts (List[str]): List of input text sequences
        labels (Optional[List[str]]): List of labels for each text sequence
        tokenizer: Tokenizer instance for text preprocessing

    Returns:
        Dataset instance that can be used with PyTorch DataLoader
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
    Compute evaluation metrics for model predictions.

    For NER: Uses token-level metrics
    For Sentiment: Uses binary classification metrics

    Args:
        pred: Prediction object with label_ids and predictions

    Returns:
        Dict with accuracy, f1, precision, and recall
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) if len(pred.predictions.shape) > 1 else (pred.predictions > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class BaseModel:
    """
    Base class for insider threat detection models.
    Provides common functionality for both NER and Sentiment models.

    Args:
        model_name (str): Pre-trained model name/path
        num_labels (int): Number of output labels
    """
    def __init__(self, model_name: str, num_labels: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def prepare_data(self, texts: List[str], labels: Optional[List] = None):
        """
        Prepare input data for model training or inference.

        Args:
            texts (List[str]): List of input text sequences
            labels (Optional[List]): List of labels (if for training)

        Returns:
            dict: Encoded inputs containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Encoded labels (if provided)
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        if labels is not None:
            encodings['labels'] = torch.tensor(labels)
        
        return encodings

    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir, 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the model using the provided data.

        Args:
            train_texts (List[str]): Training text sequences
            train_labels (List): Training labels
            val_texts (List[str]): Validation text sequences
            val_labels (List): Validation labels
            output_dir (str): Directory to save model checkpoints
            num_epochs (int, optional): Number of training epochs. Defaults to 3
            batch_size (int, optional): Batch size for training. Defaults to 16
            learning_rate (float, optional): Learning rate. Defaults to 2e-5

        Returns:
            Trainer: Trained model trainer instance
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

class CorporateNERModel(BaseModel):
    """
    Named Entity Recognition model for identifying corporate entities.

    This model specializes in detecting entities such as:
    - Roles (e.g., "CEO", "System Admin")
    - Facilities (e.g., "Server Room", "R&D Lab")
    - Access Codes
    - Sensitive Data references

    Args:
        model_name (str, optional): Pre-trained model name. Defaults to "microsoft/deberta-v3-base"
        num_labels (int, optional): Number of NER labels. Defaults to len(NER_LABELS)
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
        Predict NER tags for input texts.

        Args:
            texts: List of input texts

        Returns:
            List[List[str]]: Predicted NER tags for each token in each sequence
        """
        self.model.eval()
        encodings = self.prepare_data(texts)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = outputs.logits.argmax(dim=-1)
        
        return [[NER_LABELS[p.item()] for p in pred_seq] for pred_seq in predictions]

class InsiderThreatLogisticModel(BaseModel):
    """
    Logistic regression model for insider threat detection.
    Uses sigmoid activation to output threat probabilities.

    Args:
        model_name (str): Pre-trained model name
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
        
        # Replace classification head with binary logistic
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1),
            torch.nn.Sigmoid()
        ).to(self.device)

    def compute_loss(self, logits, labels):
        """Compute binary cross entropy loss"""
        return F.binary_cross_entropy(logits.squeeze(), labels.float())

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict threat probabilities for input texts.

        Args:
            texts: List of input texts

        Returns:
            Array of threat probabilities (0 to 1)
        """
        self.model.eval()
        encodings = self.prepare_data(texts)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probabilities = outputs.logits.squeeze().cpu().numpy()
        
        return probabilities

    def predict_with_threshold(self, texts: List[str], threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict binary threats with a custom threshold.

        Args:
            texts: List of input texts
            threshold: Classification threshold (default: 0.5)

        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        probabilities = self.predict(texts)
        return probabilities, (probabilities >= threshold).astype(int)
