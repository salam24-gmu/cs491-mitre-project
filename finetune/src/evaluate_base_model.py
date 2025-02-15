"""
This module provides evaluation utilities for assessing the performance of base (non-finetuned) 
models on insider threat detection tasks. It supports evaluation of both Named Entity Recognition (NER) 
and sentiment analysis models as complementary approaches to threat detection.

Key components:
- NER evaluation: Identifies and evaluates detection of key entities (e.g., sensitive data, actions)
- Sentiment analysis: Analyzes emotional tone which may indicate suspicious behavior
- Metrics computation: Provides detailed performance metrics and visualizations
"""

import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models import CorporateNERModel, InsiderThreatLogisticModel
import numpy as np

def evaluate_ner_base_model(test_data, test_labels):
    """
    Evaluates the performance of the base NER model on insider threat detection.
    
    Design choices:
    - Uses CorporateNERModel which is pre-configured for corporate entity detection
    - Processes data in batch to optimize performance
    
    Args:
        test_data (List[str]): List of text samples to evaluate
        test_labels (List[List[str]]): Ground truth NER labels for each token in each text sample
        
    Returns:
        dict: Contains:
            - predictions: Model's NER predictions for each token
            - metrics: Detailed evaluation metrics including precision, recall, and F1 score
    """
    model = CorporateNERModel()
    predictions = model.predict(test_data)
    
    results = {
        'predictions': predictions,
        'metrics': compute_ner_metrics(test_labels, predictions)
    }
    
    return results

def evaluate_sentiment_base_model(test_data, test_labels):
    """
    Evaluates sentiment analysis as a complementary signal for insider threat detection.
    
    Design choices:
    - Uses RoBERTa model fine-tuned on Twitter data for robust sentiment detection
    - Leverages GPU acceleration when available
    - Binary classification: maps sentiment to binary indicators (positive/negative)
    
    Args:
        test_data (List[str]): Text samples to analyze
        test_labels (List[int]): Binary ground truth labels (0: negative, 1: positive)
        
    Returns:
        dict: Contains:
            - predictions: Binary sentiment predictions
            - classification_report: Detailed performance metrics
            - confusion_matrix: Error analysis visualization data
    """
    # Initialize sentiment pipeline
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Get predictions
    predictions = sentiment_pipe(test_data)
    pred_labels = [1 if pred['label'] == 'POSITIVE' else 0 for pred in predictions]
    
    # Calculate metrics
    report = classification_report(test_labels, pred_labels, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    
    return {
        'predictions': pred_labels,
        'classification_report': report,
        'confusion_matrix': cm
    }

def compute_ner_metrics(true_labels, predicted_labels):
    """
    Computes detailed metrics for NER performance evaluation.
    
    Implementation details:
    - Flattens nested label structure for token-level evaluation
    - Generates comprehensive classification report including per-class metrics
    
    Args:
        true_labels (List[List[str]]): Ground truth labels for each token in each sequence
        predicted_labels (List[List[str]]): Predicted labels for each token in each sequence
        
    Returns:
        dict: Classification report containing precision, recall, F1 score per class
    """
    # Convert labels to flat lists
    true_flat = [label for seq in true_labels for label in seq]
    pred_flat = [label for seq in predicted_labels for label in seq]
    
    # Calculate metrics
    report = classification_report(true_flat, pred_flat, output_dict=True)
    
    return report

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Visualizes model prediction errors through a confusion matrix heatmap.
    
    Design choices:
    - Uses seaborn for enhanced visualization aesthetics
    - Color intensity corresponds to prediction frequency
    - Includes actual counts for detailed error analysis
    
    Args:
        cm (np.array): Confusion matrix data
        labels (List[str]): Class labels for axes
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    """
    Orchestrates the evaluation workflow for both NER and sentiment analysis models.
    
    Workflow:
    1. Loads test data from CSV
    2. Evaluates NER model performance
    3. Evaluates sentiment analysis performance
    4. Generates visualization for error analysis
    
    Note: Ensure test_data.csv contains 'text', 'ner_labels', and 'sentiment' columns
    """
    # Load your test data
    # Modify this path to point to your test dataset
    test_data = pd.read_csv('../data/test_data.csv')
    
    # Evaluate NER model
    print("\nEvaluating Base NER Model...")
    ner_results = evaluate_ner_base_model(
        test_data['text'].tolist(),
        test_data['ner_labels'].tolist()
    )
    print("\nNER Model Results:")
    print(pd.DataFrame(ner_results['metrics']).T)
    
    # Evaluate Sentiment model
    print("\nEvaluating Base Sentiment Model...")
    sentiment_results = evaluate_sentiment_base_model(
        test_data['text'].tolist(),
        test_data['sentiment'].tolist()
    )
    print("\nSentiment Model Results:")
    print(pd.DataFrame(sentiment_results['classification_report']).T)
    
    # Plot confusion matrix for sentiment
    plot_confusion_matrix(
        sentiment_results['confusion_matrix'],
        labels=['Negative', 'Positive'],
        title='Base Sentiment Model Confusion Matrix'
    )

if __name__ == "__main__":
    main()
