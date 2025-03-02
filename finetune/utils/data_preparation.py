"""
Data Splitting, Label Conversion, Data Visualization, and Dataset Loading

This module handles the preparation and preprocessing of data for the insider threat detection.
It provides functionality for:
1. Data splitting (train/validation/test)
2. Label conversion (insider threat to sentiment)
3. Data visualization
4. Dataset loading and preparation

The module is designed to work with the following data format:
- Input: CSV file with columns:
    * Category: Type of insider threat
    * Username: User identifier
    * Timestamp: Time of the event
    * Tweet: The actual text content
    * Is Insider Threat: Yes/No indicator

- Output: Three CSV files (train.csv, val.csv, test.csv) with additional columns:
    * sentiment: Binary values (0: Not a threat, 1: Threat)
    * ner_labels: NER tags for each token (added during model evaluation)

This module integrates with:
- models.py: Provides prepared data for model training and evaluation
- evaluate_base_model.py: Supplies test data for model performance assessment
- insider_threat.py: Feeds processed data into the main detection pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict

def prepare_dataset(data_path: str, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Prepare and split the dataset for model training and evaluation.
    
    This function:
    1. Loads the raw CSV data
    2. Converts string labels to boolean
    3. Performs stratified splitting:
        - 70% training
        - 15% validation
        - 15% test
    4. Saves splits to separate CSV files
    
    The splits maintain the same distribution of insider threat labels to ensure
    representative samples in each set.
    
    Args:
        data_path (str): Path to the raw CSV file
        random_state (int): Seed for reproducible splitting
    
    Returns:
        dict: Contains three DataFrames:
            - 'train': Training data (70%)
            - 'val': Validation data (15%)
            - 'test': Test data (15%)
            
    Integration:
        - Used by models.BaseModel for training data preparation
        - Feeds into evaluate_base_model.py for performance assessment
        - Supports insider_threat.py's main detection pipeline
    """
    df = pd.read_csv(data_path)
    
    # Convert boolean strings to actual boolean values
    df['Is Insider Threat'] = df['Is Insider Threat'].map({'Yes': True, 'No': False})
    
    # First split: separate test set (15%)
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df['Is Insider Threat'],
        random_state=random_state
    )
    
    # Second split: separate validation set from training set (15% of original = 17.6% of remaining)
    train, val = train_test_split(
        train_val,
        test_size=0.176,  # 0.176 of 85% ≈ 15% of total
        stratify=train_val['Is Insider Threat'],
        random_state=random_state
    )

    # Save splits to CSV files
    output_dir = os.path.dirname(data_path)    
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("\nDataset Split Stats:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

def visualize_distribution(df: pd.DataFrame, before_col: str, after_col: str, 
                         title: str = "Label Distribution") -> None:
    """
    Visualize label distribution before and after conversion.
    
    Creates a side-by-side comparison of label distributions to verify that
    the conversion process maintains the proper balance of classes.
    
    This visualization is crucial for:
    1. Validating data preprocessing
    2. Identifying potential class imbalances
    3. Ensuring proper stratification
    
    Args:
        df (pd.DataFrame): DataFrame with both original and converted labels
        before_col (str): Column name for original labels
        after_col (str): Column name for converted labels
        title (str): Title for the plot
        
    Integration:
        - Used during prepare_sentiment_labels() to verify conversion
        - Supports data quality assessment in the pipeline
        - Helps in identifying potential bias in the dataset
    """
    plt.figure(figsize=(12, 5))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    df[before_col].value_counts().plot(kind='bar')
    plt.title(f"Original {before_col}")
    plt.ylabel('Count')
    
    # Converted distribution
    plt.subplot(1, 2, 2)
    df[after_col].value_counts().plot(kind='bar')
    plt.title(f"Converted {after_col}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def display_sample_conversions(df: pd.DataFrame, n_samples: int = 5) -> None:
    """
    Display sample rows showing label conversion results.
    
    Provides a human-readable view of how labels are transformed,
    helping to verify the conversion logic and identify potential issues.
    
    Args:
        df (pd.DataFrame): DataFrame with both original and converted labels
        n_samples (int): Number of sample rows to display
        
    Integration:
        - Used during data preparation to validate conversions
        - Supports quality assurance in the pipeline
        - Helps in debugging conversion issues
    """
    print("\nSample Label Conversions:")
    print(df[['Is Insider Threat', 'sentiment']].head(n_samples))

def prepare_sentiment_labels(df: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
    """
    Convert insider threat labels to binary sentiment scores.
    
    This function is a crucial part of the preprocessing pipeline:
    1. Converts categorical insider threat labels to numerical sentiment scores
    2. Maintains the semantic meaning of the labels
    3. Prepares data for sentiment analysis model
    
    Conversion mapping:
    - 'Yes' (Insider Threat) -> 1
    - 'No' (Not a Threat) -> 0
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Is Insider Threat' column
        visualize (bool): Whether to show distribution visualizations
        
    Returns:
        pd.DataFrame: DataFrame with new 'sentiment' column
        
    Integration:
        - Feeds prepared data to InsiderThreatLogisticModel
        - Supports sentiment analysis in the detection pipeline
        - Enables correlation analysis with NER features
    """
    # Convert insider threat labels to sentiment scores
    df['sentiment'] = df['Is Insider Threat'].map({'Yes': 1, 'No': 0})
    
    if visualize:
        visualize_distribution(df, 'Is Insider Threat', 'sentiment')
        display_sample_conversions(df)
    
    return df

def load_and_prepare_data(data_dir: str, visualize: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare all data splits with sentiment scores.
    
    This is the main entry point for data preparation:
    1. Loads train/val/test splits
    2. Converts labels to sentiment scores
    3. Prepares data for model consumption
    
    Args:
        data_dir (str): Directory containing split CSV files
        visualize (bool): Whether to show distribution visualizations
        
    Returns:
        dict: Contains prepared DataFrames:
            - 'train': Training data with sentiment scores
            - 'val': Validation data with sentiment scores
            - 'test': Test data with sentiment scores
            
    Integration:
        - Main data preparation entry point for the entire pipeline
        - Feeds prepared data to all model components
        - Supports both training and evaluation workflows
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f'{split}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            splits[split] = prepare_sentiment_labels(df, visualize=visualize)
        else:
            print(f"Warning: {split}.csv not found in {data_dir}")
    
    return splits

if __name__ == "__main__":
    # Example usage and testing
    data_path = r"c:\Users\Triet\OneDrive\GMU\Spring 25\cs491\data\synthetic_insider_threat.csv"
    
    # Prepare and split dataset
    splits = prepare_dataset(data_path)
    
    # Load and prepare data with sentiment scores
    prepared_splits = load_and_prepare_data(os.path.dirname(data_path))
