import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict

def prepare_dataset(data_path, random_state=42):
    """
    Prepare the dataset by splitting it into train, validation, and test sets.
    Maintains stratification across the insider threat label.
    
    Args:
        data_path (str): Path to the CSV file
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing train, validation, and test DataFrames
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
    
    print("\nInsider Threat Distribution:")
    for name, dataset in [('Training', train), ('Validation', val), ('Test', test)]:
        threat_dist = dataset['Is Insider Threat'].value_counts(normalize=True)
        print(f"\n{name} set:")
        print(f"Malicious: {threat_dist[True]*100:.1f}%")
        print(f"Non-malicious: {threat_dist[False]*100:.1f}%")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

def visualize_distribution(df: pd.DataFrame, before_col: str, after_col: str, title: str = "Label Distribution"):
    """
    Visualize the distribution of labels before and after conversion.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        before_col (str): Column name for original labels
        after_col (str): Column name for converted labels
        title (str): Title for the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original distribution
    sns.countplot(data=df, x=before_col, ax=ax1)
    ax1.set_title("Original Distribution")
    ax1.set_ylabel("Count")
    
    # Plot converted distribution
    sns.countplot(data=df, x=after_col, ax=ax2)
    ax2.set_title("Converted Distribution")
    ax2.set_ylabel("Count")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def display_sample_conversions(df: pd.DataFrame, n_samples: int = 5):
    """
    Display sample rows showing the conversion from insider threat to sentiment.
    
    Args:
        df (pd.DataFrame): DataFrame with both original and converted labels
        n_samples (int): Number of samples to display
    """
    samples = df[['Tweet', 'Is Insider Threat', 'sentiment']].sample(n=n_samples)
    print("\nSample Conversions:")
    for _, row in samples.iterrows():
        print(f"\nTweet: {row['Tweet']}")
        print(f"Original: {row['Is Insider Threat']}")
        print(f"Converted: {row['sentiment']} ({'Insider Threat' if row['sentiment'] == 1 else 'Not a Threat'})")

def prepare_sentiment_labels(df: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
    """
    Convert insider threat labels to binary sentiment scores.
    
    Converts 'Is Insider Threat' column to sentiment scores where:
    1 = Insider Threat (Yes)
    0 = Not a Threat (No)
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Is Insider Threat' column
        visualize (bool): Whether to show distribution visualizations
        
    Returns:
        pd.DataFrame: DataFrame with new 'sentiment' column
    """
    df = df.copy()
    
    # Store original values for visualization
    df['original_label'] = df['Is Insider Threat']
    
    # Convert to binary sentiment where 1 is positive (insider threat)
    df['sentiment'] = (df['Is Insider Threat'] == True).astype(int)
    
    # Print distribution of sentiment scores
    sentiment_dist = df['sentiment'].value_counts(normalize=True)
    print("\nSentiment Distribution:")
    print(f"Positive (Insider Threat): {sentiment_dist[1]*100:.1f}%")
    print(f"Negative (Non-Threat): {sentiment_dist[0]*100:.1f}%")
    
    if visualize:
        # Visualize the distribution
        fig = visualize_distribution(df, 'original_label', 'sentiment', 
                                   "Insider Threat to Sentiment Conversion")
        plt.show()
        
        # Display sample conversions
        display_sample_conversions(df)
    
    # Drop the temporary column
    df = df.drop('original_label', axis=1)
    
    return df

def load_and_prepare_data(data_dir: str, visualize: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare data splits with sentiment scores.
    
    Args:
        data_dir (str): Directory containing train.csv, val.csv, and test.csv
        visualize (bool): Whether to show distribution visualizations
        
    Returns:
        dict: Dictionary containing prepared DataFrames for train, val, and test sets
    """
    # Load split datasets
    datasets = {
        'train': pd.read_csv(os.path.join(data_dir, 'train.csv')),
        'val': pd.read_csv(os.path.join(data_dir, 'val.csv')),
        'test': pd.read_csv(os.path.join(data_dir, 'test.csv'))
    }
    
    # Convert insider threat labels to sentiment scores for each split
    for split in datasets:
        print(f"\nPreparing {split} set:")
        datasets[split] = prepare_sentiment_labels(datasets[split], visualize=visualize)
    
    return datasets

if __name__ == "__main__":
    # Path to your dataset
    data_path = r"c:\Users\Triet\OneDrive\GMU\Spring 25\cs491\data\synthetic_insider_threat.csv"
    
    # Prepare the dataset
    splits = prepare_dataset(data_path)
    
    # Convert to sentiment scores (optional)
    splits_with_sentiment = {
        split: prepare_sentiment_labels(df) 
        for split, df in splits.items()
    }
