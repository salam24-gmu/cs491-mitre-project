import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

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

if __name__ == "__main__":
    # Path to your dataset
    data_path = r"c:\Users\Triet\OneDrive\GMU\Spring 25\cs491\data\synthetic_insider_threat.csv"
    
    # Prepare the dataset
    splits = prepare_dataset(data_path)
