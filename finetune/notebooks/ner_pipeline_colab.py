"""
Named Entity Recognition (NER) Pipeline
================================================================

Overview
--------
This pipeline implements Named Entity Recognition on tweets to identify potential insider threats.
It uses the BERT base NER model (dslim/bert-base-NER) to extract entities that might indicate
sensitive information disclosure or security risks.

Architecture & Design Decisions
-----------------------------
1. Model Selection:
   - Using dslim/bert-base-NER over Twitter-specific models because:
     * Better general-purpose entity recognition
     * More stable and well-tested
     * Identifies 4 critical entity types: PER (Person), ORG (Organization), 
       LOC (Location), MISC (Miscellaneous)

2. Pipeline Components:
   ```
   [Input Tweet Data] → [GPU-Accelerated NER] → [Entity Extraction] → [JSON Serialization] → [CSV Output]
                                ↓
                     [Entity Type Analysis]
   ```

3. Error Handling Strategy:
   - Two-level error handling:
     * Batch-level for system errors
     * Individual tweet-level for content errors
   - Graceful degradation with empty lists for failed items

4. Performance Considerations:
   - GPU acceleration when available
   - Batch processing for efficiency
   - Optimized JSON serialization for large datasets, 
        - JSON serialization is the process of converting a Python object (like a list of entities) into a JSON string.

Dependencies
-----------
- transformers: Hugging Face Transformers library for NER model
- torch: PyTorch for GPU acceleration 
- pandas: Data manipulation and CSV handling
- tqdm: Progress tracking
- numpy: Numerical operations and type handling


Output Format
------------
The pipeline generates a CSV file with the following additional columns:
- ner_entities: All identified entities
- person_entities: Person names and identifiers
- org_entities: Organization names
- loc_entities: Location information
- misc_entities: Other potentially sensitive information


Due to the lack of Dataset that is relevant to the tweets context. Using spaCy to create synthetic datasets for NER is viable. 
(https://medium.com/@lukas.kriesch/creating-synthetic-datasets-for-named-entity-recognition-with-spacy-2f7d28d41dc9)

    - spaCy provides a pipeline that can be used to create synthetic datasets for NER tasks by 

"""

import os
import torch
from transformers import pipeline
import pandas as pd
import json
from tqdm import tqdm
import time
import requests
import numpy as np

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

def setup_paths():
    base_path = "/content/drive/MyDrive/cs491_DataAnalyzer/Dataset"

    paths = {
        'input_file': os.path.join(base_path, "synthetic_insider_threat.csv"),
        'output_file': os.path.join(base_path, "train_with_ner.csv")
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(paths['output_file']), exist_ok=True)

    return paths

def load_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} tweets")
    return df

def apply_ner(texts, batch_size=32, retries=3):
    """
    Apply Named Entity Recognition to a list of texts.
    
    Args:
        texts (list): List of texts to apply NER to
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        retries (int, optional): Number of retries for failed model loads. Defaults to 3.
    
    Returns:
        list: List of extracted entities
    """
    print("Initializing NER model...")
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

    for attempt in range(retries):
        try:
            ner = pipeline("ner",
                          model="dslim/bert-base-NER",
                          aggregation_strategy="simple",
                          device=device)
            break  # Exit the loop if successful
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    else:
        raise RuntimeError(f"Failed to load the model after {retries} attempts.")

    all_entities = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"Processing {len(texts)} tweets in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            # Process each text individually to handle errors better
            batch_results = []
            for text in batch:
                try:
                    # Handle potential None or empty strings
                    if not text or pd.isna(text):
                        batch_results.append([])
                        continue

                    result = ner(str(text))
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing text: {str(e)[:100]}...")
                    batch_results.append([])

            all_entities.extend(batch_results)

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)[:100]}...")
            all_entities.extend([[] for _ in batch])

    return all_entities

def convert_to_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Design Decision:
    - Using try-except for robustness against different numeric types
    - Converting all numpy types to native Python types for JSON compatibility
    - Maintaining original non-numeric data types
    
    Args:
        obj: Any object to be converted for JSON serialization
        
    Returns:
        JSON serializable version of the object
    """
    try:
        return float(obj) if isinstance(obj, (np.floating, float)) else obj
    except:
        return obj

def save_results(df, entities, output_path):
    """
    Save the original data with NER labels.
    
    Args:
        df (pd.DataFrame): Original dataset
        entities (list): List of extracted entities
        output_path (str): Path to the output CSV file
    """
    # Convert entities to JSON-serializable format
    json_entities = []
    for ents in entities:
        json_ents = []
        for ent in ents:
            # Convert each field to JSON-serializable format
            json_ent = {
                'entity_group': ent['entity_group'],
                'score': convert_to_json_serializable(ent['score']),
                'word': ent['word'],
                'start': convert_to_json_serializable(ent['start']),
                'end': convert_to_json_serializable(ent['end'])
            }
            json_ents.append(json_ent)
        json_entities.append(json_ents)

    # Add NER results as a new column
    df['ner_entities'] = [json.dumps(ent) for ent in json_entities]

    # Add individual columns for each entity type
    df['person_entities'] = [
        json.dumps([e for e in ents if e['entity_group'] == 'PER'])
        for ents in json_entities
    ]
    df['org_entities'] = [
        json.dumps([e for e in ents if e['entity_group'] == 'ORG'])
        for ents in json_entities
    ]
    df['loc_entities'] = [
        json.dumps([e for e in ents if e['entity_group'] == 'LOC'])
        for ents in json_entities
    ]
    df['misc_entities'] = [
        json.dumps([e for e in ents if e['entity_group'] == 'MISC'])
        for ents in json_entities
    ]

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def analyze_entity_distribution(entities):
    """
    Analyze the distribution of entity types.
    
    Args:
        entities (list): List of extracted entities
    """
    entity_counts = {
        'PER': 0,
        'ORG': 0,
        'LOC': 0,
        'MISC': 0
    }

    for tweet_entities in entities:
        for entity in tweet_entities:
            entity_type = entity.get('entity_group')
            if entity_type in entity_counts:
                entity_counts[entity_type] += 1

    print("\nEntity type distribution:")
    for entity_type, count in entity_counts.items():
        print(f"{entity_type}: {count}")

# Run the pipeline
paths = setup_paths()

# Load data
df = load_dataset(paths['input_file'])

# Get tweets from the Tweet column
texts = df['Tweet'].tolist()

# Apply NER
print("Applying NER to tweets...")
entities = apply_ner(texts)

# Analyze entity distribution
analyze_entity_distribution(entities)

# Save results
save_results(df, entities, paths['output_file'])

# Print sample results
print("\nSample results:")
for text, ents in zip(texts[:10], entities[:10]):
    print(f"\nTweet: {text}")
    print(f"Entities: {ents}")

    # Print entity types found
    entity_types = set(e['entity_group'] for e in ents)
    if entity_types:
        print(f"Entity types found: {', '.join(entity_types)}")

# PER: 14
# The model identified 14 person entities overall. 
# This low number suggests that tweets in the dataset rarely mention specific 
# individuals, which makes sense if most insider threat tweets focus more on 
# processes, errors, or generic descriptions rather than naming people.

# ORG: 149
# With 149 organization entities, the model is picking up many instances where 
# company names or other business-related terms appear. Insider threat tweets 
# often mention organizations—perhaps when referring to departments, clients, 
# or companies—and this category dominates the recognized entities.

# LOC: 12
# Only 12 location entities were detected, this might be because insider threat
# examples don’t often specify locations, or because the model
# doesn’t generalize as well to informal tweet language for locations.

# MISC: 110
# The miscellaneous category captures entities that don’t fit 
# into PER, ORG, or LOC. In the dataset, 110 entities were classified as MISC. 
# These could be technical terms, project names, or other relevant phrases that
# the model deems important but doesn’t recognize as a standard entity type.