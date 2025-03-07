"""
Synthetic NER Dataset Generator for Insider Threat Detection (LLM-based, Colab Version)
================================================================

Overview
--------
This script generates synthetic data for Named Entity Recognition (NER) specifically tailored
to insider threat detection in short texts (tweets, messages, logs). It uses a hybrid approach
that combines LLM-generated content with spaCy rule-based augmentation to create a diverse,
high-quality dataset for specialized NER training.

This version is specifically adapted for Google Colab with Google Drive integration.

Architecture & Design Decisions
-----------------------------
1. Hybrid Generation Approach:
   - LLM Generation + Rule-based Augmentation for higher quality data
   - Using Claude 3.5 Sonnet for context-aware, realistic insider threat text generation
   - SpaCy pattern matching to ensure consistency and add additional examples

2. Domain-Specific Entity Types:
   - Standard NER categories: PERSON, ORG, LOC
   - Insider threat specific entities:
     * TIME_ANOMALY: Suspicious timing indicators
     * SENSITIVE_INFO: Confidential data references
     * TECH_ASSET: IT systems/assets
     * MEDICAL_CONDITION: Health-related entities
     * SUSPICIOUS_BEHAVIOR: Behavioral indicators
     * SENTIMENT_INDICATOR: Emotional signals

3. Pipeline Components:
[LLM-Generated Examples] → [Entity Extraction] → [spaCy Augmentation] → [Format Conversion] → [JSON Output]

4. Design Goals:
- Higher quality than rule-based generation alone
- More contextual understanding of insider threats
- Ability to generate nuanced examples beyond templates
- Accurate entity labeling for specialized entity types

Dependencies
-----------
- anthropic: Claude API for high-quality text generation
- spacy: NLP library for pattern matching and augmentation
- pandas: Data manipulation
- tqdm: Progress tracking
- json: Output formatting
- random: Randomization for data diversity

Google Colab Setup
-----------------
This script includes:
- Google Drive mounting
- GPU availability checking
- Path management for Google Drive
- Colab-friendly output handling

Output Format
------------
The script generates three files in your Google Drive:
- llm_synthetic_data.csv: Raw LLM-generated data
- augmented_synthetic_data.csv: Combined LLM + rule-based data
- insider_threat_ner_training.json: Final NER training data in spaCy format
- insider_threat_ner_training.jsonl: JSONL format for broader compatibility
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.matcher import Matcher
import random
from datetime import datetime, timedelta
import torch

import time
import sys
import pip

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Error mounting Google Drive. Please run this in Google Colab.")
    if not os.path.exists('/content/drive'):
        print("Creating placeholder directory for testing")
        os.makedirs('/content/drive/MyDrive/cs491_DataAnalyzer/Dataset', exist_ok=True)

# Check for GPU
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))

# Install required packages
def install_if_needed(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        pip.main(['install', package])

# Install anthropic if not available
install_if_needed('anthropic')
import anthropic

INSIDER_THREAT_ENTITIES = [
    "PERSON",                 # Standard person entity
    "ORG",                    # Standard organization entity
    "LOC",                    # Standard location entity
    "TIME_ANOMALY",           # References to off-hours, unusual timing
    "SENSITIVE_INFO",         # References to confidential data, credentials, etc.
    "TECH_ASSET",             # IT systems, databases, servers, etc.
    "MEDICAL_CONDITION",      # Health issues, symptoms
    "SUSPICIOUS_BEHAVIOR",    # Behavioral indicators of insider threats
    "SENTIMENT_INDICATOR"     # Words indicating negative sentiment
]

def setup_paths():
    """Set up paths for Google Drive storage"""
    base_path = "/content/drive/MyDrive/cs491_DataAnalyzer/Dataset"
    output_dir = os.path.join(base_path, "insider_threat_ner_data")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        'output_dir': output_dir,
        'llm_data': os.path.join(output_dir, "llm_synthetic_data.csv"),
        'augmented_data': os.path.join(output_dir, "augmented_synthetic_data.csv"),
        'training_json': os.path.join(output_dir, "insider_threat_ner_training.json"),
        'training_jsonl': os.path.join(output_dir, "insider_threat_ner_training.jsonl")
    }
    
    return paths

def generate_synthetic_ner_data(num_samples=100, batch_size=10):
    """        
    LLMs produce more realistic, nuanced examples than rule-based generation alone,
    with better understanding of both language patterns and insider threat context
    
    1. Creates prompts with diverse scenarios (accessing sensitive data, frustration, etc.)
    2. Requests Claude to generate synthetic tweets with structured entity annotations
    3. Parses responses and collects into dataframe
        
    Args:
        num_samples (int): Total number of examples to generate
        batch_size (int): Number of examples to generate in each API call
        
    Returns:
        pandas.DataFrame: DataFrame containing generated samples with columns:
            - tweet: The synthetic tweet text
            - entities: List of (entity_text, entity_type) tuples
            - classification: 'malicious' or 'non-malicious'
            - risk_score: Integer score from 0-100
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not found in environment variables.")
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    client = anthropic.Anthropic(api_key=api_key)
    results = []
    
    scenarios = [
        "employee accessing sensitive data outside work hours",
        "staff member expressing frustration with company policies",
        "user mentioning health issues affecting work",
        "employee discussing competitor companies",
        "suspicious login activity mentions",
        "references to financial struggles",
        "unusual system access patterns",
        "mentions of leaving the company"
    ]
    
    base_time = datetime(2025, 3, 1, 8, 0, 0)
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_size_actual = min(batch_size, num_samples - i)
        
        # Create prompt for batch generation
        prompt = f"""Generate {batch_size_actual} realistic tweets that might indicate insider threat behavior. 
        For each tweet, identify entities related to insider threat detection.
        
        Use these entity types:
        - PERSON: People's names or identifiers
        - ORG: Organizations, departments, company names
        - LOC: Locations, buildings, facilities
        - TIME_ANOMALY: References to unusual timing or off-hours
        - SENSITIVE_INFO: References to confidential data, credentials
        - TECH_ASSET: References to systems, databases, servers
        - MEDICAL_CONDITION: Health issues or symptoms
        - SUSPICIOUS_BEHAVIOR: Concerning activities or behaviors
        - SENTIMENT_INDICATOR: Words indicating negative sentiment
        
        For each tweet, also provide a 'malicious' or 'non-malicious' classification and a risk score from 0-100.
        
        Format each example as:
        
        Format each example as:
        Tweet: [tweet text]
        Entities: [("entity text", "ENTITY_TYPE"), ...]
        Classification: [malicious/non-malicious]
        Risk_Score: [0-100]
        
        Make sure to include diverse scenarios like: {", ".join(scenarios)}
        """
        
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8192,
                temperature=0.7,
                system="You are an expert in cybersecurity, healthcare/disability, medical conditions, IT, and insider threat detection.",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text if hasattr(response, 'content') and response.content else ""
            parsed_examples = parse_claude_output(content)
            # Add synthetic metadata to each example
            for idx, ex in enumerate(parsed_examples):
                ex['tweet_id'] = i + idx + 1
                ex['user_id'] = f"U{random.randint(1, 20):03d}"
                # Increment timestamp by a random delta (0-10 minutes) from base_time
                delta = timedelta(minutes=random.randint(0, 600))
                ex['timestamp'] = (base_time + delta).strftime("%Y-%m-%d %H:%M:%S")
                # Randomly assign sentiment if not provided by LLM (for demo purposes)
                if 'sentiment' not in ex:
                    ex['sentiment'] = random.choice(["Positive", "Neutral", "Negative"])
                # Off_hours flag based on hour
                hour = int(ex['timestamp'].split()[1].split(':')[0])
                ex['off_hours'] = "Yes" if hour < 6 or hour > 20 else "No"
            results.extend(parsed_examples)
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)[:200]}...")
            time.sleep(5)
            continue  # Skip on error for now
    
    df = pd.DataFrame(results)
    return df

def parse_claude_output(text):
    """
    Parses line-by-line to extract tweet text, entities, classification, and risk score
    
    Purpose:
        Transforms Claude's text responses into structured Python objects
        
    Args:
        text (str): Raw text response from Claude API
        
    Returns:
        list: List of dictionaries, each containing:
            - tweet: Tweet text
            - entities: List of (entity_text, entity_type) tuples
            - classification: 'malicious' or 'non-malicious'
            - risk_score: Integer from 0-100
    """
    examples = []
    current_example = {}
    
    # Simple parsing logic - will need to be adapted based on actual output format
    for line in text.split('\n'):
        if line.startswith('Tweet: '):
            if current_example and 'tweet' in current_example:
                examples.append(current_example)
                current_example = {}
            current_example['tweet'] = line[7:].strip()
        elif line.startswith('Entities: '):
            try:
                entities_str = line[10:].strip()
                entities = eval(entities_str)
                current_example['entities'] = entities
            except:
                current_example['entities'] = []
        elif line.startswith('Classification: '):
            current_example['classification'] = line[15:].strip()
        elif line.startswith('Risk_Score: '):
            try:
                current_example['risk_score'] = int(line[12:].strip())
            except:
                current_example['risk_score'] = 0
    if current_example and 'tweet' in current_example:
        examples.append(current_example)
    return examples

def create_spacy_patterns():
    """
    
    Uses spaCy's pattern matching for consistent, rule-based entity detection        
    
    Returns:
        tuple: (nlp, matcher) where:
            - nlp is the spaCy language model
            - matcher is a configured spaCy Matcher with patterns
            
    Process:
        Creates detailed patterns for each specialized entity type:
        - TIME_ANOMALY: Patterns for unusual access times
        - SENSITIVE_INFO: Patterns for confidential information
        - TECH_ASSET: Patterns for technical systems
        - MEDICAL_CONDITION: Patterns for health issues
        - SUSPICIOUS_BEHAVIOR: Patterns for concerning actions
        - SENTIMENT_INDICATOR: Patterns for negative emotions
    """
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        print("Installing spaCy model...")
        os.system("python -m spacy download en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    
    matcher = Matcher(nlp.vocab)
    
    """ 
    This is a list containing three separate pattern lists that is used for data augmentation.
    These patterns lists are used by the spaCy Matcher to find additional entities that the LLM may not have captured. 
    Each of these lists is a sequence of dictionaries. 
    Each dictionary specifies criteria for one token in the sequence. 
    When a document is processed, spaCy’s Matcher will try to find token sequences that match these criteria. 
    
    """
    time_anomaly_patterns = [
        [{"LOWER": {"IN": ["midnight", "late", "night", "weekend", "after", "hours"]}}, 
         {"POS": "NOUN", "OP": "?"}],
        [{"LOWER": "at"}, {"SHAPE": "d:dd"}],
        [{"LOWER": {"IN": ["3am", "4am", "11pm", "midnight"]}}, {"POS": "ADP", "OP": "?"}]
    ]
    matcher.add("TIME_ANOMALY", time_anomaly_patterns)
    
    sensitive_info_patterns = [
        [{"LOWER": {"IN": ["password", "credentials", "confidential", "secret"]}}, 
         {"POS": "NOUN", "OP": "?"}],
        [{"LOWER": {"IN": ["database", "db"]}}, {"LOWER": "access"}],
        [{"LOWER": {"IN": ["document", "file", "data"]}}, {"LOWER": {"IN": ["classified", "sensitive", "private"]}}]
    ]
    matcher.add("SENSITIVE_INFO", sensitive_info_patterns)
    
    tech_asset_patterns = [
        [{"LOWER": {"IN": ["server", "database", "system", "network", "vpn"]}}, 
         {"POS": "NOUN", "OP": "?"}],
        [{"LOWER": {"IN": ["admin", "root", "sudo"]}}, {"LOWER": "access", "OP": "?"}],
        [{"LOWER": {"IN": ["production", "staging", "test"]}}, {"LOWER": {"IN": ["environment", "system", "server"]}}]
    ]
    matcher.add("TECH_ASSET", tech_asset_patterns)
    
    medical_condition_patterns = [
        [{"LOWER": {"IN": ["depression", "anxiety", "stress", "insomnia", "burnout"]}}, 
         {"POS": "NOUN", "OP": "?"}],
        [{"POS": "ADJ"}, {"LOWER": {"IN": ["disorder", "syndrome", "condition"]}}],
        [{"LOWER": "mental"}, {"LOWER": "health", "OP": "?"}]
    ]
    matcher.add("MEDICAL_CONDITION", medical_condition_patterns)
    
    suspicious_behavior_patterns = [
        [{"LOWER": {"IN": ["downloading", "copying", "exporting", "transferring"]}}, 
         {"POS": "NOUN", "OP": "*"}],
        [{"LOWER": {"IN": ["bypassing", "circumventing", "avoiding"]}}, {"POS": "NOUN", "OP": "*"}],
        [{"LOWER": {"IN": ["unauthorized", "illegal", "suspicious"]}}, {"POS": "NOUN", "OP": "+"}]
    ]
    matcher.add("SUSPICIOUS_BEHAVIOR", suspicious_behavior_patterns)
    
    sentiment_indicator_patterns = [
        [{"LOWER": {"IN": ["hate", "angry", "frustrated", "annoyed", "upset"]}}, 
         {"POS": "NOUN", "OP": "?"}],
        [{"LOWER": {"IN": ["tired", "fed up", "done", "quitting"]}}, {"POS": "ADP", "OP": "?"}],
        [{"LOWER": {"IN": ["deserve", "revenge", "payback", "show them"]}}, {"POS": "PRON", "OP": "?"}]
    ]
    matcher.add("SENTIMENT_INDICATOR", sentiment_indicator_patterns)
    
    return nlp, matcher

def augment_llm_data_with_spacy(df):
    """
    Hybrid approach combining LLM quality with template efficiency
    Uses entities from LLM data and calls create_spacy_patterns()
    Dramatically increases dataset size while maintaining quality and diversity,
    focusing on scenario-specific templates for insider threats
    
    Purpose:
        Expands the dataset with additional examples using templates
        
    Args:
        df (pandas.DataFrame): DataFrame containing LLM-generated examples
        
    Returns:
        pandas.DataFrame: Combined dataset of original + augmented examples
        
    Process:
        1. Extracts entities from LLM data
        2. Fills templates with these entities to create diverse new examples
        3. Tracks precise character positions for entity spans
        4. Adds appropriate classification and risk scores
    """
    nlp, matcher = create_spacy_patterns()
    augmented_tweets = []
    
    # Templates for generating additional examples - expanded set
    templates = [
        # Time-based anomalies
        "Need to {SUSPICIOUS_BEHAVIOR} the {TECH_ASSET} {TIME_ANOMALY} to finish the project",
        "Working on {TECH_ASSET} {TIME_ANOMALY} due to tight deadline",
        "Had to access {SENSITIVE_INFO} {TIME_ANOMALY} because management needed it urgently",
        
        # Sentiment and behavior
        "Feeling {SENTIMENT_INDICATOR} about the way {ORG} treats its employees",
        "My {MEDICAL_CONDITION} is making me consider {SUSPICIOUS_BEHAVIOR}",
        "So {SENTIMENT_INDICATOR} at {ORG} after what happened with {PERSON} yesterday",
        
        # Technical assets and sensitive info
        "Found a way to {SUSPICIOUS_BEHAVIOR} the {TECH_ASSET} without anyone noticing",
        "Could easily {SUSPICIOUS_BEHAVIOR} all {SENSITIVE_INFO} before leaving {ORG}",
        "Need to download {SENSITIVE_INFO} for the meeting at {LOC} tomorrow",
        
        # Medical and personal issues
        "My {MEDICAL_CONDITION} is getting worse because of the stress at {ORG}",
        "Taking time off because my {MEDICAL_CONDITION} is flaring up again",
        "Can't believe {ORG} won't let me work remotely despite my {MEDICAL_CONDITION}",
        
        # Vague threats
        "They'll regret treating me this way at {ORG} when I {SUSPICIOUS_BEHAVIOR}",
        "{ORG} won't know what hit them when I finally {SUSPICIOUS_BEHAVIOR}",
        "Just a few more days at {ORG} before I {SUSPICIOUS_BEHAVIOR}",
        
        # Complex patterns (multiple entities)
        "{PERSON} from {ORG} showed me how to {SUSPICIOUS_BEHAVIOR} the {TECH_ASSET} {TIME_ANOMALY}",
        "Moving all {SENSITIVE_INFO} to my personal drive because of my {SENTIMENT_INDICATOR} with {ORG}",
        "My {MEDICAL_CONDITION} makes it hard to work at {LOC} so I have to {SUSPICIOUS_BEHAVIOR}"
    ]
    
    # Extract entities from LLM data to reuse
    all_entities = {}
    for entity_type in INSIDER_THREAT_ENTITIES:
        all_entities[entity_type] = []
    
    # Extract entities from existing data
    for _, row in df.iterrows():
        for entity_text, entity_type in row.get('entities', []):
            if entity_type in all_entities:
                all_entities[entity_type].append(entity_text)
    
    # Fallback default values in case no entities were found
    default_entities = {
        "PERSON": ["John", "Sarah", "Michael", "David", "Lisa"],
        "ORG": ["Acme Corp", "the company", "IT department", "HR", "management"],
        "LOC": ["headquarters", "the office", "building B", "server room", "downtown office"],
        "TIME_ANOMALY": ["at midnight", "on the weekend", "after hours", "at 2am", "during the holiday"],
        "SENSITIVE_INFO": ["customer data", "financial records", "employee files", "passwords", "credentials"],
        "TECH_ASSET": ["admin portal", "database", "server", "main system", "source code"],
        "MEDICAL_CONDITION": ["depression", "anxiety", "chronic pain", "insomnia", "migraines"],
        "SUSPICIOUS_BEHAVIOR": ["download everything", "bypass security", "delete records", "access restricted areas", "copy files"],
        "SENTIMENT_INDICATOR": ["angry", "frustrated", "upset", "fed up", "irritated"]
    }
    
    # Add default entities if none found in LLM data
    for entity_type in INSIDER_THREAT_ENTITIES:
        if not all_entities[entity_type] and entity_type in default_entities:
            all_entities[entity_type] = default_entities[entity_type]
    
    # Generate new examples with templates
    print("Generating augmented examples...")
    for _ in tqdm(range(500)):  # Hardcoded for now
        template = random.choice(templates)
        
        # Fill template with entities
        tweet = template
        entities = []
        start_positions = {}
        
        # Track character positions for precise entity spans
        for entity_type in all_entities:
            placeholder = f"{{{entity_type}}}"
            if placeholder in tweet and all_entities[entity_type]:
                entity = random.choice(all_entities[entity_type])
                start_pos = tweet.find(placeholder)
                start_positions[start_pos] = (entity, entity_type, len(placeholder))
        
        # Replace placeholders with entities in order of appearance
        offset = 0
        ordered_positions = sorted(start_positions.keys())
        
        for pos in ordered_positions:
            entity, entity_type, placeholder_len = start_positions[pos]
            adjusted_pos = pos + offset
            tweet = tweet[:adjusted_pos] + entity + tweet[adjusted_pos + placeholder_len:]
            
            # Record the entity with precise character positions
            entities.append((entity, entity_type))
            offset += len(entity) - placeholder_len
        
        # Add to results with classification and risk score
        is_malicious = random.random() > 0.7 or "SUSPICIOUS_BEHAVIOR" in template
        risk_score = random.randint(60, 95) if is_malicious else random.randint(5, 40)
        
        augmented_tweets.append({
            'tweet_id': random.randint(1000, 9999),
            'user_id': f"U{random.randint(1, 20):03d}",
            'tweet': tweet,
            'timestamp': (datetime(2025, 3, random.randint(1,10), random.randint(0,23), random.randint(0,59))).strftime("%Y-%m-%d %H:%M:%S"),
            'sentiment': random.choice(["Positive", "Neutral", "Negative"]),
            'off_hours': random.choice(["Yes", "No"]),
            'entities': entities,
            'classification': 'malicious' if is_malicious else 'non-malicious',
            'risk_score': risk_score
        })
    
    # Combine with original data
    print(f"Generated {len(augmented_tweets)} augmented examples")
    return pd.concat([df, pd.DataFrame(augmented_tweets)], ignore_index=True)

def convert_to_spacy_format(df, output_file):
    """Convert DataFrame to spaCy training format.
    
    Why Multiple Formats?:
        Different training frameworks require different formats:
        - spaCy format for direct use with spaCy
        - JSONL for broader compatibility with other frameworks
        
    Args:
        df (pandas.DataFrame): DataFrame with tweet text and entities
        output_file (str): Path to save the output JSON file
        
    Returns:
        tuple: (training_data, jsonl_data) where:
            - training_data: List in spaCy's native training format
            - jsonl_data: List of dictionaries in JSONL format
            
    Process:
        1. Converts entity annotations to character spans
        2. Creates two output formats (spaCy JSON and JSONL)
        3. Ensures entity boundaries are valid (word boundary checks)
        4. Saves both formats to disk
    """
    training_data = []
    jsonl_data = []
    
    print("Converting to training formats...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['tweet']
        entities = []
        jsonl_entities = []
        
        # Convert our entity format to spaCy's format with precise character positions
        for entity_text, entity_type in row.get('entities', []):
            # Find all occurrences of this entity text
            start = 0
            while True:
                start = text.find(entity_text, start)
                if start == -1:
                    break
                    
                end = start + len(entity_text)
                
                # Check if we're not splitting a word (basic word boundary check)
                valid_start = start == 0 or not text[start-1].isalnum()
                valid_end = end == len(text) or not text[end].isalnum()
                
                if valid_start and valid_end:
                    entities.append((start, end, entity_type))
                    jsonl_entities.append({
                        "start": start,
                        "end": end,
                        "label": entity_type,
                        "text": entity_text
                    })
                
                start = end  # Move to position after this occurrence
        
        # Add both formats
        training_data.append((text, {"entities": entities}))
        jsonl_data.append({
            "text": text,
            "entities": jsonl_entities,
            "classification": row.get('classification', 'non-malicious'),
            "risk_score": row.get('risk_score', 0)
        })
    
    # Save training data in spaCy format
    with open(output_file, 'w') as f:
        json.dump(training_data, f)
    print(f"Saved spaCy format training data to {output_file}")
        
    # Save as JSONL format (like the medical examples) for more compatibility
    jsonl_path = output_file.replace('.json', '.jsonl')
    with open(jsonl_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved JSONL format training data to {jsonl_path}")
    
    return training_data, jsonl_data

def main():
    """Generate synthetic insider threat NER dataset
    
    Purpose:
        Orchestrates the entire process flow of dataset generation
        
    Process:
        1. Setup paths for data storage
        2. Generate base data with LLM (Claude)
        3. Augment with spaCy rules and templates
        4. Convert to training formats (spaCy and JSONL)
        5. Generate and display statistics
        
    Design Decision:
        Modular workflow allows for isolated testing and development of components
        
    Relations:
        Calls all other functions in sequence:
        setup_paths → generate_synthetic_ner_data → augment_llm_data_with_spacy → convert_to_spacy_format
        
    Output Files:
        - llm_synthetic_data.csv: Raw LLM-generated data
        - augmented_synthetic_data.csv: Combined LLM + rule-based data
        - insider_threat_ner_training.json: Final NER training data in spaCy format
        - insider_threat_ner_training.jsonl: JSONL format for broader compatibility
    """
    # Setup paths
    paths = setup_paths()
    
    print("\n===== 1: Generating Synthetic Data with LLM =====")
    num_samples = 400 # Note, this only generates 400 samples from LLM, the final number of samples will be atleast double with augmentation
    llm_data = generate_synthetic_ner_data(num_samples=num_samples)
    llm_data.to_csv(paths['llm_data'], index=False)
    print(f"Raw LLM data saved to {paths['llm_data']}")
    
    print("\n===== 2: Augmenting Data with spaCy Patterns =====")
    augmented_data = augment_llm_data_with_spacy(llm_data)
    augmented_data.to_csv(paths['augmented_data'], index=False)
    print(f"Augmented data saved to {paths['augmented_data']}")
    
    print("\n===== 3: Converting to Training Format =====")
    training_data, jsonl_data = convert_to_spacy_format(
        augmented_data, 
        paths['training_json']
    )
    
    print("\n===== 4: Generating Statistics =====")
    entity_counts = {}
    for item in jsonl_data:
        for entity in item["entities"]:
            label = entity["label"]
            if label not in entity_counts:
                entity_counts[label] = 0
            entity_counts[label] += 1
    
    print(f"\nGenerated {len(augmented_data)} synthetic examples with insider threat NER annotations")
    print(f"Data saved to Google Drive at {paths['output_dir']}")
    print("\nEntity distribution:")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"- {entity_type}: {count}")
    
    print("\nSample generated example:")
    sample = jsonl_data[0]
    print(f"Text: {sample['text']}")
    print("Entities:")
    for entity in sample["entities"]:
        print(f"- {entity['text']} ({entity['label']}) at positions {entity['start']}:{entity['end']}")
    print(f"Classification: {sample['classification']}")
    print(f"Risk Score: {sample['risk_score']}")
    
    print("\n===== COMPLETED SUCCESSFULLY =====")
    print(f"All output files are available in your Google Drive at {paths['output_dir']}")

if __name__ == "__main__":
    main()