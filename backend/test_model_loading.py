#!/usr/bin/env python
"""
Model Loading Test Script

This script tests loading the NER model directly from the model files
and running inference to ensure everything is working correctly.
"""

import os
import sys
import json
import time
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

def find_model_path():
    # Start with current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible model locations
    possible_paths = [
        os.path.join(base_dir, "..", "finetune", "data", "insider-threat-ner_model", "model-4n4ekgnp"),
        os.path.join(base_dir, "models", "model-4n4ekgnp"),
        os.path.join(base_dir, "..", "models", "model-4n4ekgnp"),
        os.path.join(base_dir, "..", "finetune", "data", "insider-threat-ner_model")
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            # Check if this path has necessary model files
            if (os.path.exists(os.path.join(path, "config.json")) or 
                os.path.exists(os.path.join(path, "model.safetensors"))):
                print(f"Found model at: {path}")
                return path
    
    print("Model not found in any of the expected locations!")
    return None

def test_model_loading(model_path):
    """Test loading the model and tokenizer"""
    print(f"Testing model loading from: {model_path}")
    start_time = time.time()
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded successfully")
        
        print("Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            device_map="cpu",
            torchscript=True,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully")
        
        load_time = time.time() - start_time
        print(f"Model loading completed in {load_time:.2f} seconds")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def test_inference(model, tokenizer):
    """Test running inference with the model"""
    if not model or not tokenizer:
        print("Cannot run inference test - model or tokenizer not loaded")
        return
    
    print("\nCreating NER pipeline...")
    try:
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            # device=-1  # Use CPU
        )
        print("Pipeline created successfully")
        
        # Example texts to test
        # test_texts = [
        #     "I'm planning to download the customer database tonight after everyone has gone home.",
        #     "I need the admin password for the server room door.",
        #     "I've been working late every night this week, but nobody appreciates my hard work.",
        #     "I'm going to copy some files to my personal device for the weekend."
        # ]
        
        test_texts = [
            "[2025-03-01 16:53:00] I'm planning to download the customer database tonight after everyone has gone home.",
            "[2025-03-01 16:53:00] I need the admin password for the server room door.",
            "[2025-03-01 16:53:00] I've been working late every night this week, but nobody appreciates my hard work.",
            "[2025-03-01 16:53:00] I'm going to copy some files to my personal device for the weekend."
        ]
        
        print("\nRunning inference tests on sample texts:")
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: {text}")
            
            start_time = time.time()
            entities = ner_pipeline(text)
            inference_time = time.time() - start_time
            
            print(f"Detected {len(entities)} entities in {inference_time:.3f} seconds:")
            
            # Group by entity type
            entity_types = {}
            for entity in entities:
                entity_type = entity["entity_group"]
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity)
            
            # Display results
            for entity_type, entities_list in entity_types.items():
                print(f"  {entity_type}:")
                for entity in entities_list:
                    print(f"    - {entity['word']} (Score: {entity['score']:.4f})")
        
    except Exception as e:
        print(f"Error during inference testing: {str(e)}")

def main():
    """Main test function"""
    print("="*60)
    print("MODEL LOADING TEST SCRIPT")
    print("="*60)
    
    # Find model path
    model_path = find_model_path()
    if not model_path:
        print("Could not find model path. Exiting.")
        sys.exit(1)
    
    # Test model loading
    model, tokenizer = test_model_loading(model_path)
    
    # Test inference
    test_inference(model, tokenizer)
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 