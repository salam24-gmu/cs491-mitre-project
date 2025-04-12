import os
import logging
import asyncio
from typing import Dict, List, Union, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to store model components
model = None
tokenizer = None
ner_pipeline = None
thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)

# Entity labels from our NER model
INSIDER_THREAT_LABELS = [
    "O",  # Outside any entity
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-TIME_ANOMALY", "I-TIME_ANOMALY",
    "B-SENSITIVE_INFO", "I-SENSITIVE_INFO",
    "B-TECH_ASSET", "I-TECH_ASSET",
    "B-MEDICAL_CONDITION", "I-MEDICAL_CONDITION",
    "B-SUSPICIOUS_BEHAVIOR", "I-SUSPICIOUS_BEHAVIOR",
    "B-SENTIMENT_INDICATOR", "I-SENTIMENT_INDICATOR"
]

async def load_model() -> None:
    """
    Load the NER model and tokenizer into memory.
    This is called during application startup.
    """
    global model, tokenizer, ner_pipeline
    
    try:
        logger.info(f"Loading model from {settings.MODEL_PATH}")
        
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"Model path {settings.MODEL_PATH} does not exist")
        
        # Run the model loading in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # Load model and tokenizer in the thread pool
        model, tokenizer = await loop.run_in_executor(
            thread_pool,
            _load_model_and_tokenizer
        )
        
        # Create the NER pipeline in the thread pool
        ner_pipeline = await loop.run_in_executor(
            thread_pool,
            _create_pipeline,
            model,
            tokenizer
        )
        
        logger.info(f"Model loaded successfully on CPU (device={settings.DEVICE})")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def _load_model_and_tokenizer():
    """
    Helper function to load model and tokenizer in a separate thread.
    Optimized for CPU usage.
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
        
        # Load model for token classification, explicitly on CPU
        model = AutoModelForTokenClassification.from_pretrained(
            settings.MODEL_PATH,
            device_map="cpu",
            torchscript=True,  # Enable TorchScript for better CPU performance
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise

def _create_pipeline(model, tokenizer):
    """
    Helper function to create the pipeline in a separate thread.
    """
    try:
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            # NOTE: when loading the model in (AutoModelForTokenClassification.from_pretrained(...)), the underlying "transformers" library used the "accelerate" library handles how the mode's components are loaded onto devices automatically.
            # device=settings.DEVICE,  # -1 for CPU
            batch_size=1  # Use batch size 1 for now, can be adjusted for better performance
        )
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise

async def cleanup_model() -> None:
    """
    Clean up model resources.
    This is called during application shutdown.
    """
    global model, tokenizer, ner_pipeline, thread_pool
    
    logger.info("Cleaning up model resources")
    
    # Shut down thread pool
    thread_pool.shutdown(wait=True)
    
    # Clear model references to free memory
    model = None
    tokenizer = None
    ner_pipeline = None
    
    # Create a new thread pool
    thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)
    
    logger.info("Model cleanup completed")

async def run_inference(text: str) -> List[Dict[str, Any]]:
    """
    Run NER inference on the provided text.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected entities with their positions, types, and confidence scores
    """
    if ner_pipeline is None:
        raise RuntimeError("Model not loaded. Please ensure the model is loaded before inference.")
    
    logger.debug(f"Running inference on text: {text[:50]}...")
    
    # Run the inference in a separate thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            thread_pool,
            ner_pipeline,
            text
        )
        return result
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise RuntimeError(f"Inference failed: {str(e)}")

def evaluate_entities(entities: List[Dict[str, Any]], text: str, tweet_timestamp: Optional[str] = None) -> str:
    """
    Formats the detected entities into a human-readable analysis.
    
    Args:
        entities: List of entities detected by the NER pipeline
        text: The original text that was analyzed
        tweet_timestamp: Optional timestamp for the tweet
        
    Returns:
        Formatted analysis text
    """
    if not entities:
        return f"No entities found in text: {text}"
    
    # Group entities by type
    entity_types = {}
    for entity in entities:
        entity_type = entity["entity_group"]
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)
    
    # Flag high-risk combinations
    flagged_indicators = []
    if "SUSPICIOUS_BEHAVIOR" in entity_types and "SENSITIVE_INFO" in entity_types:
        flagged_indicators.append("SUSPICIOUS_BEHAVIOR + SENSITIVE_INFO")
    if "TIME_ANOMALY" in entity_types and "TECH_ASSET" in entity_types:
        flagged_indicators.append("TIME_ANOMALY + TECH_ASSET")
    if "SUSPICIOUS_BEHAVIOR" in entity_types and "TIME_ANOMALY" in entity_types:
        flagged_indicators.append("SUSPICIOUS_BEHAVIOR + TIME_ANOMALY")
    if "SUSPICIOUS_BEHAVIOR" in entity_types and "TECH_ASSET" in entity_types:
        flagged_indicators.append("SUSPICIOUS_BEHAVIOR + TECH_ASSET")
    
    # Format the result
    result = f"Text: {text}\n\nEntities detected:\n"
    for entity_type, entities_of_type in entity_types.items():
        result += f"\n{entity_type}:\n"
        for entity in entities_of_type:
            result += f"  - {entity['word']} (Score: {entity['score']:.4f})\n"
    
    if flagged_indicators:
        result += "\nFlagged High-Risk Indicators:\n"
        for indicator in flagged_indicators:
            result += f"  - {indicator}\n"
        
        # Assign risk level based on number of high-risk indicators
        risk_level = "HIGH" if len(flagged_indicators) >= 2 else "MEDIUM"
        result += f"\nOverall Risk Assessment: {risk_level}\n"
    else:
        result += "\nNo high-risk indicators flagged.\n"
        result += "\nOverall Risk Assessment: LOW\n"
    
    if tweet_timestamp:
        result += f"\nTimestamp: {tweet_timestamp}\n"
    
    return result 