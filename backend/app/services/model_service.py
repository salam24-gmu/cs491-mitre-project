import os
import logging
import asyncio
import joblib
import json
import numpy as np
from typing import Dict, List, Union, Any, Optional, Tuple
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
classification_model = None
classification_scaler = None
classification_feature_names = None
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
    # Ensure assignment updates the global variables, not local ones
    global model, tokenizer, ner_pipeline, classification_model, classification_scaler, classification_feature_names
    
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
        
        # Load classification artifacts in thread pool
        logger.info(f"Loading classification model from {settings.CLASSIFICATION_MODEL_PATH}")
        # Assign result to the *global* variables declared above
        classification_model, classification_scaler, classification_feature_names = await loop.run_in_executor(
            thread_pool,
            _load_classification_artifacts
        )
        if classification_model:
            logger.info("Classification model artifacts loaded successfully.")
        else:
            logger.warning("Classification model artifacts failed to load. Prediction disabled.")
        
    except Exception as e:
        logger.error(f"Failed to load model(s): {str(e)}")
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

def _load_classification_artifacts():
    """Loads the classification model, scaler, and feature names."""
    model_run_path = settings.CLASSIFICATION_MODEL_PATH
    model_path = os.path.join(model_run_path, 'random_forest_model.joblib')
    scaler_path = os.path.join(model_run_path, 'scaler.joblib')
    features_path = os.path.join(model_run_path, 'feature_names.json')

    loaded_model, loaded_scaler, loaded_feature_names = None, None, None

    try:
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            logger.info(f"Loaded classification model from: {model_path}")
        else:
            logger.error(f"Classification model not found at {model_path}")

        if os.path.exists(scaler_path):
            loaded_scaler = joblib.load(scaler_path)
            logger.info(f"Loaded classification scaler from: {scaler_path}")
        else:
            logger.error(f"Classification scaler not found at {scaler_path}")

        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                loaded_feature_names = json.load(f)
            logger.info(f"Loaded {len(loaded_feature_names)} classification feature names from: {features_path}")
        else:
            logger.error(f"Classification feature names file not found at {features_path}")

        if loaded_model and loaded_scaler and loaded_feature_names:
            return loaded_model, loaded_scaler, loaded_feature_names
        else:
            logger.warning("Failed to load all classification artifacts. Classification prediction will be disabled.")
            return None, None, None
    except Exception as e:
        logger.error(f"Error loading classification artifacts from {model_run_path}: {e}")
        return None, None, None

def _create_pipeline(model, tokenizer):
    """
    Helper function to create the pipeline in a separate thread.
    """
    try:
        # aggregation_strategy is a parameter for HuggingFace's transformers.pipeline (NER) that controls how token-level predictions
        # are grouped into higher-level entities. This is especially important for models that output predictions per token,
        # but we want to return whole words or phrases as entities.
        #
        # Options for aggregation_strategy (as of transformers v4.35.0):
        #   - "none": No aggregation; each token is a separate entity.
        #   - "simple": Consecutive tokens with the same entity label are grouped together (most common for NER).
        #   - "first": Only the first token of a group is returned.
        #   - "average": Scores of grouped tokens are averaged.
        #   - "max": The token with the highest score in a group is used.
        #
        # "simple" mode (used here) means that if the model predicts a multi-token entity (e.g., "confidential data"),
        # the pipeline will merge those tokens into a single entity span, making the output more human-readable and useful.
        #
        # Example:
        #   Input: "User downloaded confidential data at 2am."
        #   Token-level output (aggregation_strategy="none"):
        #     [{'entity_group': 'B-SENSITIVE_INFO', 'word': 'confidential', ...},
        #      {'entity_group': 'I-SENSITIVE_INFO', 'word': 'data', ...}]
        #   Output with aggregation_strategy="simple":
        #     [{'entity_group': 'SENSITIVE_INFO', 'word': 'confidential data', ...}]
        #
        # Sources:
        #   - https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline
        #   - https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py
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
    classification_model = None
    classification_scaler = None
    classification_feature_names = None
    
    # Create a new thread pool
    thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)
    
    logger.info("Model cleanup completed")



async def run_inference(text: str) -> List[Dict[str, Any]]:
    """
    Run NER inference on the provided text using a thread pool to ensure concurrency
    and non-blocking behavior in the FastAPI async context.

    Deep Dive:
    -----------
    - The NER pipeline (ner_pipeline) is a HuggingFace transformers pipeline, which is CPU/GPU-bound and not async-aware.
    - FastAPI endpoints are async, so blocking operations (like model inference) must be offloaded to a thread pool.
    - We use `loop.run_in_executor` to submit the inference task to a dedicated ThreadPoolExecutor (`thread_pool`).
    - This allows multiple requests to be handled concurrently without blocking the main event loop.
    - The thread pool size is controlled by settings.THREAD_POOL_SIZE, so concurrency is limited to that number of threads.
    - Each inference call is thread-safe as long as the pipeline/model is thread-safe (HuggingFace pipelines are generally thread-safe for inference).
    - If the model is not loaded (`ner_pipeline is None`), we raise an error to avoid undefined behavior.

    Example Data Flow:
    ------------------
    Input:
        text = "User downloaded confidential data at 2am from server."
    Output:
        [
            {'entity_group': 'SUSPICIOUS_BEHAVIOR', 'score': 0.98, 'word': 'downloaded', 'start': 5, 'end': 15},
            {'entity_group': 'SENSITIVE_INFO', 'score': 0.95, 'word': 'confidential data', 'start': 16, 'end': 33},
            {'entity_group': 'TIME_ANOMALY', 'score': 0.92, 'word': '2am', 'start': 38, 'end': 41},
            {'entity_group': 'TECH_ASSET', 'score': 0.90, 'word': 'server', 'start': 47, 'end': 53}
        ]

    Returns:
        List of detected entities with their positions, types, and confidence scores.

    Sources:
        - https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
        - https://fastapi.tiangolo.com/advanced/concurrency/
        - https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline

    Args:
        text (str): The text to analyze.

    Returns:
        List[Dict[str, Any]]: List of detected entities.

    """
    # NER pipeline definition (for reference, see above in this file):
    # ner_pipeline = pipeline(
    #     "ner",
    #     model=model,
    #     tokenizer=tokenizer,
    #     aggregation_strategy="simple",
    #     batch_size=1
    # )
    # The ner_pipeline is initialized in the load_model() async function and stored as a global variable.
    if ner_pipeline is None:
        # Model not loaded, cannot proceed
        raise RuntimeError("Model not loaded. Please ensure the model is loaded before inference.")

    logger.debug(f"Running inference on text: {text[:50]}...")

    # Get the current event loop (thread-safe in FastAPI context)
    loop = asyncio.get_running_loop()
    try:
        # Offload the blocking NER pipeline call to the thread pool
        # This ensures the FastAPI server can handle other requests concurrently
        result = await loop.run_in_executor(
            thread_pool,  # ThreadPoolExecutor instance
            ner_pipeline, # Callable: the NER pipeline
            text          # Argument: the text to analyze
        )
        # Example: If text = "User downloaded confidential data at 2am from server."
        # result might be:
        # [
        #   {'entity_group': 'SUSPICIOUS_BEHAVIOR', 'score': 0.98, 'word': 'downloaded', ...},
        #   ...
        # ]
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

# +++ Add Feature Extraction and Prediction Logic +++
# NOTE: These constants need to be defined or imported if not already available
# Assuming they are defined similar to the Colab script globally or can be imported
# For simplicity, defining them here, but ideally manage them centrally.
ALL_ENTITY_TYPES = sorted([
    "SUSPICIOUS_BEHAVIOR", "SENSITIVE_INFO", "TIME_ANOMALY", "TECH_ASSET",
    "MEDICAL_CONDITION", "SENTIMENT_INDICATOR", "PERSON", "ORG", "LOC"
])
ALL_CONTEXT_SIGNALS = sorted([
    "SECURITY_CONTEXT", "SECRECY_INDICATORS", "FINANCIAL_MOTIVATION",
    "SECURITY_WITH_SECRECY"
])
ALL_SEMANTIC_INDICATORS = sorted([
    "NEGATIVE_SENTIMENT", "THREATENING_LANGUAGE", "PERSONAL_GRIEVANCE",
    "PERSONAL_NEGATIVITY", "PERSONAL_THREAT"
])

def extract_hybrid_tweet_features(
    risk_metrics: Dict[str, Any],
    all_entity_types: List[str],
    all_context_signals: List[str],
    all_semantic_indicators: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts a feature vector for a single tweet based on its risk_metrics.
    (Copied/adapted from Colab script - ensure consistency)
    """
    features = {}
    feature_names = []

    core_scores = [
        'total_risk', 'base_risk', 'adjusted_base_risk', 'context_risk',
        'semantic_risk', 'risk_multiplier', 'high_risk_combinations', 'risk_density'
    ]
    for score_name in core_scores:
        features[score_name] = risk_metrics.get(score_name, 0.0)
        feature_names.append(score_name)

    features['off_hours'] = 1.0 if risk_metrics.get('off_hours', False) else 0.0
    feature_names.append('off_hours')

    entity_counts = risk_metrics.get('entity_type_counts', {})
    for entity_type in all_entity_types:
        key = f"count_{entity_type.lower()}"
        features[key] = float(entity_counts.get(entity_type, 0))
        feature_names.append(key)

    context_signals_present = set(risk_metrics.get('context_signals', []))
    for signal in all_context_signals:
        key = f"signal_{signal.lower()}"
        features[key] = 1.0 if signal in context_signals_present else 0.0
        feature_names.append(key)

    semantic_indicators_present = set(risk_metrics.get('semantic_indicators', []))
    for indicator in all_semantic_indicators:
        key = f"indicator_{indicator.lower()}"
        features[key] = 1.0 if indicator in semantic_indicators_present else 0.0
        feature_names.append(key)

    feature_vector = np.array([features[name] for name in feature_names], dtype=float)
    return feature_vector, feature_names

async def predict_maliciousness(risk_metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Predicts the maliciousness of a tweet using the loaded classification model.
    Runs the prediction logic in the thread pool.

    Args:
        risk_metrics (Dict[str, Any]): Risk metrics dictionary for the tweet.

    Returns:
        Tuple[Optional[str], Optional[float]]: predicted_class, malicious_probability
    """
    global classification_model, classification_scaler, classification_feature_names

    if not all([classification_model, classification_scaler, classification_feature_names]):
        logger.warning("Classification artifacts not loaded. Skipping prediction.")
        return None, None

    loop = asyncio.get_running_loop()
    try:
        # Offload the prediction logic to the thread pool
        predicted_class, malicious_probability = await loop.run_in_executor(
            thread_pool,
            _run_classification_prediction, # Helper function for sync code
            risk_metrics,
            classification_model,
            classification_scaler,
            classification_feature_names
        )
        return predicted_class, malicious_probability
    except Exception as e:
        logger.error(f"Error during classification prediction task: {e}")
        return None, None

def _run_classification_prediction(
    risk_metrics: Dict[str, Any],
    model: Any,
    scaler: Any,
    expected_feature_names: List[str]
) -> Tuple[Optional[str], Optional[float]]:
    """Synchronous helper function to run the classification prediction."""
    try:
        # 1. Extract features
        feature_vector, feature_names = extract_hybrid_tweet_features(
            risk_metrics,
            ALL_ENTITY_TYPES,
            ALL_CONTEXT_SIGNALS,
            ALL_SEMANTIC_INDICATORS
        )

        # 2. Verify and align features
        if feature_names != expected_feature_names:
            logger.warning(f"Feature name mismatch. Current: {len(feature_names)} vs Expected: {len(expected_feature_names)}")
            if set(feature_names) == set(expected_feature_names):
                feature_dict = dict(zip(feature_names, feature_vector))
                aligned_vector = np.array([feature_dict.get(name, 0.0) for name in expected_feature_names])
                feature_vector = aligned_vector
                logger.info("Feature vector reordered.")
            else:
                 logger.error("Cannot align feature names. Aborting prediction.")
                 missing = set(expected_feature_names) - set(feature_names)
                 extra = set(feature_names) - set(expected_feature_names)
                 if missing: logger.error(f"Missing expected features: {missing}")
                 if extra: logger.error(f"Found unexpected features: {extra}")
                 return None, None

        # 3. Reshape
        feature_vector_2d = feature_vector.reshape(1, -1)

        # 4. Scale
        scaled_features = scaler.transform(feature_vector_2d)

        # 5. Predict class
        predicted_class = model.predict(scaled_features)[0]

        # 6. Predict probability
        proba = model.predict_proba(scaled_features)[0]
        malicious_probability = 0.0
        if hasattr(model, 'classes_'):
            classes_list = list(model.classes_)
            if 'malicious' in classes_list:
                malicious_idx = classes_list.index('malicious')
                malicious_probability = proba[malicious_idx]
            else:
                logger.warning("'malicious' class not found in model.")
        else:
             logger.warning("Cannot determine model classes.")

        return predicted_class, malicious_probability

    except Exception as e:
        logger.error(f"Error in _run_classification_prediction: {e}")
        # Avoid propagating error, return None
        return None, None 