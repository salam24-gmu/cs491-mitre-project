"""
Risk Profiling Service

This service implements the user risk profiling pipeline to analyze tweet data
and detect potential insider threats based on NER model outputs.
"""

import os
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)

# ============== 1. Entity Extraction and Standardization ==============

def extract_fallback_entities(text):
    """
    Basic fallback entity extraction for cases where no entities were detected.
    Uses simple pattern matching to identify potential entities.
    """
    fallback_entities = []

    # Simple keyword-based detection for common entity types
    tech_patterns = ["server", "database", "system", "network", "access", "login", "password",
                    "file", "document", "data", "computer", "laptop"]

    sensitive_patterns = ["confidential", "private", "secret", "personal", "sensitive",
                        "restricted", "classified", "patient", "record", "financial"]

    suspicious_patterns = ["delete", "hide", "bypass", "override", "unauthorized", "after hours",
                         "no one will know", "don't tell", "between us", "off the record"]

    time_patterns = ["late", "night", "weekend", "early", "after hours", "before opening",
                   "midnight", "sunday", "saturday", "holiday"]

    # Check for tech assets
    for pattern in tech_patterns:
        if pattern in text.lower():
            fallback_entities.append({
                "entity_type": "TECH_ASSET",
                "entity_text": pattern,
                "confidence": 0.75
            })

    # Check for sensitive info
    for pattern in sensitive_patterns:
        if pattern in text.lower():
            fallback_entities.append({
                "entity_type": "SENSITIVE_INFO",
                "entity_text": pattern,
                "confidence": 0.8
            })

    # Check for suspicious behavior
    for pattern in suspicious_patterns:
        if pattern in text.lower():
            fallback_entities.append({
                "entity_type": "SUSPICIOUS_BEHAVIOR",
                "entity_text": pattern,
                "confidence": 0.85
            })

    # Check for time anomalies
    for pattern in time_patterns:
        if pattern in text.lower():
            fallback_entities.append({
                "entity_type": "TIME_ANOMALY",
                "entity_text": pattern,
                "confidence": 0.8
            })

    return fallback_entities


def extract_and_standardize_entities(data, confidence_threshold=0.65):
    """
    Unifies diverse entity detection formats into a standardized representation,
    critical for consistent downstream analysis.
    """
    processed_tweets = []

    for i, tweet_data in enumerate(data):
        tweet_id = tweet_data.get("tweet_id", f"tweet_{i}")
        user_id = tweet_data.get("user_id", "unknown_user")
        text = tweet_data.get("tweet", "")
        timestamp = tweet_data.get("timestamp", None)
        entities_data = tweet_data.get("entities", [])
        std_entities = []

        # Handle dictionary-based NER model outputs (common in spaCy, HuggingFace, etc.)
        if isinstance(entities_data, list) and entities_data and isinstance(entities_data[0], tuple):
            for entity_text, entity_type in entities_data:
                confidence = 0.9  # Assume high confidence for rule-based entities
                std_word = entity_text.lower().strip()
                std_entities.append({
                    "entity_type": entity_type,
                    "entity_text": std_word,
                    "confidence": confidence
                })
        # Handle tuple-based outputs from rule engines (common in older systems)
        else:
            for entity in entities_data:
                if isinstance(entity, dict):
                    entity_type = entity.get("entity_group", entity.get("label", ""))
                    confidence = entity.get("score", 0.0)
                    word = entity.get("word", "")

                    if confidence < confidence_threshold:
                        continue

                    std_word = word.lower().strip()
                    std_entities.append({
                        "entity_type": entity_type,
                        "entity_text": std_word,
                        "confidence": confidence
                    })

        # Add validation for empty entity lists
        if not std_entities:
            # Apply basic entity extraction for empty entities
            std_entities = extract_fallback_entities(text)

        metadata = {
            "classification": tweet_data.get("classification", None),
            "sentiment": tweet_data.get("sentiment", None),
            "off_hours": tweet_data.get("off_hours", None)
        }

        grouped_entities = group_similar_entities(std_entities)

        processed_tweet = {
            "tweet_id": tweet_id,
            "user_id": user_id,
            "text": text,
            "timestamp": timestamp,
            "entities": grouped_entities,
            "metadata": metadata
        }

        processed_tweets.append(processed_tweet)

    return processed_tweets


def group_similar_entities(entities):
    """
    Consolidates semantically related entity mentions to reduce redundancy.
    """
    sorted_entities = sorted(entities, key=lambda x: x["confidence"], reverse=True)
    entity_groups = {}

    for entity in sorted_entities:
        entity_type = entity["entity_type"]
        entity_text = entity["entity_text"]

        is_substring = False
        for existing_text in list(entity_groups.get(entity_type, {}).keys()):
            if entity_text in existing_text or existing_text in entity_text:
                if len(entity_text) > len(existing_text):
                    new_confidence = max(entity["confidence"],
                                        entity_groups[entity_type][existing_text]["confidence"])

                    entity_groups.setdefault(entity_type, {})[entity_text] = {
                        "entity_type": entity_type,
                        "entity_text": entity_text,
                        "confidence": new_confidence
                    }
                    del entity_groups[entity_type][existing_text]
                else:
                    entity_groups[entity_type][existing_text]["confidence"] = max(
                        entity["confidence"],
                        entity_groups[entity_type][existing_text]["confidence"]
                    )
                is_substring = True
                break

        if not is_substring:
            entity_groups.setdefault(entity_type, {})[entity_text] = entity

    grouped_entities = []
    for type_dict in entity_groups.values():
        grouped_entities.extend(list(type_dict.values()))

    return grouped_entities

# ============== 2. Risk-Weighted Entity Quantification ==============

def refined_entity_risk_weights():
    """
    Provides more granular and context-aware entity risk weights.
    """
    return {
        # Highest risk entities
        "SUSPICIOUS_BEHAVIOR": 0.95,
        "SENSITIVE_INFO": 0.90,

        # High risk entities
        "TIME_ANOMALY": 0.75,
        "TECH_ASSET": 0.70,

        # Medium risk entities
        "MEDICAL_CONDITION": 0.60,
        "SENTIMENT_INDICATOR": 0.55,

        # Base entities
        "PERSON": 0.35,
        "ORG": 0.40,
        "LOC": 0.20
    }

def assign_risk_weights(standardized_tweets, entity_risk_weights=None):
    """
    Applies domain-specific risk weights to entities based on their type.
    """
    if entity_risk_weights is None:
        entity_risk_weights = refined_entity_risk_weights()

    risk_weighted_tweets = []

    for tweet in standardized_tweets:
        weighted_tweet = tweet.copy()
        weighted_entities = []
        has_precalculated_risk = tweet.get("metadata", {}).get("risk_score") is not None

        for entity in tweet["entities"]:
            weighted_entity = entity.copy()
            entity_type = entity["entity_type"]
            risk_weight = entity_risk_weights.get(entity_type, 0.1)
            entity_risk = entity["confidence"] * risk_weight
            weighted_entity["risk_score"] = entity_risk
            weighted_entity["risk_weight"] = risk_weight
            weighted_entities.append(weighted_entity)

        weighted_tweet["entities"] = weighted_entities

        if has_precalculated_risk:
            weighted_tweet["precalculated_risk"] = weighted_tweet["metadata"]["risk_score"]

        risk_weighted_tweets.append(weighted_tweet)

    return risk_weighted_tweets

# ============== 3. Tweet-Level Risk Assessment ==============
def context_aware_risk_assessment(text):
    """
    Analyzes text context for risk signals beyond entity detection.
    """
    context_signals = []
    risk_score = 0.0

    # Check for patterns indicating security context
    security_phrases = ["security", "access", "clearance", "permission", "classified",
                       "authorization", "authorized", "confidential", "restricted"]

    # Check for patterns indicating secrecy or covert behavior
    secrecy_phrases = ["secret", "private", "hide", "delete", "erase", "remove",
                      "nobody", "no one", "between us", "don't tell", "keep this quiet"]

    # Check for financial motivation
    financial_phrases = ["money", "payment", "cash", "pay", "debt", "financial",
                        "opportunity", "offer", "compensation"]

    # Count occurrences of each category
    security_count = sum(1 for phrase in security_phrases if phrase in text.lower())
    secrecy_count = sum(1 for phrase in secrecy_phrases if phrase in text.lower())
    financial_count = sum(1 for phrase in financial_phrases if phrase in text.lower())

    # Add signals based on occurrence
    if security_count > 0:
        context_signals.append("SECURITY_CONTEXT")
        risk_score += 0.1 * min(security_count, 3)  # Cap at 0.3

    if secrecy_count > 0:
        context_signals.append("SECRECY_INDICATORS")
        risk_score += 0.15 * min(secrecy_count, 3)  # Cap at 0.45

    if financial_count > 0:
        context_signals.append("FINANCIAL_MOTIVATION")
        risk_score += 0.1 * min(financial_count, 3)  # Cap at 0.3

    # Check for combination of security context and secrecy
    if security_count > 0 and secrecy_count > 0:
        context_signals.append("SECURITY_WITH_SECRECY")
        risk_score += 0.2  # Additional risk for this combination

    return risk_score, context_signals

def semantic_risk_analysis(text):
    """
    Performs semantic analysis of text to identify risk indicators.
    """
    indicators = []
    risk_score = 0.0

    # Check for negative sentiment words
    negative_words = ["angry", "frustrated", "disappointed", "upset", "hate",
                     "stupid", "useless", "terrible", "awful", "annoyed"]

    # Check for threatening language
    threatening_words = ["threat", "revenge", "get back", "pay for this",
                        "they'll see", "they will regret", "make them sorry"]

    # Check for personal pronouns with negative context (indication of personal grievance)
    personal_negative = ["I hate", "I quit", "I'm leaving", "they fired",
                        "my boss", "they don't appreciate", "I deserve"]

    # Count occurrences
    negative_count = sum(1 for word in negative_words if word in text.lower())
    threatening_count = sum(1 for word in threatening_words if word in text.lower())
    personal_count = sum(1 for phrase in personal_negative if phrase.lower() in text.lower())

    # Add indicators and score based on occurrences
    if negative_count > 0:
        indicators.append("NEGATIVE_SENTIMENT")
        risk_score += 0.05 * min(negative_count, 4)  # Cap at 0.2

    if threatening_count > 0:
        indicators.append("THREATENING_LANGUAGE")
        risk_score += 0.15 * min(threatening_count, 3)  # Cap at 0.45

    if personal_count > 0:
        indicators.append("PERSONAL_GRIEVANCE")
        risk_score += 0.1 * min(personal_count, 3)  # Cap at 0.3

    # Check for combinations
    if negative_count > 0 and personal_count > 0:
        indicators.append("PERSONAL_NEGATIVITY")
        risk_score += 0.1  # Additional risk for this combination

    if threatening_count > 0 and personal_count > 0:
        indicators.append("PERSONAL_THREAT")
        risk_score += 0.25  # Significant risk increase for this combination

    return risk_score, indicators

def improve_risk_combo_detection():
    """
    Enhanced risk combination detection with expanded combinations and dynamic thresholds.
    """
    return {
        # Original combinations with adjusted multipliers
        ("SUSPICIOUS_BEHAVIOR", "SENSITIVE_INFO"): 1.8,
        ("TIME_ANOMALY", "TECH_ASSET"): 1.5,
        ("SUSPICIOUS_BEHAVIOR", "SUSPICIOUS_BEHAVIOR"): 1.4,

        # New combinations that signal potential data exfiltration
        ("SENSITIVE_INFO", "TECH_ASSET"): 1.6,
        ("SUSPICIOUS_BEHAVIOR", "ORG"): 1.3,
        ("SENSITIVE_INFO", "TIME_ANOMALY"): 1.7,

        # Additional context-aware combinations
        ("PERSON", "SENSITIVE_INFO"): 1.3,
        ("SENTIMENT_INDICATOR", "SUSPICIOUS_BEHAVIOR"): 1.2,
        ("TIME_ANOMALY", "SUSPICIOUS_BEHAVIOR"): 1.4
    }

def enhanced_time_analysis(tweet_timestamp, tweet_text=None, time_anomaly_present=False):
    """
    Improved time analysis that considers both timestamp and content.
    """
    # First check if there's explicit time anomaly detection
    if time_anomaly_present:
        return True

    # Check the timestamp
    if tweet_timestamp is None:
        return False

    try:
        dt = datetime.strptime(tweet_timestamp, '%Y-%m-%d %H:%M:%S')

        # Weekend check
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return True

        # Expanded definition (before 8AM or after 5PM)
        if dt.hour < 8 or dt.hour >= 17:
            return True

        return False
    except Exception:
        # Try alternative formats
        try:
            # Try alternative formats
            alternative_formats = ['%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S']
            for fmt in alternative_formats:
                try:
                    dt = datetime.strptime(tweet_timestamp, fmt)
                    # Same checks as above
                    return dt.weekday() >= 5 or dt.hour < 8 or dt.hour >= 17
                except:
                    continue
            return False
        except:
            return False

def assess_tweet_risk(risk_weighted_tweets, risk_multipliers=None, use_precalculated=False):
    """
    Calculates multiple risk dimensions for a single tweet.
    """
    if risk_multipliers is None:
        risk_multipliers = improve_risk_combo_detection()

    assessed_tweets = []

    for tweet in risk_weighted_tweets:
        assessed_tweet = tweet.copy()
        entities = tweet["entities"]
        tweet_text = tweet.get("text", "")
        tweet_timestamp = tweet.get("timestamp")

        # Check if TIME_ANOMALY entity is present
        time_anomaly_present = any(entity["entity_type"] == "TIME_ANOMALY" for entity in entities)

        # Check for off-hours activity with explicit time anomaly detection
        is_off_hours = enhanced_time_analysis(tweet_timestamp, tweet_text, time_anomaly_present)

        context_risk, context_signals = context_aware_risk_assessment(tweet_text)
        semantic_risk, semantic_indicators = semantic_risk_analysis(tweet_text)

        if not entities:
            # Handle empty entities case
            assessed_tweet["risk_metrics"] = {
                "total_risk": (context_risk + semantic_risk) * 0.5,
                "base_risk": 0.0,
                "context_risk": context_risk,
                "semantic_risk": semantic_risk,
                "entity_type_risks": {},
                "high_risk_combinations": 0,
                "risk_density": 0.0,
                "is_precalculated": False,
                "context_signals": context_signals,
                "semantic_indicators": semantic_indicators,
                "off_hours": is_off_hours
            }
            assessed_tweets.append(assessed_tweet)
            continue

        base_risk = sum(entity["risk_score"] for entity in entities)
        entity_type_risks = {}
        entity_types_present = set()
        entity_type_counts = {}

        # Count entity types
        for entity in entities:
            entity_type = entity["entity_type"]
            entity_types_present.add(entity_type)
            entity_type_risks[entity_type] = entity_type_risks.get(entity_type, 0) + entity["risk_score"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # Detect high-risk combinations
        high_risk_combinations = 0
        risk_multiplier = 1.0
        high_risk_combo_details = []

        # Check for combinations where both types exist
        for combo, multiplier in risk_multipliers.items():
            type1, type2 = combo

            if type1 == type2:
                # Same type combination (e.g. multiple SUSPICIOUS_BEHAVIOR)
                if type1 in entity_type_counts and entity_type_counts[type1] > 1:
                    additional_entities = entity_type_counts[type1] - 1
                    high_risk_combinations += additional_entities
                    risk_multiplier *= multiplier ** additional_entities
                    high_risk_combo_details.append(f"Multiple {type1}: {entity_type_counts[type1]}")
            else:
                # Different type combination
                if type1 in entity_types_present and type2 in entity_types_present:
                    # Count this as one combination regardless of how many of each type
                    high_risk_combinations += 1
                    risk_multiplier *= multiplier
                    high_risk_combo_details.append(f"{type1} + {type2}")

        # Apply off-hours multiplication if detected
        if is_off_hours:
            risk_multiplier *= 1.2  # 20% increase for off-hours activity

        # Integrate contextual and semantic risk
        adjusted_base_risk = base_risk + (context_risk * 0.7) + (semantic_risk * 0.7)

        final_risk = adjusted_base_risk * risk_multiplier
        risk_density = final_risk / max(len(entities), 1)

        risk_metrics = {
            "base_risk": base_risk,
            "adjusted_base_risk": adjusted_base_risk,
            "context_risk": context_risk,
            "semantic_risk": semantic_risk,
            "risk_multiplier": risk_multiplier,
            "total_risk": final_risk,
            "entity_type_risks": entity_type_risks,
            "high_risk_combinations": high_risk_combinations,
            "high_risk_combo_details": high_risk_combo_details,
            "entity_type_counts": entity_type_counts,
            "risk_density": risk_density,
            "is_precalculated": False,
            "context_signals": context_signals,
            "semantic_indicators": semantic_indicators,
            "off_hours": is_off_hours
        }

        # Add metadata if available
        if "metadata" in tweet and tweet["metadata"].get("classification"):
            risk_metrics["classification"] = tweet["metadata"]["classification"]

        if "metadata" in tweet and tweet["metadata"].get("sentiment"):
            risk_metrics["sentiment"] = tweet["metadata"]["sentiment"]

        assessed_tweet["risk_metrics"] = risk_metrics
        assessed_tweets.append(assessed_tweet)

    return assessed_tweets

# ============== Main Pipeline Function for API Integration ==============

async def process_text_for_risk_analysis(text, user_id=None, tweet_id=None, timestamp=None, entities=None):
    """
    Process a single text for risk analysis using the NER model output.
    
    Args:
        text (str): The text to analyze
        user_id (str, optional): User identifier
        tweet_id (str, optional): Tweet/message identifier
        timestamp (str, optional): Timestamp of the message
        entities (list, optional): Pre-detected entities (if available)
        
    Returns:
        dict: Risk assessment results
    """
    # If entities are not provided, they should be detected by the NER model
    # in the API route before calling this function
    
    # Prepare the data structure expected by the risk pipeline
    tweet_data = {
        "tweet_id": tweet_id or f"tweet_{hash(text) % 10000}",
        "user_id": user_id or "anonymous_user",
        "tweet": text,
        "timestamp": timestamp,
        "entities": entities or [],
    }
    
    # Run the pipeline
    standardized_tweets = extract_and_standardize_entities([tweet_data])
    risk_weighted_tweets = assign_risk_weights(standardized_tweets)
    assessed_tweets = assess_tweet_risk(risk_weighted_tweets)
    
    # Return the assessment results
    if assessed_tweets:
        return assessed_tweets[0]
    else:
        return {"error": "Failed to process text"}

async def process_texts_batch(batch_data):
    """
    Process a batch of texts for risk analysis.
    
    Args:
        batch_data (list): List of dictionaries with text and other metadata
        
    Returns:
        list: List of risk assessment results
    """
    # Map the batch data to the format expected by the pipeline
    tweet_data_batch = []
    
    for i, item in enumerate(batch_data):
        tweet_data = {
            "tweet_id": item.get("tweet_id", f"tweet_{i}"),
            "user_id": item.get("user_id", f"user_{hash(item.get('text', '')) % 10000}"),
            "tweet": item.get("text", ""),
            "timestamp": item.get("timestamp"),
            "entities": item.get("entities", []),
        }
        tweet_data_batch.append(tweet_data)
    
    # Process the batch through the pipeline
    loop = asyncio.get_running_loop()
    
    # Run each step in the thread pool
    standardized_tweets = await loop.run_in_executor(
        thread_pool, 
        extract_and_standardize_entities, 
        tweet_data_batch
    )
    
    risk_weighted_tweets = await loop.run_in_executor(
        thread_pool,
        assign_risk_weights,
        standardized_tweets
    )
    
    assessed_tweets = await loop.run_in_executor(
        thread_pool,
        assess_tweet_risk,
        risk_weighted_tweets
    )
    
    return assessed_tweets 