"""
Profile Service

This service implements the user-level risk profiling pipeline to aggregate 
tweet-level risk assessments into comprehensive user profiles.
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

# ============== 4. User-Level Profile Aggregation ==============

def detect_behavior_changes(user_tweets):
    """
    Detects temporal patterns in user behavior by analyzing risk metrics over time.
    
    Args:
        user_tweets (list): List of tweets from a single user, with risk metrics

    Returns:
        tuple: (temporal_risk_score, anomalies) with detected behavioral anomalies
    """
    if len(user_tweets) < 3:
        return 0.0, []

    # Sort tweets by timestamp if available
    sorted_tweets = sorted(user_tweets, key=lambda t: t.get("timestamp", ""), reverse=False)

    risk_scores = [t.get("risk_metrics", {}).get("total_risk", 0) for t in sorted_tweets]
    anomalies = []
    temporal_risk = 0.0

    # Detect risk spikes
    for i in range(1, len(risk_scores)):
        if i > 0 and risk_scores[i] > 2 * risk_scores[i-1] and risk_scores[i] > 0.5:
            anomalies.append(f"RISK_SPIKE_AT_{i}")
            temporal_risk += 0.2

    # Detect increasing risk trends
    if len(risk_scores) >= 3:
        first_third = sum(risk_scores[:len(risk_scores)//3]) / (len(risk_scores)//3)
        last_third = sum(risk_scores[-len(risk_scores)//3:]) / (len(risk_scores)//3)

        if last_third > 1.5 * first_third and last_third > 0.4:
            anomalies.append("INCREASING_RISK_TREND")
            temporal_risk += 0.3

    # Check for off-hours pattern
    off_hours_count = [t.get("risk_metrics", {}).get("off_hours", False) for t in sorted_tweets].count(True)
    if off_hours_count >= len(sorted_tweets) * 0.7 and len(sorted_tweets) >= 3:
        anomalies.append("PREDOMINANTLY_OFF_HOURS")
        temporal_risk += 0.25

    # Check for new high-risk entity types
    entity_types_by_tweet = []
    for tweet in sorted_tweets:
        risk_metrics = tweet.get("risk_metrics", {})
        entity_types = set(risk_metrics.get("entity_type_counts", {}).keys())
        entity_types_by_tweet.append(entity_types)

    high_risk_types = {"SUSPICIOUS_BEHAVIOR", "SENSITIVE_INFO", "TIME_ANOMALY"}

    # Check for sudden appearance of high-risk entity types
    for i in range(1, len(entity_types_by_tweet)):
        new_high_risk = high_risk_types.intersection(entity_types_by_tweet[i]) - \
                       high_risk_types.intersection(entity_types_by_tweet[i-1])
        if len(new_high_risk) >= 2:
            anomalies.append(f"NEW_HIGH_RISK_ENTITIES_AT_{i}")
            temporal_risk += 0.15

    return temporal_risk, anomalies

def aggregate_user_profiles(assessed_tweets):
    """
    Transforms tweet-level risk signals into holistic user-level profiles.
    
    Args:
        assessed_tweets (list): List of tweets with risk metrics
        
    Returns:
        dict: User profiles with aggregated risk metrics
    """
    # Group tweets by user
    tweets_by_user = defaultdict(list)
    for tweet in assessed_tweets:
        user_id = tweet["user_id"]
        tweets_by_user[user_id].append(tweet)

    # Analyze each user profile
    user_profiles = {}
    for user_id, user_tweets in tweets_by_user.items():
        # Detect behavior changes over time
        temporal_risk, anomalies = detect_behavior_changes(user_tweets)

        profile = {
            "tweets": user_tweets,
            "tweet_count": len(user_tweets),
            "total_risk": 0.0,
            "entity_counts": defaultdict(int),
            "entity_risks": defaultdict(float),
            "high_risk_combinations": 0,
            "high_risk_combo_details": [],
            "max_tweet_risk": 0.0,
            "classifications": defaultdict(int),
            "sentiments": defaultdict(int),
            "off_hours_count": 0,
            "temporal_risk": temporal_risk,
            "behavior_anomalies": anomalies,
            "context_signals": defaultdict(int),
            "semantic_indicators": defaultdict(int)
        }

        for tweet in user_tweets:
            risk_metrics = tweet.get("risk_metrics", {})

            profile["total_risk"] += risk_metrics.get("total_risk", 0)

            # Count high risk combinations
            high_risk_count = risk_metrics.get("high_risk_combinations", 0)
            if high_risk_count > 0:
                profile["high_risk_combinations"] += high_risk_count
                profile["high_risk_combo_details"].extend(risk_metrics.get("high_risk_combo_details", []))

            profile["max_tweet_risk"] = max(
                profile["max_tweet_risk"],
                risk_metrics.get("total_risk", 0)
            )

            # Track entity counts and risks
            for entity_type, count in risk_metrics.get("entity_type_counts", {}).items():
                profile["entity_counts"][entity_type] += count

            for entity_type, risk in risk_metrics.get("entity_type_risks", {}).items():
                profile["entity_risks"][entity_type] += risk

            # Track classifications and sentiments
            if "classification" in risk_metrics:
                classification = risk_metrics["classification"]
                profile["classifications"][classification] += 1

            if "sentiment" in risk_metrics:
                sentiment = risk_metrics["sentiment"]
                profile["sentiments"][sentiment] += 1

            # Count off-hours activity
            if risk_metrics.get("off_hours", False):
                profile["off_hours_count"] += 1

            # Aggregate context signals
            for signal in risk_metrics.get("context_signals", []):
                profile["context_signals"][signal] += 1

            # Aggregate semantic indicators
            for indicator in risk_metrics.get("semantic_indicators", []):
                profile["semantic_indicators"][indicator] += 1

        tweet_count = profile["tweet_count"]

        if tweet_count > 0:
            profile["avg_risk_per_tweet"] = profile["total_risk"] / tweet_count

            total_entities = sum(profile["entity_counts"].values())
            profile["total_entity_count"] = total_entities

            if total_entities > 0:
                profile["entity_density"] = total_entities / tweet_count
                profile["entity_distribution"] = {
                    entity_type: (count / total_entities) * 100
                    for entity_type, count in profile["entity_counts"].items()
                }
            else:
                profile["entity_density"] = 0
                profile["entity_distribution"] = {}

            profile["high_risk_combo_density"] = profile["high_risk_combinations"] / tweet_count

            # Track classification distributions
            total_classifications = sum(profile["classifications"].values())
            if total_classifications > 0:
                profile["classification_distribution"] = {
                    cls: (count / total_classifications) * 100
                    for cls, count in profile["classifications"].items()
                }
            else:
                profile["classification_distribution"] = {}

            # Track sentiment distributions
            total_sentiments = sum(profile["sentiments"].values())
            if total_sentiments > 0:
                profile["sentiment_distribution"] = {
                    sentiment: (count / total_sentiments) * 100
                    for sentiment, count in profile["sentiments"].items()
                }
            else:
                profile["sentiment_distribution"] = {}

            # Calculate off_hours_percentage
            profile["off_hours_percentage"] = (profile["off_hours_count"] / tweet_count) * 100

            # Calculate frequency of context signals and semantic indicators
            profile["context_signal_frequency"] = {
                signal: (count / tweet_count) * 100
                for signal, count in profile["context_signals"].items()
            }

            profile["semantic_indicator_frequency"] = {
                indicator: (count / tweet_count) * 100
                for indicator, count in profile["semantic_indicators"].items()
            }

            # Calculate risk statistics if we have enough data
            if tweet_count > 1:
                risk_values = [t.get("risk_metrics", {}).get("total_risk", 0) for t in profile["tweets"]]
                profile["risk_std_dev"] = np.std(risk_values)

                # Calculate risk trend (positive means increasing risk)
                if len(risk_values) >= 2:
                    first_half = sum(risk_values[:len(risk_values)//2]) / max(len(risk_values)//2, 1)
                    second_half = sum(risk_values[len(risk_values)//2:]) / max(len(risk_values) - len(risk_values)//2, 1)
                    profile["risk_trend"] = second_half - first_half
                else:
                    profile["risk_trend"] = 0.0
            else:
                profile["risk_std_dev"] = 0.0
                profile["risk_trend"] = 0.0
        else:
            # Default values for empty profiles
            profile["avg_risk_per_tweet"] = 0.0
            profile["total_entity_count"] = 0
            profile["entity_density"] = 0.0
            profile["entity_distribution"] = {}
            profile["high_risk_combo_density"] = 0.0
            profile["classification_distribution"] = {}
            profile["sentiment_distribution"] = {}
            profile["off_hours_percentage"] = 0.0
            profile["risk_std_dev"] = 0.0
            profile["risk_trend"] = 0.0
            profile["context_signal_frequency"] = {}
            profile["semantic_indicator_frequency"] = {}

        # Convert defaultdicts to regular dicts for serialization
        profile["entity_counts"] = dict(profile["entity_counts"])
        profile["entity_risks"] = dict(profile["entity_risks"])
        profile["classifications"] = dict(profile["classifications"])
        profile["sentiments"] = dict(profile["sentiments"])
        profile["context_signals"] = dict(profile["context_signals"])
        profile["semantic_indicators"] = dict(profile["semantic_indicators"])

        # Remove duplicate high risk combination details
        if profile["high_risk_combo_details"]:
            profile["high_risk_combo_details"] = list(set(profile["high_risk_combo_details"]))

        user_profiles[user_id] = profile

    return user_profiles

# ============== 5. Feature Vector Construction ==============

def construct_feature_vectors(user_profiles, entity_risk_weights=None):
    """
    Translates multi-dimensional user risk profiles into fixed-length numerical feature vectors.
    
    Args:
        user_profiles (dict): User profiles with aggregated risk metrics
        entity_risk_weights (dict, optional): Risk weights for entity types
        
    Returns:
        dict: Feature vectors for each user
    """
    if entity_risk_weights is None:
        entity_risk_weights = {
            "SUSPICIOUS_BEHAVIOR": 0.9,
            "SENSITIVE_INFO": 0.8,
            "TIME_ANOMALY": 0.7,
            "TECH_ASSET": 0.6,
            "MEDICAL_CONDITION": 0.4,
            "SENTIMENT_INDICATOR": 0.3,
            "PERSON": 0.2,
            "ORG": 0.2,
            "LOC": 0.1
        }

    feature_vectors = {}
    all_entity_types = set(entity_risk_weights.keys())
    all_classifications = set()
    all_sentiments = set()

    for profile in user_profiles.values():
        all_classifications.update(profile.get("classifications", {}).keys())
        all_sentiments.update(profile.get("sentiments", {}).keys())

    for user_id, profile in user_profiles.items():
        tweet_count = profile.get("tweet_count", len(profile.get("tweets", [])))

        if tweet_count == 0:
            continue

        features = {
            "total_risk_score": profile["total_risk"],
            "avg_risk_per_tweet": profile["avg_risk_per_tweet"],
            "max_tweet_risk": profile["max_tweet_risk"],
            "risk_std_dev": profile.get("risk_std_dev", 0.0),
            "entity_density": profile["entity_density"],
            "high_risk_combo_density": profile["high_risk_combo_density"],
            "off_hours_percentage": profile.get("off_hours_percentage", 0.0)
        }

        for entity_type in all_entity_types:
            count = profile["entity_counts"].get(entity_type, 0)
            features[f"{entity_type.lower()}_count"] = count
            features[f"{entity_type.lower()}_per_tweet"] = count / tweet_count

        for entity_type in all_entity_types:
            risk = profile["entity_risks"].get(entity_type, 0.0)
            features[f"{entity_type.lower()}_risk"] = risk
            features[f"{entity_type.lower()}_risk_per_tweet"] = risk / tweet_count if risk > 0 else 0

        for entity_type in all_entity_types:
            percentage = profile.get("entity_distribution", {}).get(entity_type, 0.0)
            features[f"{entity_type.lower()}_percentage"] = percentage

        for classification in all_classifications:
            count = profile.get("classifications", {}).get(classification, 0)
            percentage = profile.get("classification_distribution", {}).get(classification, 0.0)
            features[f"classification_{classification.lower()}_count"] = count
            features[f"classification_{classification.lower()}_percentage"] = percentage

        for sentiment in all_sentiments:
            count = profile.get("sentiments", {}).get(sentiment, 0)
            percentage = profile.get("sentiment_distribution", {}).get(sentiment, 0.0)
            features[f"sentiment_{sentiment.lower()}_count"] = count
            features[f"sentiment_{sentiment.lower()}_percentage"] = percentage

        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])

        feature_vectors[user_id] = {
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "feature_dict": features
        }

    return feature_vectors

# ============== 6. Normalization and Outlier Detection ==============

def normalize_features(feature_vectors, method="minmax"):
    """
    Applies statistical normalization techniques to harmonize feature scales.
    
    Args:
        feature_vectors (dict): Feature vectors for each user
        method (str): Normalization method ('minmax' or 'zscore')
        
    Returns:
        dict: Normalized feature vectors
    """
    normalized_vectors = {}
    user_ids = list(feature_vectors.keys())

    if not user_ids:
        return {}

    feature_names = feature_vectors[user_ids[0]]["feature_names"]
    all_vectors = np.vstack([feature_vectors[uid]["feature_vector"] for uid in user_ids])

    if method == "minmax":
        min_vals = np.min(all_vectors, axis=0)
        max_vals = np.max(all_vectors, axis=0)
        range_vals = np.maximum(max_vals - min_vals, 1e-10) # Prevents division by zero
        normalized = (all_vectors - min_vals) / range_vals
    elif method == "zscore":
        mean_vals = np.mean(all_vectors, axis=0)
        std_vals = np.std(all_vectors, axis=0)
        std_vals = np.maximum(std_vals, 1e-10)
        normalized = (all_vectors - mean_vals) / std_vals
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    for i, user_id in enumerate(user_ids):
        normalized_vectors[user_id] = {
            "feature_vector": normalized[i],
            "feature_names": feature_names,
            "feature_dict": {
                name: normalized[i, j] for j, name in enumerate(feature_names)
            }
        }

    return normalized_vectors

def detect_outliers(feature_vectors, threshold=2.0, method="zscore"):
    """
    Identifies statistically anomalous users through outlier detection.
    
    Args:
        feature_vectors (dict): Feature vectors for each user
        threshold (float): Threshold for outlier detection
        method (str): Outlier detection method ('zscore' or 'percentile')
        
    Returns:
        dict: Outlier scores for each user
    """
    if method == "zscore" and list(feature_vectors.values())[0]["feature_vector"].max() <= 1.0:
        normalized = normalize_features(feature_vectors, method="zscore")
    else:
        normalized = feature_vectors

    outlier_scores = {}

    for user_id, vector_data in normalized.items():
        feature_vector = vector_data["feature_vector"]

        if method == "zscore":
            # For each feature, find how many std devs away a user is 
            outlier_score = np.max(np.abs(feature_vector))
        elif method == "percentile":
            # Calculate what percentage of features are unusual
            outlier_score = np.sum(feature_vector > threshold) / len(feature_vector)
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        outlier_scores[user_id] = outlier_score

    return outlier_scores

# ============== API Integration Functions ==============

async def create_user_profile(assessed_tweets):
    """
    Creates a user profile from a list of assessed tweets.
    
    Args:
        assessed_tweets (list): List of tweets with risk metrics
        
    Returns:
        dict: User profiles with aggregated risk metrics
    """
    # Use thread pool for CPU-intensive operations
    loop = asyncio.get_running_loop()
    
    # Aggregate user profiles
    user_profiles = await loop.run_in_executor(
        thread_pool,
        aggregate_user_profiles,
        assessed_tweets
    )
    
    if not user_profiles:
        return {}
    
    # Construct feature vectors
    feature_vectors = await loop.run_in_executor(
        thread_pool,
        construct_feature_vectors, 
        user_profiles
    )
    
    # Normalize features
    normalized_vectors = await loop.run_in_executor(
        thread_pool,
        normalize_features,
        feature_vectors
    )
    
    # Detect outliers
    outlier_scores = await loop.run_in_executor(
        thread_pool,
        detect_outliers,
        normalized_vectors
    )
    
    # Merge outlier scores into profiles
    for user_id, score in outlier_scores.items():
        if user_id in user_profiles:
            user_profiles[user_id]["outlier_score"] = score
    
    return {
        "user_profiles": user_profiles,
        "feature_vectors": feature_vectors,
        "normalized_vectors": normalized_vectors,
        "outlier_scores": outlier_scores
    }

async def get_user_summary(user_id, assessed_tweets):
    """
    Gets a summary of a user's risk profile.
    
    Args:
        user_id (str): User identifier
        assessed_tweets (list): List of tweets with risk metrics
        
    Returns:
        dict: Summary of user risk profile
    """
    # Filter tweets for this user
    user_tweets = [t for t in assessed_tweets if t.get("user_id") == user_id]
    
    if not user_tweets:
        return {"error": "User not found or has no tweets"}
    
    # Create profiles for this user only
    profile_data = await create_user_profile(user_tweets)
    
    if not profile_data.get("user_profiles"):
        return {"error": "Failed to create user profile"}
    
    # Extract just this user's profile
    user_profile = profile_data["user_profiles"].get(user_id, {})
    
    # Create a summary with key metrics
    summary = {
        "user_id": user_id,
        "tweet_count": user_profile.get("tweet_count", 0),
        "total_risk": user_profile.get("total_risk", 0),
        "avg_risk_per_tweet": user_profile.get("avg_risk_per_tweet", 0),
        "max_tweet_risk": user_profile.get("max_tweet_risk", 0),
        "risk_std_dev": user_profile.get("risk_std_dev", 0),
        "entity_density": user_profile.get("entity_density", 0),
        "high_risk_combo_density": user_profile.get("high_risk_combo_density", 0),
        "off_hours_percentage": user_profile.get("off_hours_percentage", 0),
        "temporal_risk": user_profile.get("temporal_risk", 0),
        "outlier_score": user_profile.get("outlier_score", 0),
        "behavior_anomalies": user_profile.get("behavior_anomalies", []),
        "high_risk_combo_details": user_profile.get("high_risk_combo_details", []),
        "entity_counts": user_profile.get("entity_counts", {})
    }
    
    # Add risk assessment level
    total_risk = summary["total_risk"]
    outlier_score = summary["outlier_score"]
    
    if total_risk > 15 or outlier_score > 4:
        risk_level = "HIGH"
    elif total_risk > 8 or outlier_score > 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    summary["risk_level"] = risk_level
    
    return summary

async def get_all_users_summary(assessed_tweets):
    """
    Gets summaries for all users.
    
    Args:
        assessed_tweets (list): List of tweets with risk metrics
        
    Returns:
        dict: Summaries for all users
    """
    # Get unique user IDs
    user_ids = list(set(t.get("user_id") for t in assessed_tweets if t.get("user_id")))
    
    if not user_ids:
        return {"users": []}
    
    # Create profiles for all users
    profile_data = await create_user_profile(assessed_tweets)
    
    user_summaries = []
    
    for user_id in user_ids:
        # Extract just this user's profile
        user_profile = profile_data["user_profiles"].get(user_id, {})
        
        if not user_profile:
            continue
        
        # Create a summary with key metrics
        summary = {
            "user_id": user_id,
            "tweet_count": user_profile.get("tweet_count", 0),
            "total_risk": user_profile.get("total_risk", 0),
            "avg_risk_per_tweet": user_profile.get("avg_risk_per_tweet", 0),
            "max_tweet_risk": user_profile.get("max_tweet_risk", 0),
            "high_risk_combo_density": user_profile.get("high_risk_combo_density", 0),
            "outlier_score": profile_data["outlier_scores"].get(user_id, 0),
            "behavior_anomalies": user_profile.get("behavior_anomalies", []),
        }
        
        # Add risk assessment level
        total_risk = summary["total_risk"]
        outlier_score = summary["outlier_score"]
        
        if total_risk > 15 or outlier_score > 4:
            risk_level = "HIGH"
        elif total_risk > 8 or outlier_score > 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        summary["risk_level"] = risk_level
        
        user_summaries.append(summary)
    
    # Sort by total risk (descending)
    user_summaries.sort(key=lambda x: x["total_risk"], reverse=True)
    
    return {"users": user_summaries} 