"""
Integration Service

This service integrates the NER model and risk profiling pipeline to provide
a unified interface for text analysis and risk assessment.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.services.model_service import run_inference
from app.services.risk_profiling_service import process_text_for_risk_analysis, process_texts_batch
from app.services.profile_service import create_user_profile, get_user_summary, get_all_users_summary
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

async def analyze_text_with_risk_profile(text: str, user_id: Optional[str] = None, 
                                        tweet_id: Optional[str] = None, 
                                        timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Integrates NER model output with risk profiling to analyze text.
    
    Args:
        text (str): The text to analyze
        user_id (str, optional): User identifier
        tweet_id (str, optional): Tweet/message identifier
        timestamp (str, optional): Timestamp of the message
        
    Returns:
        dict: Combined analysis results
    """
    try:
        # Step 1: Run NER model to detect entities
        logger.info(f"Running NER inference for text: {text[:50]}...")
        entities = await run_inference(text)
        
        if entities is None:
            return {
                "error": "NER model failed to detect entities",
                "text": text,
                "user_id": user_id,
                "tweet_id": tweet_id
            }
        
        # Step 2: Process text for risk analysis using the entities
        logger.info(f"Processing text for risk analysis...")
        risk_analysis = await process_text_for_risk_analysis(
            text=text,
            user_id=user_id,
            tweet_id=tweet_id,
            timestamp=timestamp,
            entities=entities
        )
        
        # Step 3: Prepare the response
        response = {
            "text": text,
            "user_id": user_id or risk_analysis.get("user_id", "anonymous_user"),
            "tweet_id": tweet_id or risk_analysis.get("tweet_id", f"tweet_{hash(text) % 10000}"),
            "timestamp": timestamp,
            "entities": entities,
            "risk_metrics": risk_analysis.get("risk_metrics", {}),
        }
        
        logger.info(f"Analysis completed with {len(entities)} entities detected")
        return response
    
    except Exception as e:
        logger.error(f"Error in text analysis pipeline: {str(e)}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "text": text,
            "user_id": user_id,
            "tweet_id": tweet_id
        }

async def analyze_texts_batch_with_risk_profile(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes a batch of texts with NER and risk profiling.
    
    Args:
        batch_data (list): List of dictionaries with texts and metadata
        
    Returns:
        dict: Analysis results for all texts and aggregated user profiles
    """
    try:
        logger.info(f"Processing batch of {len(batch_data)} texts...")
        processed_items = []
        
        # Step 1: Process each text with NER model first
        for item in batch_data:
            text = item.get("text", "")
            user_id = item.get("user_id")
            tweet_id = item.get("tweet_id")
            timestamp = item.get("timestamp")
            
            try:
                # Run NER model
                entities = await run_inference(text)
                
                # Add entities to the item
                item["entities"] = entities or []
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {str(e)}")
                # Add a placeholder with error information
                processed_items.append({
                    "text": text,
                    "user_id": user_id,
                    "tweet_id": tweet_id,
                    "timestamp": timestamp,
                    "error": f"NER analysis failed: {str(e)}",
                    "entities": []
                })
        
        # Step 2: Process the batch through risk analysis
        logger.info(f"Running batch risk analysis on {len(processed_items)} texts...")
        assessed_texts = await process_texts_batch(processed_items)
        
        # Step 3: Generate user profiles and summaries
        logger.info("Generating user profiles...")
        user_profiles_data = await create_user_profile(assessed_texts)
        user_summaries = await get_all_users_summary(assessed_texts)
        
        # Step 4: Prepare response
        response = {
            "assessed_texts": assessed_texts,
            "user_summaries": user_summaries.get("users", []),
            "processed_count": len(assessed_texts),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Batch processing completed with {len(user_summaries.get('users', []))} user profiles")
        return response
    
    except Exception as e:
        logger.error(f"Error in batch analysis pipeline: {str(e)}")
        return {
            "error": f"Batch analysis failed: {str(e)}",
            "processed_count": 0,
            "assessed_texts": [],
            "user_summaries": []
        }

async def get_user_risk_profile(user_id: str, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gets a comprehensive risk profile for a specific user.
    
    Args:
        user_id (str): User identifier
        texts (list): List of texts associated with this user
        
    Returns:
        dict: User risk profile
    """
    try:
        logger.info(f"Generating risk profile for user {user_id}...")
        
        # Process each text for this user
        processed_items = []
        
        for item in texts:
            text = item.get("text", "")
            tweet_id = item.get("tweet_id")
            timestamp = item.get("timestamp")
            
            # Always set the user_id to ensure consistency
            item["user_id"] = user_id
            
            try:
                # Run NER model
                entities = await run_inference(text)
                
                # Add entities to the item
                item["entities"] = entities or []
                processed_items.append(item)
                
            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {str(e)}")
                processed_items.append({
                    "text": text,
                    "user_id": user_id,
                    "tweet_id": tweet_id,
                    "timestamp": timestamp,
                    "error": f"NER analysis failed: {str(e)}",
                    "entities": []
                })
        
        # Process through risk analysis
        assessed_texts = await process_texts_batch(processed_items)
        
        # Generate a summary for this user
        user_summary = await get_user_summary(user_id, assessed_texts)
        
        if "error" in user_summary:
            return user_summary
        
        # Prepare detailed response
        response = {
            "user_id": user_id,
            "profile": user_summary,
            "assessed_texts": assessed_texts,
            "text_count": len(assessed_texts),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Risk profile generated for user {user_id} with {len(assessed_texts)} texts")
        return response
    
    except Exception as e:
        logger.error(f"Error generating user risk profile: {str(e)}")
        return {
            "error": f"User risk profile generation failed: {str(e)}",
            "user_id": user_id
        } 