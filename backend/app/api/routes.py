from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import os

from app.services.model_service import run_inference, evaluate_entities
from app.services.integration_service import (
    analyze_text_with_risk_profile,
    analyze_texts_batch_with_risk_profile,
    get_user_risk_profile
)
from app.core.config import settings

# Create router
api_router = APIRouter()

# ----- Pydantic Models for Request/Response -----

class ThreatDetectionRequest(BaseModel):
    """Request model for threat detection"""
    text: str
    timestamp: Optional[str] = None
    user_id: Optional[str] = None
    tweet_id: Optional[str] = None

class BatchThreatDetectionRequest(BaseModel):
    """Request model for batch threat detection"""
    texts: List[Dict[str, Any]] = Field(..., 
        description="List of text objects with 'text' and optional 'timestamp', 'user_id', 'tweet_id' fields",
        example=[{
            "text": "I'm going to download customer data tonight", 
            "timestamp": "2023-04-15T22:10:00Z",
            "user_id": "user_123",
            "tweet_id": "tweet_456"
        }])

class UserRiskProfileRequest(BaseModel):
    """Request model for user risk profile"""
    user_id: str
    texts: List[Dict[str, Any]] = Field(...,
        description="List of text objects for this user",
        example=[{
            "text": "I'm going to download customer data tonight", 
            "timestamp": "2023-04-15T22:10:00Z",
            "tweet_id": "tweet_456"
        }])

class Entity(BaseModel):
    """Entity detected by the NER pipeline"""
    entity_group: str
    score: float
    word: str
    start: int
    end: int

class ThreatDetectionResponse(BaseModel):
    """Response model for threat detection"""
    entities: List[Entity]
    analysis_result: str
    processing_time_ms: float

class BatchThreatDetectionResponse(BaseModel):
    """Response model for batch threat detection"""
    results: List[ThreatDetectionResponse]
    total_processing_time_ms: float

class SystemInfoResponse(BaseModel):
    """Response model for system info"""
    version: str
    system_info: Dict[str, Any]
    model_path: str
    status: str
    model_exists: bool

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_id: str
    model_path: str
    exists: bool
    files: Optional[List[str]] = None
    size_mb: Optional[float] = None

# ----- Routes -----

@api_router.get("/", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information and API status"""
    model_path = settings.MODEL_PATH
    model_exists = os.path.exists(model_path)
    
    return SystemInfoResponse(
        version=settings.VERSION,
        system_info=settings.SYSTEM_INFO,
        model_path=model_path,
        status="running",
        model_exists=model_exists
    )

@api_router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed information about the loaded model"""
    model_path = settings.MODEL_PATH
    model_exists = os.path.exists(model_path)
    
    response = ModelInfoResponse(
        model_id=settings.MODEL_ID,
        model_path=model_path,
        exists=model_exists,
        files=None,
        size_mb=None
    )
    
    if model_exists:
        try:
            # Get list of files in model directory
            files = os.listdir(model_path)
            response.files = files
            
            # Calculate total size of model files
            total_size = 0
            for file in files:
                file_path = os.path.join(model_path, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            
            response.size_mb = total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            # If we can't access the files, just return what we have
            pass
    
    return response

@api_router.post("/detect-threats", response_model=ThreatDetectionResponse)
async def detect_threats(request: ThreatDetectionRequest):
    """
    Detect threats and entities in provided text
    
    Uses the NER model to identify potentially suspicious entities and 
    combinations that could indicate insider threats.
    """
    try:
        start_time = time.time()
        
        # Run inference on the text
        entities = await run_inference(request.text)
        
        # Generate analysis based on the detected entities
        analysis = evaluate_entities(
            entities=entities,
            text=request.text,
            tweet_timestamp=request.timestamp
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return ThreatDetectionResponse(
            entities=entities,
            analysis_result=analysis,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during threat detection: {str(e)}"
        )

@api_router.post("/batch-detect-threats", response_model=BatchThreatDetectionResponse)
async def batch_detect_threats(request: BatchThreatDetectionRequest):
    """
    Process multiple texts for threat detection in a single request
    
    This endpoint is useful for analyzing multiple messages or documents at once.
    """
    try:
        start_time = time.time()
        results = []
        
        for item in request.texts:
            item_start_time = time.time()
            
            # Extract text and timestamp from the item
            text = item.get("text", "")
            timestamp = item.get("timestamp")
            
            if not text:
                # Skip empty texts
                continue
            
            try:
                # Run inference on this text
                entities = await run_inference(text)
                
                # Generate analysis
                analysis = evaluate_entities(
                    entities=entities,
                    text=text,
                    tweet_timestamp=timestamp
                )
                
                item_processing_time = (time.time() - item_start_time) * 1000
                
                # Add to results
                results.append(ThreatDetectionResponse(
                    entities=entities,
                    analysis_result=analysis,
                    processing_time_ms=item_processing_time
                ))
            except Exception as item_error:
                # If one item fails, log it but continue processing others
                results.append(ThreatDetectionResponse(
                    entities=[],
                    analysis_result=f"Error processing this text: {str(item_error)}",
                    processing_time_ms=(time.time() - item_start_time) * 1000
                ))
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchThreatDetectionResponse(
            results=results,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch threat detection: {str(e)}"
        )

# ----- New Risk Profiling Routes -----

@api_router.post("/analyze-risk", response_model=Dict[str, Any])
async def analyze_risk(request: ThreatDetectionRequest):
    """
    Analyze text with comprehensive risk profiling
    
    This endpoint integrates NER detection with risk profiling to provide
    a detailed risk assessment for the input text.
    """
    try:
        start_time = time.time()
        
        # Use the integrated analysis service
        analysis_result = await analyze_text_with_risk_profile(
            text=request.text,
            user_id=request.user_id,
            tweet_id=request.tweet_id,
            timestamp=request.timestamp
        )
        
        if "error" in analysis_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=analysis_result["error"]
            )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        analysis_result["processing_time_ms"] = processing_time
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during risk analysis: {str(e)}"
        )

@api_router.post("/batch-analyze-risk", response_model=Dict[str, Any])
async def batch_analyze_risk(request: BatchThreatDetectionRequest):
    """
    Process multiple texts with risk profiling in a single request
    
    This endpoint analyzes multiple texts and produces individual assessments
    as well as aggregated user profiles for risk evaluation.
    """
    try:
        start_time = time.time()
        
        # Use the integrated batch analysis service
        analysis_result = await analyze_texts_batch_with_risk_profile(
            batch_data=request.texts
        )
        
        if "error" in analysis_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=analysis_result["error"]
            )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        analysis_result["processing_time_ms"] = processing_time
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch risk analysis: {str(e)}"
        )

@api_router.post("/user-risk-profile", response_model=Dict[str, Any])
async def user_risk_profile(request: UserRiskProfileRequest):
    """
    Generate a comprehensive risk profile for a specific user
    
    This endpoint analyzes all provided texts for a user and produces
    a detailed risk profile with multiple risk dimensions.
    """
    try:
        start_time = time.time()
        
        # Use the user profile service
        profile_result = await get_user_risk_profile(
            user_id=request.user_id,
            texts=request.texts
        )
        
        if "error" in profile_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=profile_result["error"]
            )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        profile_result["processing_time_ms"] = processing_time
        
        return profile_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating user risk profile: {str(e)}"
        ) 