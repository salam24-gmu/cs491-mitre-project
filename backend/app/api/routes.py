from fastapi import APIRouter, HTTPException, BackgroundTasks, status, UploadFile, File
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time
import os
import io
import pandas as pd
import tempfile
import numpy as np
import logging

from app.services.model_service import run_inference, evaluate_entities
from app.services.integration_service import (
    analyze_text_with_risk_profile,
    analyze_texts_batch_with_risk_profile,
    get_user_risk_profile
)
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()

# ----- Utility Functions -----

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types to ensure they can be serialized by Pydantic.
    
    Args:
        obj: Any object that might contain numpy values
        
    Returns:
        Object with numpy types converted to native Python types
    
    Examples:
        - np.float32(1.5) -> float(1.5)
        - np.int64(10) -> int(10)
        - {'score': np.float32(0.95)} -> {'score': 0.95}
    """
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    except Exception as e:
        logger.error(f"Error in convert_numpy_types: {str(e)}, for obj type: {type(obj)}")
        # Return the original object if conversion fails
        return obj

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

class SystemInfoResponse(BaseModel):
    """Response model for system info"""
    version: str
    system_info: Dict[str, Any]
    model_path: str # Path to NER model
    classification_model_path: str # Path to Classification model
    status: str
    models_exist: Dict[str, bool] # Indicate existence of both models

    model_config = {
        "protected_namespaces": ()
    }

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str # 'NER' or 'Classification'
    model_id: str
    model_path: str
    exists: bool
    files: Optional[List[str]] = None
    size_mb: Optional[float] = None

    model_config = {
        "protected_namespaces": ()
    }

class Entity(BaseModel):
    """Entity detected by the NER pipeline"""
    entity_group: str
    score: float
    word: str
    start: int
    end: int

class RiskAnalysisResponse(BaseModel):
    """Response model for single text risk analysis"""
    text: str
    user_id: Optional[str] = None
    tweet_id: Optional[str] = None
    timestamp: Optional[str] = None
    entities: List[Entity]
    risk_metrics: Dict[str, Any]
    predicted_class: Optional[str] = None
    malicious_probability: Optional[float] = None
    processing_time_ms: float

class AssessedText(BaseModel):
    """Represents a single assessed text within a batch"""
    text: str
    user_id: Optional[str] = None
    tweet_id: Optional[str] = None
    timestamp: Optional[str] = None
    entities: List[Entity]
    risk_metrics: Dict[str, Any]
    predicted_class: Optional[str] = None
    malicious_probability: Optional[float] = None
    error: Optional[str] = None

class UserSummary(BaseModel):
    """Summary of a user's risk profile"""
    user_id: str
    tweet_count: int
    total_risk: float
    avg_risk_per_tweet: float
    max_tweet_risk: float
    high_risk_combo_density: float
    outlier_score: float
    behavior_anomalies: List[str]
    risk_level: str
    predicted_malicious_count: Optional[int] = None
    predicted_malicious_percentage: Optional[float] = None
    avg_malicious_probability: Optional[float] = None

class BatchRiskAnalysisResponse(BaseModel):
    """Response model for batch text risk analysis"""
    assessed_texts: List[AssessedText]
    user_summaries: List[UserSummary]
    processed_count: int
    timestamp: str
    processing_time_ms: float

class UserProfileDetail(UserSummary):
    """Detailed user risk profile"""
    pass

class UserProfileResponse(BaseModel):
    """Response model for user risk profile endpoint"""
    user_id: str
    profile: UserProfileDetail
    assessed_texts: List[AssessedText]
    text_count: int
    timestamp: str
    processing_time_ms: float

class FileRowAnalysisResult(BaseModel):
    """Result model for a single row in a file analysis"""
    row_number: int
    text: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    tweet_id: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None
    predicted_class: Optional[str] = None
    malicious_probability: Optional[float] = None
    error: Optional[str] = None
    processing_time_ms: float

class FileUploadResponse(BaseModel):
    """Response model for file upload and analysis"""
    results: List[FileRowAnalysisResult]
    total_rows: int
    successful_rows: int
    failed_rows: int
    processing_time_ms: float
    file_type: str

# ----- Routes -----

@api_router.get("/", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information and API status"""
    ner_model_path = settings.MODEL_PATH
    classification_model_path = settings.CLASSIFICATION_MODEL_PATH
    ner_model_exists = os.path.exists(ner_model_path)
    classification_model_exists = os.path.exists(classification_model_path)

    return SystemInfoResponse(
        version=settings.VERSION,
        system_info=settings.SYSTEM_INFO,
        model_path=ner_model_path, # Keep original name for backward compatibility?
        classification_model_path=classification_model_path,
        status="running",
        # Indicate existence of both models
        models_exist={
            "ner_model": ner_model_exists,
            "classification_model": classification_model_exists
        }
    )

@api_router.get("/model-info/{model_type}", response_model=ModelInfoResponse)
async def get_model_info(model_type: str):
    """Get detailed information about the loaded NER or Classification model"""

    if model_type.lower() == 'ner':
        model_path = settings.MODEL_PATH
        model_id = settings.MODEL_ID
    elif model_type.lower() == 'classification':
        model_path = settings.CLASSIFICATION_MODEL_PATH
        model_id = settings.CLASSIFICATION_RUN_ID
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'ner' or 'classification'.")

    model_exists = os.path.exists(model_path)

    response = ModelInfoResponse(
        model_type=model_type.lower(),
        model_id=model_id,
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
            logger.warning(f"Could not list files or get size for {model_path}: {e}")
            # If we can't access the files, just return what we have
            pass

    return response

@api_router.post("/analyze-risk", response_model=RiskAnalysisResponse)
async def analyze_risk(request: ThreatDetectionRequest):
    """
    Analyze text with risk profiling
    
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
        
        # Convert any numpy types before returning
        return convert_numpy_types(analysis_result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during risk analysis: {str(e)}"
        )

@api_router.post("/batch-analyze-risk", response_model=BatchRiskAnalysisResponse)
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
        
        # Convert any numpy types before returning
        return convert_numpy_types(analysis_result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch risk analysis: {str(e)}"
        )

@api_router.post("/user-risk-profile", response_model=UserProfileResponse)
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
        
        # Convert any numpy types before returning
        return convert_numpy_types(profile_result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating user risk profile: {str(e)}"
        )

@api_router.post("/upload-analyze-file", response_model=FileUploadResponse)
async def upload_analyze_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file and analyze each row for threats
    
    This endpoint processes files row by row to efficiently handle large files
    without excessive memory usage. Each row is analyzed individually using
    the risk profile analysis service.
    
    Expected columns (not all are required):
    - text: The content to analyze (required)
    - user_id: Optional identifier for the user
    - timestamp: Optional timestamp for the content
    - tweet_id: Optional identifier for the tweet/post
    
    Supports both .csv and .xlsx/.xls formats.
    """
    start_time = time.time()
    results = []
    successful_rows = 0
    failed_rows = 0
    file_type = ""
    
    logger.info(f"Starting file upload and analysis for file: {file.filename}")
    
    try:
        # Get file extension to determine type
        filename = file.filename
        if filename.endswith('.csv'):
            file_type = "csv"
            logger.info(f"Detected CSV file: {filename}")
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_type = "excel"
            logger.info(f"Detected Excel file: {filename}")
        else:
            error_msg = f"Unsupported file format for file: {filename}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Read file content
        logger.info("Reading file content")
        try:
            content = await file.read()
            logger.info(f"Successfully read {len(content)} bytes from file")
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            raise
        
        # Process file based on type
        if file_type == "csv":
            # For CSV, use StringIO
            try:
                logger.info("Parsing CSV file")
                file_obj = io.StringIO(content.decode('utf-8'))
                df = pd.read_csv(file_obj)
                logger.info(f"Successfully parsed CSV with {len(df)} rows and {len(df.columns)} columns")
            except Exception as e:
                error_msg = f"Error reading CSV file: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
        else:
            # For Excel, use BytesIO and a temporary file
            try:
                logger.info("Parsing Excel file")
                # Excel requires binary data
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    logger.info(f"Created temporary file at {temp_file_path}")
                
                df = pd.read_excel(temp_file_path)
                logger.info(f"Successfully parsed Excel with {len(df)} rows and {len(df.columns)} columns")
                
                # Remove the temporary file
                try:
                    os.unlink(temp_file_path)
                    logger.info("Removed temporary file")
                except Exception as e:
                    logger.warning(f"Could not remove temp file: {str(e)}")
            except Exception as e:
                error_msg = f"Error reading Excel file: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
        
        text_column_name = None
        if 'text' in df.columns:
            text_column_name = 'text'
            logger.info("Found 'text' column.")
        elif 'tweet' in df.columns:
            text_column_name = 'tweet'
            logger.info("Found 'tweet' column.")
        
        if text_column_name is None:
            error_msg = "The file must contain either a 'text' or 'tweet' column with content to analyze."
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Log column names
        logger.info(f"Found columns in file: {', '.join(df.columns)}")
        
        # Process each row
        total_rows = len(df)
        logger.info(f"Beginning to process {total_rows} rows using column '{text_column_name}'")
        
        for idx, row in df.iterrows():
            row_start_time = time.time()
            row_number = idx + 2  # +2 because idx is 0-based and we skip header row
            
            logger.info(f"Processing row {row_number}/{total_rows + 1}")
            
            try:
                # Extract data from row using the identified text column
                text = row.get(text_column_name, None)
                user_id_val = row.get('user_id', None) if 'user_id' in df.columns else None
                user_id = str(user_id_val) if user_id_val is not None else None
                
                timestamp_val = row.get('timestamp', None) if 'timestamp' in df.columns else None
                timestamp = str(timestamp_val) if timestamp_val is not None else None
                
                tweet_id_val = row.get('tweet_id', None) if 'tweet_id' in df.columns else None
                tweet_id = str(tweet_id_val) if tweet_id_val is not None else None
                
                logger.debug(f"Row {row_number} data: user_id={user_id}, timestamp={timestamp}, tweet_id={tweet_id}")
                
                # Skip rows with empty text
                if not text or pd.isna(text):
                    logger.warning(f"Row {row_number}: Empty or missing text in column '{text_column_name}', skipping")
                    row_processing_time = (time.time() - row_start_time) * 1000
                    results.append(FileRowAnalysisResult(
                        row_number=row_number,
                        text=None,
                        user_id=user_id,
                        timestamp=timestamp,
                        tweet_id=tweet_id,
                        analysis_result=None,
                        predicted_class=None,
                        malicious_probability=None,
                        error="Empty or missing text",
                        processing_time_ms=row_processing_time
                    ))
                    failed_rows += 1
                    continue
                
                # Convert to string if not already
                if not isinstance(text, str):
                    logger.debug(f"Row {row_number}: Converting text from {type(text)} to string")
                    text = str(text)
                
                # Analyze the text
                logger.info(f"Row {row_number}: Running analysis on text (length: {len(text)})")
                analysis_result_dict = await analyze_text_with_risk_profile(
                    text=text,
                    user_id=user_id,
                    tweet_id=tweet_id,
                    timestamp=timestamp
                )
                
                # Extract classification results
                pred_class = analysis_result_dict.pop('predicted_class', None)
                pred_prob = analysis_result_dict.pop('malicious_probability', None)
                
                logger.info(f"Row {row_number}: Analysis complete, converting numpy types")
                
                # Check for numpy values before conversion
                has_numpy = False
                if isinstance(analysis_result_dict, dict):
                    for k, v in analysis_result_dict.items():
                        if isinstance(v, (np.integer, np.floating, np.ndarray)):
                            has_numpy = True
                            logger.debug(f"Row {row_number}: Found numpy type in result key '{k}': {type(v)}")
                
                # Convert any numpy types in the analysis result
                try:
                    analysis_result = convert_numpy_types(analysis_result_dict)
                    logger.info(f"Row {row_number}: Successfully converted numpy types")
                except Exception as e:
                    logger.error(f"Row {row_number}: Error converting numpy types: {str(e)}")
                    # If conversion fails, attempt a more basic approach
                    if isinstance(analysis_result, dict):
                        for k, v in analysis_result.items():
                            if isinstance(v, np.floating):
                                analysis_result[k] = float(v)
                            elif isinstance(v, np.integer):
                                analysis_result[k] = int(v)
                    logger.info(f"Row {row_number}: Applied fallback conversion")
                
                row_processing_time = (time.time() - row_start_time) * 1000
                logger.info(f"Row {row_number}: Completed in {row_processing_time:.2f}ms")
                
                # Append result
                results.append(FileRowAnalysisResult(
                    row_number=row_number,
                    text=text,
                    user_id=user_id,
                    timestamp=timestamp,
                    tweet_id=tweet_id,
                    analysis_result=analysis_result,
                    predicted_class=pred_class,
                    malicious_probability=pred_prob,
                    error=None,
                    processing_time_ms=row_processing_time
                ))
                successful_rows += 1
                
            except Exception as e:
                logger.error(f"Row {row_number}: Error processing: {str(e)}")
                row_processing_time = (time.time() - row_start_time) * 1000
                results.append(FileRowAnalysisResult(
                    row_number=row_number,
                    text=row.get(text_column_name, None) if text_column_name in df.columns else None,
                    user_id=row.get('user_id', None) if 'user_id' in df.columns else None,
                    timestamp=row.get('timestamp', None) if 'timestamp' in df.columns else None,
                    tweet_id=row.get('tweet_id', None) if 'tweet_id' in df.columns else None,
                    analysis_result=None,
                    predicted_class=None,
                    malicious_probability=None,
                    error=f"Error processing row: {str(e)}",
                    processing_time_ms=row_processing_time
                ))
                failed_rows += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        logger.info(f"File processing complete. Total: {total_rows}, Successful: {successful_rows}, Failed: {failed_rows}")
        
        # Create and debug the response object before returning
        response = FileUploadResponse(
            results=results,
            total_rows=total_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            processing_time_ms=total_processing_time,
            file_type=file_type
        )
        
        logger.info(f"Returning response with {len(results)} results")
        
        return response
        
    except Exception as e:
        # Handle overall process errors
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )