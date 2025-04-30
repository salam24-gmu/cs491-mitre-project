from fastapi import APIRouter, HTTPException, BackgroundTasks, status, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time
import os
import io
import pandas as pd
import tempfile
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from app.services.model_service import run_inference, evaluate_entities
from app.services.integration_service import (
    analyze_text_with_risk_profile,
    analyze_texts_batch_with_risk_profile,
    get_user_risk_profile
)
from app.services.visualization_service import generate_visualizations
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()

# --- Define base path for generated plots ---
# Ensure this path exists or is created on startup if needed
# For security, ensure this path is outside the main app code if possible
PLOTS_BASE_DIR = Path(settings.PLOTS_OUTPUT_DIR) # Get base dir from settings

# ----- Utility Functions -----

def _clean_value(value):
    """ Utility function to handle potential NaN values """
    if pd.isna(value):
        return None
    return value

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

    # Custom validator to handle NaN -> None
    # Pydantic v2 doesn't strictly need this if Optional is used correctly,
    # but explicit handling can prevent issues if data isn't clean.
    class Config:
        validate_assignment = True # Ensure validators run on assignment

    @classmethod
    def model_validate(cls, obj, *, context=None):
        # Clean fields before standard validation
        if isinstance(obj, dict):
            cleaned_obj = {k: _clean_value(v) for k, v in obj.items()}
            return super().model_validate(cleaned_obj, context=context)
        return super().model_validate(obj, context=context)

class FileUploadResponse(BaseModel):
    """Response model for file upload and analysis"""
    results: List[FileRowAnalysisResult]
    total_rows: int
    successful_rows: int
    failed_rows: int
    processing_time_ms: float
    file_type: str
    plot_paths: Optional[Dict[str, Any]] = Field(None, 
        description="Dictionary containing request_id and relative plot filenames.",
        example={
            "request_id": "uuid_string",
            "base_path": "generated_plots/uuid_string",
            "overall": {
                "temporal_distribution": "temporal_distribution.png"
            },
            "user_plots": {
                "user_123": {
                    "entity_distribution": "entity_distribution_user_123.png",
                    "risk_profile": "risk_profile_user_123.png",
                    "risk_trend": "risk_trend_user_123.png"
                }
            },
            "error": None
        }
    )

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
    Upload a CSV or Excel file, analyze each row, and generate summary plots.
    Returns analysis results per row and information needed to retrieve plots.
    """
    start_time = time.time()
    results = [] # Will store FileRowAnalysisResult dictionaries/objects
    successful_rows = 0
    failed_rows = 0
    file_type = ""
    df = None
    
    logger.info(f"Starting file upload and analysis for file: {file.filename}")
    
    # Determine file type
    if file.filename.endswith('.csv'):
        file_type = "csv"
    elif file.filename.endswith(('.xlsx', '.xls')):
        file_type = "excel"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload a CSV or Excel file."
        )
    
    logger.info(f"Detected file type: {file_type}")
    
    # Read file content
    try:
        content = await file.read()
        logger.info(f"Successfully read {len(content)} bytes from file")
    except Exception as e:
        error_msg = f"Error reading uploaded file content: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
        
    # Process file based on type
    if file_type == "csv":
        # For CSV, use StringIO
        try:
            logger.info("Parsing CSV file")
            # Handle potential BOM (Byte Order Mark)
            decoded_content = content.decode('utf-8-sig')
            file_obj = io.StringIO(decoded_content)
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
        # For Excel, use BytesIO
        try:
            logger.info("Parsing Excel file")
            file_obj = io.BytesIO(content)
            df = pd.read_excel(file_obj, engine='openpyxl') # Specify engine for broader compatibility
            logger.info(f"Successfully parsed Excel with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            error_msg = f"Error reading Excel file: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
    
    # --- DataFrame Processing ---
    if df is None:
        error_msg = "Failed to create DataFrame from file."
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
        
    # Identify text column
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
    
    # --- Row Iteration and Analysis ---
    for idx, row in df.iterrows():
        row_start_time = time.time()
        # Use 1-based index for row number, plus 1 for header -> idx + 2
        row_number = idx + 2
        
        logger.info(f"Processing row {row_number}/{total_rows + 1}")
        row_result = {
            "row_number": row_number,
            "text": None, "user_id": None, "timestamp": None, "tweet_id": None,
            "analysis_result": None, "predicted_class": None, "malicious_probability": None,
            "error": None, "processing_time_ms": 0.0
        }

        try:
            # Extract data safely using .get and handle potential NaN/None
            text = _clean_value(row.get(text_column_name))
            user_id_val = _clean_value(row.get('user_id'))
            user_id = str(user_id_val) if user_id_val is not None else None
            
            timestamp_val = _clean_value(row.get('timestamp'))
            # Attempt conversion to standard string format if needed (e.g., from Timestamp object)
            if isinstance(timestamp_val, pd.Timestamp):
                timestamp = timestamp_val.isoformat()
            elif timestamp_val is not None:
                timestamp = str(timestamp_val)
            else:
                timestamp = None
            
            tweet_id_val = _clean_value(row.get('tweet_id'))
            tweet_id = str(tweet_id_val) if tweet_id_val is not None else None

            # Update result dict with extracted values
            row_result.update({"text": text, "user_id": user_id, "timestamp": timestamp, "tweet_id": tweet_id})
            
            logger.debug(f"Row {row_number} data: user_id={user_id}, timestamp={timestamp}, tweet_id={tweet_id}")
            
            # Skip rows with empty text after cleaning
            if not text:
                logger.warning(f"Row {row_number}: Empty or missing text in column '{text_column_name}', skipping")
                row_result["error"] = "Empty or missing text"
                failed_rows += 1
            else:
                # Ensure text is string
                if not isinstance(text, str):
                    logger.debug(f"Row {row_number}: Converting text from {type(text)} to string")
                    text = str(text)
                    row_result["text"] = text # Update text in result
                
                # Analyze the text
                logger.info(f"Row {row_number}: Running analysis on text (length: {len(text)}) [{text[:50]}...]")
                analysis_result_dict = await analyze_text_with_risk_profile(
                    text=text,
                    user_id=user_id,
                    tweet_id=tweet_id,
                    timestamp=timestamp
                )
                
                # Check for errors from analysis service
                if analysis_result_dict.get("error"):
                    error_msg = f"Analysis service error: {analysis_result_dict['error']}"
                    logger.error(f"Row {row_number}: {error_msg}")
                    row_result["error"] = error_msg
                    failed_rows += 1
                else:
                    # Extract classification results if they exist
                    pred_class = analysis_result_dict.pop('predicted_class', None)
                    pred_prob = analysis_result_dict.pop('malicious_probability', None)
                    
                    logger.info(f"Row {row_number}: Analysis complete.")
                    
                    # Convert any remaining numpy types in the analysis result (entities, risk_metrics)
                    try:
                        # Note: analysis_result_dict now contains entities, risk_metrics, etc. after popping classification
                        cleaned_analysis_data = convert_numpy_types(analysis_result_dict)
                        row_result["analysis_result"] = cleaned_analysis_data
                        row_result["predicted_class"] = pred_class
                        row_result["malicious_probability"] = pred_prob
                        successful_rows += 1
                        logger.info(f"Row {row_number}: Successfully processed and converted types.")
                    except Exception as e:
                        logger.error(f"Row {row_number}: Error converting numpy types after analysis: {str(e)}")
                        row_result["error"] = f"Type conversion error: {str(e)}"
                        failed_rows += 1
                
        except Exception as e:
            # Catch errors during row extraction or the main analysis call
            error_trace = f"Error processing row {row_number}: {str(e)}"
            logger.error(error_trace, exc_info=True)
            row_result["error"] = error_trace
            failed_rows += 1
        finally:
            # Calculate processing time and append the final result for this row
            row_processing_time = (time.time() - row_start_time) * 1000
            row_result["processing_time_ms"] = row_processing_time
            results.append(row_result) # Append the dictionary
            logger.info(f"Row {row_number}: Finished in {row_processing_time:.2f}ms. Status: {'Success' if not row_result['error'] else 'Failed'}")
    
    # --- Post-Processing and Visualization ---
    total_processing_time = (time.time() - start_time) * 1000
    logger.info(f"File processing loop complete. Total: {total_rows}, Successful: {successful_rows}, Failed: {failed_rows}")
    
    # +++ Generate Visualizations using the base path from settings +++
    plot_paths_dict = None
    if successful_rows > 0:
        logger.info("Attempting to generate visualizations...")
        try:
            # Pass the list of result dictionaries and the base directory
            plot_paths_dict = await generate_visualizations(results, output_base_dir=str(PLOTS_BASE_DIR))
            logger.info(f"Successfully generated visualization paths: {plot_paths_dict}")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}", exc_info=True)
            # Don't fail the whole request, just log the error
            plot_paths_dict = {"error": f"Visualization generation failed: {str(e)}"}
    else:
        logger.warning("Skipping visualization generation as there were no successful rows.")

    # --- Prepare and Return Response ---
    # Validate results with Pydantic model before returning
    validated_results = []
    for res_dict in results:
        try:
            validated_results.append(FileRowAnalysisResult.model_validate(res_dict))
        except Exception as e:
            logger.error(f"Pydantic validation failed for row {res_dict.get('row_number')}: {e}. Raw data: {res_dict}")
            validated_results.append(FileRowAnalysisResult(
                row_number=res_dict.get('row_number', -1),
                error=f"Response validation failed: {e}",
                processing_time_ms=res_dict.get('processing_time_ms', 0.0)
            ))

    response = FileUploadResponse(
        results=validated_results,
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        processing_time_ms=total_processing_time,
        file_type=file_type,
        plot_paths=plot_paths_dict
    )
    
    logger.info(f"Returning response with {len(validated_results)} results and plot paths.")
    
    return response

@api_router.get("/plots/{request_id}/{filename}")
async def get_plot_file(request_id: str, filename: str):
    """
    Serve a generated plot image file.
    
    - **request_id**: The unique ID generated during the file upload analysis (part of the plot_paths response).
    - **filename**: The specific plot filename (e.g., 'temporal_distribution.png', 'risk_trend_user123.png').
    """
    try:
        # --- Security Check --- 
        # Basic sanitization (though FastAPI path params help)
        safe_request_id = Path(request_id).name
        safe_filename = Path(filename).name

        # Construct the full path
        plot_dir = PLOTS_BASE_DIR / safe_request_id
        file_path = plot_dir / safe_filename

        # Resolve the absolute path
        abs_file_path = file_path.resolve()
        abs_base_dir = PLOTS_BASE_DIR.resolve()

        # Check 1: Ensure the resolved path is still within the base directory
        if not str(abs_file_path).startswith(str(abs_base_dir)):
            logger.warning(f"Attempt to access file outside plot directory: {file_path}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

        # Check 2: Ensure the file exists
        if not abs_file_path.is_file():
            logger.error(f"Plot file not found: {abs_file_path}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plot file not found.")

        # Return the file
        logger.info(f"Serving plot file: {abs_file_path}")
        return FileResponse(path=str(abs_file_path), media_type='image/png')

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error serving plot file {request_id}/{filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while retrieving plot.")

# Add a check/creation for the plots base directory on startup if needed
@api_router.on_event("startup")
async def startup_event():
    try:
        PLOTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured plots directory exists: {PLOTS_BASE_DIR}")
    except Exception as e:
        logger.error(f"Failed to create plots directory {PLOTS_BASE_DIR}: {e}")