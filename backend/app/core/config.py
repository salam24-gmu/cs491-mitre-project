import os
from typing import List, Optional
from pydantic import BaseModel
import platform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- Get the parent directory (cs491-mitre-project folder) ---
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)

class Settings(BaseModel):
    # Application settings
    PROJECT_NAME: str = "Insider Threat Detection API"
    PROJECT_DESCRIPTION: str = "API for detecting insider threats using NER models"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
    
    # Model settings - paths relative to project root (BASE_DIR)
    MODEL_DIR: str = os.path.join(BASE_DIR, "finetune", "data", "insider-threat-ner_model")
    MODEL_ID: str = "model-4n4ekgnp"  # Specific NER model ID from training

    # --- New Classification Model Settings ---
    CLASSIFICATION_MODEL_DIR: str = os.path.join(BASE_DIR, "finetune", "data", "classification_model")
    CLASSIFICATION_RUN_ID: str = "classification_run_3beidkdc" # Specific classification run ID
    # ------------------------------------------
    
    # Running on CPU (-1) by default
    DEVICE: int = -1  # Use CPU for inference
    
    # Performance settings
    THREAD_POOL_SIZE: int = max(1, (os.cpu_count() or 1) - 1)  # Use all CPUs except one
    MAX_BATCH_SIZE: int = 1  # For CPU inference, keep batch size small initially
    
    # Compute platform information
    PLATFORM: str = platform.system()
    CPU_INFO: str = platform.processor()
    
    # Dataset paths
    TRAINING_DATA_PATH: str = os.path.join(BASE_DIR, "..", "finetune", "data", "main_training_dataset")
    SYNTHETIC_DATA_FILE: str = os.path.join(TRAINING_DATA_PATH, "augmented_synthetic_data_V3.csv")

    # Paths constructor
    @property
    def MODEL_PATH(self) -> str:
        """Full path to the model directory"""
        model_path = os.path.join(self.MODEL_DIR, self.MODEL_ID)
        # Ensure the path exists
        if not os.path.exists(model_path):
            print(f"WARNING: Model path {model_path} does not exist!")
        return model_path
    
    # --- New Classification Model Path Property ---
    @property
    def CLASSIFICATION_MODEL_PATH(self) -> str:
        """Full path to the classification model run directory"""
        path = os.path.join(self.CLASSIFICATION_MODEL_DIR, self.CLASSIFICATION_RUN_ID)
        if not os.path.exists(path):
            print(f"WARNING: Classification model path {path} does not exist!")
        return path
    # -------------------------------------------
    
    @property
    def SYSTEM_INFO(self) -> dict:
        """System information for diagnostics"""
        return {
            "platform": self.PLATFORM,
            "cpu_info": self.CPU_INFO,
            "cpu_count": os.cpu_count(),
            "thread_pool_size": self.THREAD_POOL_SIZE,
            "device": "CPU" if self.DEVICE == -1 else f"GPU:{self.DEVICE}",
            "max_batch_size": self.MAX_BATCH_SIZE
        }

    # --- Update PLOTS_OUTPUT_DIR to be in the project root --- 
    PLOTS_OUTPUT_DIR: str = os.path.join(PROJECT_ROOT_DIR, "generated_plots") 

# Create settings instance
settings = Settings()

# Log configuration information at import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Model path: {settings.MODEL_PATH}")
logger.info(f"Classification model path: {settings.CLASSIFICATION_MODEL_PATH}")
logger.info(f"System info: {settings.SYSTEM_INFO}")

# For direct execution - helpful for debugging configuration
if __name__ == "__main__":
    import json
    print(json.dumps(settings.dict(), indent=2)) 