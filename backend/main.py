from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.routes import api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This is where we initialize and clean up resources.
    """
    # Startup: Load resources like ML models
    logger.info("Starting up application and loading resources...")
    
    # We'll import the model loading here to avoid circular imports
    from app.services.model_service import load_model
    await load_model()
    
    # Log initialization of risk profiling services
    logger.info("Initializing risk profiling services...")
    
    # Initialize risk profiling components (if needed)
    # These services don't need explicit initialization
    from app.services.risk_profiling_service import thread_pool as risk_thread_pool
    from app.services.profile_service import thread_pool as profile_thread_pool
    
    logger.info(f"Risk profiling thread pools initialized")
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down application and cleaning up resources...")
    
    # Clean up NER model 
    from app.services.model_service import cleanup_model
    await cleanup_model()
    
    # Shut down thread pools
    logger.info("Shutting down risk profiling thread pools...")
    risk_thread_pool.shutdown(wait=True)
    profile_thread_pool.shutdown(wait=True)
    
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 