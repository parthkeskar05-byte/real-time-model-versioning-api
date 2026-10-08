"""FastAPI application for model versioning."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import sys
sys.path.append('..')

from .schemas import VersionResponse, CompareRequest, CompareResponse, HealthResponse
from ..tracking.mlflow_client import ModelTracker
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Model Versioning API",
    description="Real-time model version tracking and comparison",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tracker
tracker = ModelTracker()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="model-versioning-api")


@app.get("/versions", response_model=VersionResponse)
async def list_versions(max_results: int = 100):
    """List all model versions.

    Args:
        max_results: Maximum number of versions to return

    Returns:
        List of model versions with metadata
    """
    try:
        versions = tracker.get_all_runs(max_results=max_results)
        logger.info(f"Retrieved {len(versions)} model versions")
        return VersionResponse(versions=versions, count=len(versions))
    except Exception as e:
        logger.error(f"Error retrieving versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """Compare two model versions.

    Args:
        request: Comparison request with two run IDs

    Returns:
        Comparison results with metric and parameter diffs
    """
    try:
        comparison = tracker.compare_runs(request.run_id_1, request.run_id_2)

        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])

        logger.info(f"Compared runs {request.run_id_1} and {request.run_id_2}")
        return CompareResponse(
            run_id_1=request.run_id_1,
            run_id_2=request.run_id_2,
            comparison=comparison
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version/{run_id}")
async def get_version(run_id: str):
    """Get details for a specific model version.

    Args:
        run_id: MLflow run ID

    Returns:
        Model version details
    """
    try:
        version = tracker.get_run_by_id(run_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        logger.info(f"Retrieved version {run_id}")
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving version: {e}")
        raise HTTPException(status_code=500, detail=str(e))
