"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


class VersionResponse(BaseModel):
    """Response model for listing versions."""
    versions: List[Dict[str, Any]]
    count: int


class CompareRequest(BaseModel):
    """Request model for comparing two model versions."""
    run_id_1: str
    run_id_2: str


class CompareResponse(BaseModel):
    """Response model for model comparison."""
    run_id_1: str
    run_id_2: str
    comparison: Dict[str, Any]
