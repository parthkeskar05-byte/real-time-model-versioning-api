"""Test suite for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_versions():
    """Test listing versions endpoint."""
    response = client.get("/versions")
    assert response.status_code == 200
    assert "versions" in response.json()
    assert "count" in response.json()


def test_compare_invalid_runs():
    """Test comparing invalid run IDs."""
    response = client.post(
        "/compare",
        json={"run_id_1": "invalid1", "run_id_2": "invalid2"}
    )
    assert response.status_code == 404
