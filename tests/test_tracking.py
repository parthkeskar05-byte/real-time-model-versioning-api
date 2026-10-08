"""Test suite for MLflow tracking client."""

import pytest
from src.tracking.mlflow_client import ModelTracker


def test_tracker_initialization():
    """Test that tracker initializes correctly."""
    tracker = ModelTracker()
    assert tracker.experiment_name == "model-versioning"
    assert tracker.client is not None


def test_get_all_runs():
    """Test retrieving all runs."""
    tracker = ModelTracker()
    runs = tracker.get_all_runs(max_results=10)
    assert isinstance(runs, list)


def test_compare_runs_invalid():
    """Test comparing invalid run IDs."""
    tracker = ModelTracker()
    result = tracker.compare_runs("invalid_id_1", "invalid_id_2")
    assert "error" in result
