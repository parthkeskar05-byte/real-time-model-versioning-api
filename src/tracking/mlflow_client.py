"""MLflow client wrapper for tracking model versions."""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
import yaml


class ModelTracker:
    """Handle MLflow experiment tracking and model versioning."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MLflow tracking client.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.mlflow_config = config['mlflow']
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        self.client = MlflowClient()
        self.experiment_name = self.mlflow_config['experiment_name']

        # Create experiment if it doesn't exist
        try:
            self.experiment = self.client.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = self.client.create_experiment(
                    self.experiment_name,
                    artifact_location=self.mlflow_config['artifact_location']
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception:
            self.experiment_id = self.client.create_experiment(
                self.experiment_name,
                artifact_location=self.mlflow_config['artifact_location']
            )

    def get_all_runs(self, max_results: int = 100) -> List[Dict]:
        """Retrieve all runs from the experiment.

        Args:
            max_results: Maximum number of runs to return

        Returns:
            List of run dictionaries with metadata
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )

        result = []
        for run in runs:
            result.append({
                "run_id": run.info.run_id,
                "experiment_name": self.experiment_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "git_commit": run.data.tags.get("git_commit", "unknown")
            })

        return result

    def get_run_by_id(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID.

        Args:
            run_id: MLflow run ID

        Returns:
            Run dictionary or None if not found
        """
        try:
            run = self.client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_name": self.experiment_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "git_commit": run.data.tags.get("git_commit", "unknown")
            }
        except Exception:
            return None

    def compare_runs(self, run_id_1: str, run_id_2: str) -> Dict:
        """Compare two runs and return differences.

        Args:
            run_id_1: First run ID
            run_id_2: Second run ID

        Returns:
            Dictionary containing metric and parameter diffs
        """
        run1 = self.get_run_by_id(run_id_1)
        run2 = self.get_run_by_id(run_id_2)

        if not run1 or not run2:
            return {"error": "One or both runs not found"}

        # Calculate metric differences
        metrics_diff = {}
        all_metrics = set(run1['metrics'].keys()) | set(run2['metrics'].keys())
        for metric in all_metrics:
            val1 = run1['metrics'].get(metric, 0)
            val2 = run2['metrics'].get(metric, 0)
            metrics_diff[metric] = {
                "run_1": val1,
                "run_2": val2,
                "diff": val2 - val1
            }

        # Compare parameters
        params_diff = {}
        all_params = set(run1['params'].keys()) | set(run2['params'].keys())
        for param in all_params:
            val1 = run1['params'].get(param)
            val2 = run2['params'].get(param)
            if val1 != val2:
                params_diff[param] = {
                    "run_1": val1,
                    "run_2": val2
                }

        return {
            "metrics_diff": metrics_diff,
            "params_diff": params_diff,
            "git_commits": {
                "run_1": run1.get('git_commit', 'unknown'),
                "run_2": run2.get('git_commit', 'unknown')
            }
        }
