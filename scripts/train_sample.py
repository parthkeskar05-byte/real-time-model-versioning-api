"""Sample training script demonstrating MLflow integration."""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.git_tracker import GitTracker


def train_model():
    """Train a sample model and log to MLflow."""

    # Initialize git tracker
    git_tracker = GitTracker()
    git_commit = git_tracker.get_current_commit()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("model-versioning")

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with MLflow tracking
    with mlflow.start_run():
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log git commit
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)
            print(f"Logged git commit: {git_commit}")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_model()
