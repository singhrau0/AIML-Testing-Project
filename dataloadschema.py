"""Load, validate, clean, and preprocess DVC-tracked Iris data.

Requires the dvc command and Python packages: pandas, mlflow, scikit-learn, joblib.
"""

from pathlib import Path
import json
import subprocess

import pandas as pd
import joblib
from mlflow.models import validate_schema
from mlflow.types import ColSpec, Schema
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data():
    # Use paths relative to this script so it works from any directory.
    project_dir = Path(__file__).resolve().parent

    # Load the version in Iris.csv.dvc. Update that file to select newer data.
    # This project uses a local cache; pull when a default remote is configured.
    # Call the installed DVC command, even if it uses a different Python.
    remote = subprocess.run(
        ["dvc", "config", "core.remote"],
        cwd=project_dir, capture_output=True, text=True,
    )
    command = "pull" if remote.stdout.strip() else "checkout"
    subprocess.run(
        ["dvc", command, "Iris.csv.dvc"], cwd=project_dir, check=True,
    )

    data = pd.read_csv(project_dir / "Iris.csv")

    # Define the expected schema explicitly, rather than infer it from new data.
    schema = Schema([
        ColSpec("long", "Id"),
        ColSpec("double", "SepalLengthCm"),
        ColSpec("double", "SepalWidthCm"),
        ColSpec("double", "PetalLengthCm"),
        ColSpec("double", "PetalWidthCm"),
        ColSpec("string", "Species"),
    ])

    # Reject extra columns and missing values before MLflow checks the types.
    if set(data.columns) != set(schema.input_names()):
        raise ValueError("CSV columns must match the expected Iris schema.")
    if data.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    # MLflow raises an exception if required columns or types are invalid.
    validate_schema(data, schema)
    return data


def clean_data(data):
    # Work on a copy to keep the original data unchanged.
    data = data.copy()

    # Remove spaces around species names and drop rows with empty names.
    data["Species"] = data["Species"].str.strip()
    data = data[data["Species"].notna() & data["Species"].ne("")]

    # Flower measurements must be positive, finite numbers.
    measurements = [
        "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
    ]
    valid_measurements = (
        (data[measurements] > 0) & (data[measurements] < float("inf"))
    ).all(axis=1)
    data = data[valid_measurements]

    # Remove fully identical rows, then reset the row numbers.
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def preprocess_data(data):
    # Stop if cleaning removed every row.
    if data.empty:
        raise ValueError("No data remains for preprocessing.")

    # X contains the four numeric measurements. Id is not a model feature.
    X = data.drop(columns=["Id", "Species"]).astype(float)

    # Use fixed numeric labels so each species always has the same code.
    species_labels = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    y = data["Species"].map(species_labels)
    if y.isna().any():
        raise ValueError("The dataset contains an unknown species name.")
    y = y.astype(int)

    # Fit any feature scaler on training data after a train/test split.
    return X, y


def split_data(X, y):
    # Keep 20% for testing and 80% for training.
    # Stratify preserves species proportions; random_state makes it repeatable.
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    # Logistic Regression needs at least two different target classes.
    if y_train.nunique() < 2:
        raise ValueError("Training requires at least two species.")

    # Scaling helps Logistic Regression and KNN; save it with each model.
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "knn_classifier": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }

    # Store trained model files in a models folder beside this script.
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    # Fit models and their scalers on training data only, then save them.
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"Trained and saved: {model_path}")

    return models


def test_models(models, X_test, y_test):
    # Evaluate only the held-out test data; do not train models here.
    results = {}
    species_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        labels = sorted(set(y_test) | set(y_pred))
        class_names = [species_names[label] for label in labels]

        # The report includes precision, recall, F1, and support (row count)
        # for each species, plus macro and weighted averages.
        report = classification_report(
            y_test, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        results[name] = {
            "test_rows": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": report,
            # Matrix rows are actual classes; columns are predicted classes.
            "confusion_matrix_labels": class_names,
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=labels
            ).tolist(),
        }
        print(f"{name} test accuracy: {results[name]['accuracy']:.4f}")

    # Save metrics for all models in one readable JSON file.
    metrics_path = Path(__file__).resolve().parent / "model_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, allow_nan=False)
    print(f"Test metrics saved: {metrics_path}")
    return results


if __name__ == "__main__":
    data = load_data()
    print(f"Schema validation passed. Loaded {len(data)} rows.")
    data = clean_data(data)
    print(f"Data cleaning completed. Remaining rows: {len(data)}")
    X, y = preprocess_data(data)
    print(f"Preprocessing completed. X shape: {X.shape}, y shape: {y.shape}")
    print("Features (X):")
    print(X.head())
    print("Target (y):")
    print(y.head())
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")
    # Train on the training set, then evaluate on the separate test set.
    models = train_models(X_train, y_train)
    metrics = test_models(models, X_test, y_test)
