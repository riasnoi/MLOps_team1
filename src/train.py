from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/processed.csv")
MODEL_DIR = Path("model_store")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, float, float]:
    feature_columns = [
        "char_len",
        "word_len",
        "num_digits",
        "num_urls",
        "num_domains",
        "upper_ratio",
    ]

    missing = [col for col in feature_columns + ["target"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return model, accuracy, roc_auc


def main() -> None:
    df = load_dataset(DATA_PATH)

    mlflow.set_experiment("flight_delay")
    with mlflow.start_run():
        model, accuracy, roc_auc = train_model(df)

        joblib.dump(model, MODEL_PATH)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model saved to {MODEL_PATH}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
