from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/processed.csv")
MODEL_PATH = Path("model_store/random_forest.joblib")
REPORTS_DIR = Path("reports")
REPORT_PATH = REPORTS_DIR / "eval.json"
FEATURE_COLUMNS = [
    "char_len",
    "word_len",
    "num_digits",
    "num_urls",
    "num_domains",
    "upper_ratio",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "flight_delay"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    df = pd.read_csv(path)
    expected = FEATURE_COLUMNS + ["target"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")
    return df


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return joblib.load(path)


def evaluate(model, df: pd.DataFrame) -> tuple[dict[str, float], list[list[int]], int]:
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    metrics = {
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    return metrics, cm, int(len(y_test))


def save_report(metrics: dict[str, float], cm: list[list[int]], n_samples: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": cm,
        },
        "n_test_samples": n_samples,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def main() -> None:
    df = load_dataset(DATA_PATH)
    model = load_model(MODEL_PATH)
    metrics, cm, n_samples = evaluate(model, df)

    save_report(metrics, cm, n_samples, REPORT_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        tn, fp, fn, tp = (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        mlflow.log_metrics(
            {
                "confusion_tn": tn,
                "confusion_fp": fp,
                "confusion_fn": fn,
                "confusion_tp": tp,
            }
        )
        mlflow.log_metric("n_test_samples", n_samples)
        mlflow.log_artifact(str(REPORT_PATH))

    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Confusion matrix (labels=['ham', 'spam']): {cm}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
