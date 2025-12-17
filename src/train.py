from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/processed.csv")
FEATURE_REPO = Path("feature_repo")
MODEL_DIR = Path("model_store")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


FEATURE_COLUMNS = [
    "char_len",
    "word_len",
    "num_digits",
    "num_urls",
    "num_domains",
    "upper_ratio",
]


def load_entity_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    df = pd.read_csv(path, parse_dates=["event_timestamp"])
    required = ["sms_id", "event_timestamp", "target"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for entity dataframe: {missing}")
    return df[required]


def fetch_features_from_store(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    feature_refs = [f"sms_features:{name}" for name in FEATURE_COLUMNS]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    if "target" not in training_df.columns:
        training_df = training_df.merge(
            entity_df[["sms_id", "target"]],
            on="sms_id",
            how="left",
        )
    missing = [col for col in FEATURE_COLUMNS + ["target"] if col not in training_df.columns]
    if missing:
        raise ValueError(f"Feast dataset is missing columns: {missing}")
    return training_df


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, float, float]:
    X = df[FEATURE_COLUMNS]
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
    entity_df = load_entity_dataframe(DATA_PATH)
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    training_df = fetch_features_from_store(store, entity_df)

    mlflow.set_experiment("flight_delay")
    with mlflow.start_run():
        model, accuracy, roc_auc = train_model(training_df)

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
