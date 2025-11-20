from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register trained model if evaluation passes threshold")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model_store/random_forest.joblib"),
        help="Path to the trained model artifact",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("model_store/production/random_forest.joblib"),
        help="Destination path for the registered model",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/eval.json"),
        help="Path to the evaluation report json",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        help="Metric from the report that must satisfy the threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum metric value required to register the model",
    )
    return parser.parse_args()


def read_metric(report_path: Path, metric_name: str) -> float:
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found at {report_path}")
    with report_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    metrics = payload.get("metrics", {})
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not found in report {report_path}")
    return float(metrics[metric_name])


def main() -> None:
    args = parse_args()
    metric_value = read_metric(args.report_path, args.metric)

    if metric_value < args.threshold:
        raise ValueError(
            f"Metric {args.metric}={metric_value:.4f} is below threshold {args.threshold:.4f}."
        )

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {args.model_path}")

    args.registry_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_path, args.registry_path)

    print(
        f"Model registered at {args.registry_path} based on {args.metric}={metric_value:.4f} >= {args.threshold:.4f}"
    )


if __name__ == "__main__":
    main()
