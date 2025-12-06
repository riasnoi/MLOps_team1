from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FEATURE_COLUMNS = [
    "char_len",
    "word_len",
    "num_digits",
    "num_urls",
    "num_domains",
    "upper_ratio",
]


def population_stability_index(
    reference: Iterable[float],
    production: Iterable[float],
    bins: int = 10,
    min_fraction: float = 1e-4,
) -> float:
    """
    Compute PSI between two numeric series.
    Uses quantile bins on the reference sample and guards against zero counts.
    """
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)
    if ref.size == 0 or prod.size == 0:
        raise ValueError("Empty samples passed to PSI calculation")

    # Build quantile-based bins from the reference distribution
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:  # reference is constant
        edges = np.array([ref.min() - 0.5, ref.max() + 0.5])

    ref_counts, _ = np.histogram(ref, bins=edges)
    prod_counts, _ = np.histogram(prod, bins=edges)

    ref_percents = np.maximum(ref_counts / max(ref.size, 1), min_fraction)
    prod_percents = np.maximum(prod_counts / max(prod.size, 1), min_fraction)

    psi_values = (prod_percents - ref_percents) * np.log(prod_percents / ref_percents)
    return float(np.sum(psi_values))


def kolmogorov_smirnov_stat(reference: Iterable[float], production: Iterable[float]) -> float:
    """
    Lightweight KS statistic implementation (max distance between ECDFs).
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    prod = np.sort(np.asarray(production, dtype=float))
    if ref.size == 0 or prod.size == 0:
        raise ValueError("Empty samples passed to KS calculation")

    data = np.concatenate([ref, prod])
    ref_cdf = np.searchsorted(ref, data, side="right") / ref.size
    prod_cdf = np.searchsorted(prod, data, side="right") / prod.size
    return float(np.max(np.abs(ref_cdf - prod_cdf)))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {path} is missing feature columns: {missing}")
    return df


def compute_current_metric(model_path: Path, df: pd.DataFrame) -> float | None:
    if "target" not in df.columns:
        return None
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def read_baseline_metric(report_path: Path, metric_name: str) -> float | None:
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    metrics = payload.get("metrics", {})
    value = metrics.get(metric_name)
    return float(value) if value is not None else None


def run_drift_check(
    reference_path: Path,
    production_path: Path,
    model_path: Path,
    baseline_report_path: Path,
    report_path: Path,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.15,
    metric_drop_threshold: float = 0.05,
    psi_bins: int = 10,
) -> dict:
    reference_df = load_dataset(reference_path)
    production_df = load_dataset(production_path)

    feature_reports: dict[str, dict] = {}
    feature_drift = False

    for col in FEATURE_COLUMNS:
        psi = population_stability_index(reference_df[col], production_df[col], bins=psi_bins)
        ks = kolmogorov_smirnov_stat(reference_df[col], production_df[col])
        drifted = psi >= psi_threshold or ks >= ks_threshold
        feature_reports[col] = {
            "psi": psi,
            "ks": ks,
            "drift": drifted,
        }
        feature_drift = feature_drift or drifted

    baseline_roc_auc = read_baseline_metric(baseline_report_path, "roc_auc")
    current_roc_auc = compute_current_metric(model_path, production_df)
    metric_drop = None
    metric_drift = False
    if baseline_roc_auc is not None and current_roc_auc is not None:
        metric_drop = float(baseline_roc_auc - current_roc_auc)
        metric_drift = metric_drop >= metric_drop_threshold

    drift_detected = feature_drift or metric_drift

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(reference_path),
        "production_path": str(production_path),
        "model_path": str(model_path),
        "baseline_report_path": str(baseline_report_path),
        "psi_threshold": psi_threshold,
        "ks_threshold": ks_threshold,
        "metric_drop_threshold": metric_drop_threshold,
        "features": feature_reports,
        "metrics": {
            "baseline_roc_auc": baseline_roc_auc,
            "current_roc_auc": current_roc_auc,
            "drop": metric_drop,
            "drift": metric_drift,
        },
        "drift_detected": drift_detected,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    status = "DRIFT DETECTED" if drift_detected else "No drift detected"
    print(status)
    print(f"Report saved to {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift checks on features and performance")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("data/processed/processed.csv"),
        help="Reference/train dataset with features and optional target",
    )
    parser.add_argument(
        "--production-path",
        type=Path,
        default=Path("data/production/recent.csv"),
        help="Recent production batch with the same columns as the reference dataset",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model_store/production/random_forest.joblib"),
        help="Registered model used for metric check",
    )
    parser.add_argument(
        "--baseline-report-path",
        type=Path,
        default=Path("reports/eval.json"),
        help="Baseline evaluation report (JSON) with metrics. Optional",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/drift_report.json"),
        help="Where to write the drift report",
    )
    parser.add_argument("--psi-threshold", type=float, default=0.2, help="PSI threshold for drift flag")
    parser.add_argument("--ks-threshold", type=float, default=0.15, help="KS statistic threshold for drift flag")
    parser.add_argument(
        "--metric-drop-threshold",
        type=float,
        default=0.05,
        help="Absolute ROC-AUC drop allowed before triggering drift",
    )
    parser.add_argument("--psi-bins", type=int, default=10, help="Number of quantile bins for PSI")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Return non-zero exit code when drift is detected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_drift_check(
        reference_path=args.reference_path,
        production_path=args.production_path,
        model_path=args.model_path,
        baseline_report_path=args.baseline_report_path,
        report_path=args.report_path,
        psi_threshold=args.psi_threshold,
        ks_threshold=args.ks_threshold,
        metric_drop_threshold=args.metric_drop_threshold,
        psi_bins=args.psi_bins,
    )
    if args.fail_on_drift and report["drift_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
