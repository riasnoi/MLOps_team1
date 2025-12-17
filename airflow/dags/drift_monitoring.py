from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


PROJECT_ROOT = Variable.get("project_root", "/opt/airflow/project")
DRIFT_REFERENCE_PATH = Variable.get(
    "drift_reference_path", f"{PROJECT_ROOT}/data/processed/processed.csv"
)
DRIFT_PRODUCTION_PATH = Variable.get(
    "drift_production_path", f"{PROJECT_ROOT}/data/production/recent.csv"
)
DRIFT_REPORT_PATH = Variable.get(
    "drift_report_path", f"{PROJECT_ROOT}/reports/drift_report.json"
)
EVAL_REPORT_PATH = Variable.get(
    "eval_report_path", f"{PROJECT_ROOT}/reports/eval.json"
)
REGISTERED_MODEL_PATH = Variable.get(
    "registered_model_path",
    f"{PROJECT_ROOT}/model_store/production/random_forest.joblib",
)

PSI_THRESHOLD = float(Variable.get("drift_psi_threshold", 0.2))
KS_THRESHOLD = float(Variable.get("drift_ks_threshold", 0.15))
METRIC_DROP_THRESHOLD = float(Variable.get("drift_metric_drop_threshold", 0.05))
PSI_BINS = int(Variable.get("drift_psi_bins", 10))


default_args = {
    "owner": "team1",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def drift_branch() -> str:
    sys.path.insert(0, PROJECT_ROOT)
    # Import inside callable so Airflow workers have access to project code
    from src import drift_check

    result = drift_check.run_drift_check(
        reference_path=Path(DRIFT_REFERENCE_PATH),
        production_path=Path(DRIFT_PRODUCTION_PATH),
        model_path=Path(REGISTERED_MODEL_PATH),
        baseline_report_path=Path(EVAL_REPORT_PATH),
        report_path=Path(DRIFT_REPORT_PATH),
        psi_threshold=PSI_THRESHOLD,
        ks_threshold=KS_THRESHOLD,
        metric_drop_threshold=METRIC_DROP_THRESHOLD,
        psi_bins=PSI_BINS,
    )
    return "trigger_retrain" if result.get("drift_detected") else "no_drift"


with DAG(
    dag_id="drift_monitoring",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=["mlops", "lab12", "drift"],
) as dag:
    start = EmptyOperator(task_id="start")

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=drift_branch,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="flight_pipeline",
        conf={
            "reason": "drift_detected",
            "drift_report_path": DRIFT_REPORT_PATH,
            "production_path": DRIFT_PRODUCTION_PATH,
        },
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(task_id="no_drift")

    start >> check_drift >> [trigger_retrain, no_drift]


dag.doc_md = __doc__ = """
### Drift monitoring (Lab 12)

Периодически сравнивает распределения фичей (PSI/KS) и ROC-AUC модели на продакшен батче.
Результат сохраняется в `reports/drift_report.json`. При флаге дрейфа триггерится DAG
`flight_pipeline` для переобучения/регистрации модели.

Настраиваемые Airflow Variables:
- `drift_reference_path` — эталонный (train) датасет, default: `/opt/airflow/project/data/processed/processed.csv`
- `drift_production_path` — свежий продакшен батч, default: `/opt/airflow/project/data/production/recent.csv`
- `drift_report_path` — куда писать JSON-отчёт о дрейфе, default: `/opt/airflow/project/reports/drift_report.json`
- `drift_psi_threshold` / `drift_ks_threshold` — пороги по PSI/KS
- `drift_metric_drop_threshold` — допустимое падение ROC-AUC от baseline (`reports/eval.json`)
- `drift_psi_bins` — число квантильных бинов для PSI
"""
