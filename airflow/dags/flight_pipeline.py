from __future__ import annotations

import shlex
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Variable.get("project_root", "/opt/airflow/project")
PYTHON_BIN = Variable.get("python_bin", "python")
EVAL_REPORT_PATH = Variable.get(
    "eval_report_path", f"{PROJECT_ROOT}/reports/eval.json"
)
TRAINED_MODEL_PATH = Variable.get(
    "trained_model_path", f"{PROJECT_ROOT}/model_store/random_forest.joblib"
)
REGISTERED_MODEL_PATH = Variable.get(
    "registered_model_path",
    f"{PROJECT_ROOT}/model_store/production/random_forest.joblib",
)
ROC_AUC_THRESHOLD = float(Variable.get("roc_auc_threshold", 0.9))


def bash_python(command: str) -> str:
    project_dir = shlex.quote(PROJECT_ROOT)
    python_bin = shlex.quote(PYTHON_BIN)
    return f"cd {project_dir} && {python_bin} {command}"


default_args = {
    "owner": "team1",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="flight_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "lab8"],
) as dag:
    download_data = BashOperator(
        task_id="download_data",
        bash_command=bash_python("src/download_data.py"),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=bash_python("src/preprocess.py"),
    )

    train = BashOperator(
        task_id="train",
        bash_command=bash_python("src/train.py"),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=bash_python("src/evaluate.py"),
    )

    register_cmd = (
        "src/register.py "
        f"--model-path {shlex.quote(TRAINED_MODEL_PATH)} "
        f"--registry-path {shlex.quote(REGISTERED_MODEL_PATH)} "
        f"--report-path {shlex.quote(EVAL_REPORT_PATH)} "
        f"--metric roc_auc --threshold {ROC_AUC_THRESHOLD}"
    )

    register = BashOperator(
        task_id="register",
        bash_command=bash_python(register_cmd),
    )

    download_data >> preprocess >> train >> evaluate >> register


dag.doc_md = __doc__ = """
### Flight pipeline (Lab 8)

Учебный DAG orchestrates ETL → обучение → оценку → регистрацию модели.
Пути и пороги кастомизируются через Airflow Variables: `project_root`, `python_bin`,
`eval_report_path`, `trained_model_path`, `registered_model_path`, `roc_auc_threshold`.
"""
