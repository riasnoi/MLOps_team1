# SMS Spam Detector (MLOps labs 1–6)

Лёгкий и актуальный учебный проект: предобработка SMS/мессенджер сообщений для задач антиспама/антифишинга, обучение и оценка модели, а также минимальный REST API и Docker для инференса.
Работаем через **venv + pip**. DVC — для версионирования данных и пайплайна.

## Быстрый старт

```bash
git clone git@github.com:riasnoi/MLOps_team1.git
cd MLOps_team1

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Запуск сервиса

Локально:

```bash
MODEL_DIR="$PWD/model_store" MODEL_FILENAME="random_forest.joblib" \
uvicorn src.api:app --host 127.0.0.1 --port 8080
```

Проверка:

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" \
  -d '{"text":"hello friend"}'
```

Docker:

```bash
docker build -t spam-api:lab6 .
docker run --rm -p 8080:8080 -v "$(pwd)/model_store:/models" \
  -e MODEL_DIR=/models -e MODEL_FILENAME=random_forest.joblib spam-api:lab6
```

## Lab 8 — Airflow pipeline

Мы добавили оркестрацию пайплайна (download → preprocess → train → evaluate → register) в Airflow.

### Docker Compose с Airflow

1. Перейдите в каталог `airflow` и при необходимости скорректируйте `AIRFLOW_UID` и `AIRFLOW_GID` в `.env` (по умолчанию `50000:0`; на Linux лучше выставить `$(id -u)`). `.env` — персональный файл, его не коммитим; при необходимости можно пересоздать из `.env.example`:

   ```bash
   cd airflow
   # опционально
   cp .env .env.backup
   # отредактируйте .env любым редактором
   ```

2. Выполните инициализацию БД/переменных и создайте пользователя UI:

   ```bash
   docker compose up airflow-init
   ```

3. Поднимите основные сервисы (Postgres, scheduler, webserver, triggerer):

   ```bash
   docker compose up -d
   ```

4. Откройте UI по адресу [http://localhost:8087](http://localhost:8087) и войдите (`admin` / `admin`).

5. Найдите DAG `flight_pipeline`, вручную запустите его из UI (кнопка **Play**). Вчерашний запуск и логи доступны из UI; артефакты пишутся в общие volume'ы репозитория (`data/`, `model_store/`, `reports/`).

6. После проверки остановите все контейнеры:

   ```bash
   docker compose down --volumes
   ```

### Что делает DAG

- `download_data` → `preprocess` → `train` → `evaluate` → `register` выполняются строго последовательно через `BashOperator` и вызывают соответствующие Python‑скрипты из `src/`.
- `airflow/dags/flight_pipeline.py` настроен на работу из Docker'а: репозиторий монтируется в `/opt/airflow/project`, поэтому все пути относительно корня проекта.
- Во время `airflow-init` автоматически создаются Airflow Variables:

  | Variable | Значение по умолчанию | Назначение |
  | --- | --- | --- |
  | `project_root` | `/opt/airflow/project` | Рабочий каталог пайплайна |
  | `python_bin` | `python` | Интерпретатор, которым запускаются скрипты |
  | `eval_report_path` | `/opt/airflow/project/reports/eval.json` | JSON отчёт `evaluate` |
  | `trained_model_path` | `/opt/airflow/project/model_store/random_forest.joblib` | Модель из `train` |
  | `registered_model_path` | `/opt/airflow/project/model_store/production/random_forest.joblib` | Куда копировать модель при регистрации |
  | `roc_auc_threshold` | `0.9` | Минимальный ROC-AUC для регистрации |

  Значения можно менять в UI (Admin → Variables) перед запуском DAG.

- Скрипт `src/register.py` читает `reports/eval.json`, проверяет метрику `roc_auc`, и только при успешном пороге копирует модель в `model_store/production/random_forest.joblib`. При провале задача завершается ошибкой, и DAG подсвечивает причину.
- Логи задач сохраняются в `airflow/logs`, модели — в `model_store/`, отчёт — в `reports/eval.json`. Эти каталоги уже проброшены в Docker Compose, поэтому артефакты доступны и с хоста.
