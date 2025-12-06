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

## Lab 9 — Feature Store (Feast)

В этой лабораторной признаки вынесены в Feature Store (Feast). Каталог `feature_repo/` содержит конфигурацию (`feature_store.yaml`) и описания entity/feature view (`feature_repo.py`). Источником служит оффлайн таблица `data/processed/processed.csv` (и `processed.parquet` для самого Feast), формируемая `src/preprocess.py`.

### Быстрый запуск через Docker

```bash
docker compose -f docker-compose.lab9.yaml up --build
```

Команда соберёт образ на основе существующего `Dockerfile`, выполнит:
1. `python src/download_data.py`
2. `python src/preprocess.py`
3. `cd feature_repo && feast apply`
4. `feast materialize 2020-01-01 2020-12-31`
5. `python src/train.py`

В логах контейнера можно отследить каждый шаг; после завершения контейнер остановится. Артефакты (materialized registry в `feature_repo/data/`, обученная модель в `model_store/`, отчёт в `reports/`) доступны на хосте. По окончании работы остановите сервис: `docker compose -f docker-compose.lab9.yaml down --remove-orphans` (контейнер однократный, поэтому команда опциональна).

### Ручные шаги (если не используете Docker Compose)

1. Убедитесь, что зависимости установлены (`pip install -r requirements.txt`).
2. Скачайте исходные данные (если файла `data/raw/sms_spam.csv` нет):
   ```bash
   python src/download_data.py
   ```
3. Сформируйте набор признаков:
   ```bash
   python src/preprocess.py
   ```
4. Выполните команды Feast:
   ```bash
   cd feature_repo
   feast apply
   feast materialize 2020-01-01 2020-12-31
   cd ..
   ```
5. Запустите обучение — `src/train.py` теперь собирает признаки через Feast offline store:
   ```bash
   python src/train.py
   ```

Feast конфигурация использует file/offline store: entity `sms_id`, признаки `char_len`, `word_len`, `num_digits`, `num_urls`, `num_domains`, `upper_ratio`. Во время materialize создаются файлы `feature_repo/data/registry.db` и `feature_repo/data/online_store.db` (игнорируются в git). Обучающая выборка собирается функцией `FeatureStore.get_historical_features`, после чего модель и метрики сохраняются так же, как в предыдущих лабораториях.

## Lab 10 — Kubernetes (Minikube)

Деплой API в локальный кластер (Minikube). Образ собираем прямо в Docker внутри Minikube, чтобы не пушить в Registry.

### Шаги

1. Запустить кластер и переключить Docker на Minikube:
   ```bash
   minikube start
   minikube addons enable metrics-server   # нужно для HPA
   eval $(minikube docker-env)
   ```
2. Собрать образ (использует существующий `Dockerfile`; модель уже лежит в `model_store/` внутри образа):
   ```bash
   docker build -t spam-api:lab10 .
   ```
3. Применить манифесты:
   ```bash
   kubectl apply -f k8s/
   kubectl get pods,svc
   ```
4. Проверить доступность сервиса:
   ```bash
   SERVICE_URL=$(minikube service spam-api-svc --url)
   curl -X POST "$SERVICE_URL/predict" -H "Content-Type: application/json" \
     -d '{"text":"hello from minikube"}'
   ```
5. Масштабирование:
   ```bash
   # принудительно
   kubectl scale --replicas=3 deployment/spam-api
   # автоматически по CPU (понадобятся метрики)
   kubectl get hpa
   ```

Манифесты в `k8s/`: Deployment с образом `spam-api:lab10`, Service `spam-api-svc` (NodePort 30080), HPA `spam-api-hpa` с порогом 70% CPU. Контейнер читает модель из `/app/model_store/random_forest.joblib` (копируется в образ при сборке).

### Быстрые подсказки/отладка

- Если `minikube start` пишет про docker-env — выполните `eval $(minikube -p minikube docker-env)` перед сборкой.
- Если `minikube service ... --url` висит и вы прерываете его, используйте NodePort напрямую:  
  ```bash
  MINI_IP=$(minikube ip)
  SERVICE_URL="http://${MINI_IP}:30080"
  curl "$SERVICE_URL/health"
  ```  
  (на macOS с Docker-драйвером NodePort может блокироваться; в этом случае делайте `kubectl port-forward svc/spam-api-svc 8080:8080` и обращайтесь к `http://127.0.0.1:8080`).
- Проверка изнутри кластера:  
  ```bash
  kubectl run tmp-curl --rm -it --image=curlimages/curl --restart=Never -- \
    curl -s http://spam-api-svc:8080/health
  ```

## Lab 11 — Monitoring (Prometheus + Grafana)

API инструментирован через `prometheus_client` и отдаёт `/metrics`. Кастомные метрики: `request_count{method,endpoint,http_status}`, `request_latency_seconds_bucket/sum/count`, `prediction_proba_spam_bucket/sum/count`. Для демонстрации долгих ответов можно добавить задержку в `/predict` через переменную `SIMULATED_LATENCY_SEC` (секунды).

## CI/CD до кластера

Готов GitHub Actions workflow `.github/workflows/deploy.yml`: на push в `main` прогоняет тесты, собирает Docker-образ, пушит в GHCR и обновляет образ в Kubernetes.

Минимальные секреты:
- `KUBE_CONFIG_DATA` — base64 от kubeconfig с доступом к кластеру (создайте `cat ~/.kube/config | base64 -w0`).
- Registry использует встроенный `GITHUB_TOKEN` (`packages: write`), поэтому дополнительных секретов для GHCR не нужно. Сделайте пакет публичным один раз через UI GHCR или `imagePullSecret`, чтобы кластер мог тянуть образ.

Основные шаги pipeline:
1. `pytest -q`
2. `docker build -t ghcr.io/<owner>/spam-api:{sha|latest} .`
3. `docker push` обоих тегов в GHCR
4. `kubectl set image deployment/spam-api spam-api=ghcr.io/<owner>/spam-api:latest && kubectl rollout status`

Проверка: пушьте в feature-ветку → PR → merge в `main`; в логах Actions увидите публикацию образа и успешный rollout деплоймента `spam-api`.

### Docker Compose (API + Prometheus + Grafana)

1. Поднять стек:  
   `docker compose -f docker-compose.lab11.yaml up --build -d`  
   (опционально `SIMULATED_LATENCY_SEC=0.5 docker compose -f docker-compose.lab11.yaml up --build -d`, чтобы увидеть длинные запросы).
2. Проверка API/метрик: `curl http://localhost:8080/health`, `curl http://localhost:8080/metrics | head`.
3. Prometheus UI: http://localhost:9090 — таргет `spam-api` должен быть `UP`; правила алертов в `prometheus_rules.yml` (видны в разделе **Alerts**, без Alertmanager только статус внутри UI).
4. Grafana UI: http://localhost:3000 (admin/admin). Datasource и дашборд «Spam API / Lab11 — Spam API Monitoring» подтягиваются автоматически из `grafana/provisioning/`.

### Генерация нагрузки для графиков

- Много запросов: `hey -n 200 -c 20 -m POST -H "Content-Type: application/json" -d '{"text":"get rich quick"}' http://localhost:8080/predict`
- Долгие запросы: запустить compose с `SIMULATED_LATENCY_SEC=0.5` или прогнать серию `for i in {1..30}; do curl -s -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"text":"please respond"}' >/dev/null; done`.
- Смотрите графики: requests/s, p95 latency, средняя вероятность, распределение `prediction_proba_spam`, а также срабатывание правил в Prometheus Alerts.

### Остановка

```
docker compose -f docker-compose.lab11.yaml down -v
```
