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
