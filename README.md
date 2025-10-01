# SMS Spam Detector (MLOps labs 1–3)

Лёгкий и актуальный учебный проект: предобработка SMS/мессенджер сообщений для задач антиспама/антифишинга.  
Работаем через **venv + pip**. DVC — для версионирования данных и пайплайна.

## Быстрый старт

```bash
git clone git@github.com:riasnoi/MLOps_team1.git
cd MLOps_team1

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

dvc init
mkdir -p dvc_remote
dvc remote add -d local_remote dvc_remote
git add . && git commit -m "project scaffold (venv+pip)"

# Сырые данные
python src/download_data.py

# Версионируем
dvc add data/raw/sms_spam.csv
git add data/.gitignore data/raw/sms_spam.csv.dvc
git commit -m "add raw sms dataset via dvc"
dvc push -r local_remote

# DVC stage: предобработка
dvc stage add -n preprocess \
  -d src/preprocess.py \
  -d data/raw/sms_spam.csv \
  -o data/processed/processed.csv \
  python src/preprocess.py

dvc repro
git add dvc.yaml dvc.lock
git commit -m "add preprocess stage"
```