# План презентации: SMS Spam Detector (end-to-end MLOps)

## 1. Титульник
- Название: «SMS Spam Detector — учебный MLOps стек»
- Подзаголовок: «От сырых SMS до продового API с мониторингом»
- Команда/контакты, дата/мероприятие.

## 2. Команда
- Роли: Data/ML, MLOps/Infra, Backend/API, Monitoring.
- Кто что делал (поимённо) — для устного рассказа по минуте.

## 3. Проблема (человеческим языком)
- Люди тонут в SMS/мессенджер-спаме: фишинг, навязчивая реклама, ложные «доставка/банк».
- Риски: переходы по вредоносным ссылкам, потеря денег/данных, шум, падение доверия к бренду.
- Требования: быстрый фильтр, минимум ложных тревог, прозрачность и стабильность.

## 4. Постановка задачи
- Классифицировать сообщения на spam/ham с ROC-AUC ≥ 0.9 и низкой задержкой инференса.
- Сделать воспроизводимый пайплайн: данные → фичи → обучение → проверка порога → регистрация.
- Согласованность оффлайн/онлайн фичей, API `/predict`, метрики, мониторинг, CI/CD.

## 5. Данные и предобработка
- Источник: sms_spam_collection (HF), fallback TSV; два класса (spam/ham).
- Очистка: strip HTML, lowercase, нормализация URL → `<url>`, схлопывание пробелов.
- Признаки: char_len, word_len, num_digits, num_urls, num_domains, upper_ratio.
- Разметка: target 0/1, sms_id как entity, event_timestamp для Feast.
- Артефакты: raw `data/raw/sms_spam.csv`; processed `data/processed/{csv,parquet}`.

## 6. Модель
- RandomForestClassifier на ручных фичах (быстро, интерпретируемо).
- Порог приёмки: ROC-AUC ≥ 0.9 (скрипт register.py).
- Возможные альтернативы (упомянуть): логрег, бустинг, TF-IDF + логрег.

## 7. Схема/архитектура
- Пайплайн: download → preprocess → feast apply/materialize → train → evaluate → register.
- Feature Store (Feast): согласованные оффлайн/онлайн фичи.
- API: FastAPI `/predict`, `/health`, `/metrics`; UI `/ui`.
- Мониторинг: Prometheus метрики (`request_count`, `latency`, `prediction_proba_spam`) + Grafana.
- Доставка: Docker образ, CI/CD (pytest → build/push GHCR → kubectl rollout), K8s (Deployment/Service/HPA).

## 8. Преимущества, новизна, актуальность
- Репродьюсибельность фичей и данных (Feast, скрипты).
- Контроль качества перед выкладкой (evaluate + register с порогом).
- Наблюдаемость: метрики и дашборд «из коробки».
- Простота для обучения и расширения: ручные признаки, заменяемая модель, готовая обвязка.

## 9. Демо / фронт
- UI `/ui`: ввод текста → label + P(spam), история запросов.
- Метрики: `/metrics` и Grafana (показать ключевые графики).

## 10. Итоги и next steps
- Что сделано: end-to-end стек, модель, API, мониторинг, CI/CD.
- Что дальше: улучшить фичи/модель, A/B, alertmanager, более строгий drift-check.
- Ссылки: репозиторий, дашборд, контакт.

## Примечание для выступающих
- Каждый участник говорит ~1 мин про свой вклад (слайд «Команда» или финальный).
- На слайдах 3–7 держать фокус на «почему так» и «как защищаем качество».

## Дополнительные слайды (если нужно выделить отдельно)
- Feature Store (Feast): зачем (согласованность оффлайн/онлайн, воспроизводимость), сущности/feature views, materialize, как API считает те же фичи (нет train/serve skew).
- CI/CD: пайплайн GitHub Actions (pytest → build/push GHCR → kubectl rollout), секреты, проверка порога качества перед выкладкой, образ + K8s манифесты (Deployment/Service/HPA).
- Актуальность/бизнес-ценность: почему фильтрация спама важна (фишинг, репутация, шум), требования к latency/стабильности, как стек решает эти боли.
- Преимущества/отличия: репродьюсибельный пайплайн, готовые метрики/дашборд, простые интерпретируемые фичи, легкая замена модели без ломки контрактов.

## Отдельный слайд: Деплоймент и CI/CD (Kubernetes)
- CI: GitHub Actions (`.github/workflows/deploy.yml`) — шаги: `pytest -q` → build Docker → push в GHCR → подготовка kubeconfig (секрет `KUBE_CONFIG_DATA`) → `kubectl set image` + `rollout status`.
- Образ: `ghcr.io/<owner>/spam-api:{sha,latest}`, собирается из Dockerfile (FastAPI + модель из `model_store`).
- CD: автоматический rollout в кластер после пуша в `main`; откат через `kubectl rollout undo` при проблемах.
- Kubernetes: манифесты `k8s/deployment.yaml` (Deployment, readiness/liveness probes, ресурсные лимиты), `k8s/service.yaml` (NodePort 30080), `k8s/hpa.yaml` (автоскейл по CPU).
- Среда наблюдаемости: `/metrics` для Prometheus, Grafana дашборд; можно включить Alertmanager в проде.
