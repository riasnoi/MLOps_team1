from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# ==== те же фичи, что в src/preprocess.py ====
import re
import regex as re2
from bs4 import BeautifulSoup
import tldextract

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
DIGITS_RE = re.compile(r"\d")
UPPER_RE = re2.compile(r"\p{Lu}")

FEATURE_COLUMNS = [
    "char_len",
    "word_len",
    "num_digits",
    "num_urls",
    "num_domains",
    "upper_ratio",
]

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    s = s.lower()
    s = URL_RE.sub(" <url> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_urls(s: str) -> int:
    return len(URL_RE.findall(s)) if isinstance(s, str) else 0

def count_digits(s: str) -> int:
    return len(DIGITS_RE.findall(s)) if isinstance(s, str) else 0

def upper_ratio(s: str) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    upp = len(UPPER_RE.findall(s))
    return float(upp) / max(len(s), 1)

def num_domains(s: str) -> int:
    if not isinstance(s, str):
        return 0
    urls = URL_RE.findall(s)
    domains = set()
    for u in urls:
        ts = tldextract.extract(u)
        dom = ".".join([p for p in [ts.domain, ts.suffix] if p])
        if dom:
            domains.add(dom)
    return len(domains)

def build_features(text: str) -> dict[str, float | int]:
    text_clean = clean_text(text)
    return {
        "char_len": len(text_clean),
        "word_len": len(text_clean.split()),
        "num_digits": count_digits(text),
        "num_urls": count_urls(text),
        "num_domains": num_domains(text),
        "upper_ratio": upper_ratio(text),
    }

# ==== модель и приложение ====
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model_store"))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "random_forest.joblib")
try:
    SIMULATED_LATENCY_SEC = float(os.environ.get("SIMULATED_LATENCY_SEC", "0"))
except ValueError:
    SIMULATED_LATENCY_SEC = 0.0
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_model = None
_model_path: Optional[Path] = None

REQUEST_COUNT = Counter(
    "request_count",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "http_status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
PREDICTION_DISTRIBUTION = Histogram(
    "prediction_proba_spam",
    "Distribution of spam probability scores",
    buckets=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)

def load_model() -> bool:
    global _model, _model_path
    path = MODEL_DIR / MODEL_FILENAME
    if path.exists():
        _model = joblib.load(path)
        _model_path = path
        return True
    _model = None
    _model_path = None
    return False

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    proba_spam: float
    model_path: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: выполняется при запуске приложения
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    load_model()
    yield
    # Shutdown: здесь можно добавить код очистки ресурсов при необходимости

app = FastAPI(title="SMS Spam API (Lab6)", lifespan=lifespan)

if STATIC_DIR.exists():
    # Раздаём небольшую React-страницу без сборки
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception:
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(request.method, request.url.path, status_code).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path, status_code).observe(latency)
        raise

    latency = time.perf_counter() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path, status_code).observe(latency)
    return response

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(_model_path) if _model_path else None,
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    if _model is None:
        return PredictOut(label="unknown", proba_spam=0.0, model_path=None)

    if SIMULATED_LATENCY_SEC > 0:
        time.sleep(SIMULATED_LATENCY_SEC)

    feats = build_features(inp.text)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    proba = float(_model.predict_proba(X)[0, 1])
    PREDICTION_DISTRIBUTION.observe(proba)
    label = "spam" if proba >= 0.5 else "ham"
    return PredictOut(label=label, proba_spam=proba, model_path=str(_model_path) if _model_path else None)

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
