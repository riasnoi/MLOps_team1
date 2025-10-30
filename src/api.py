from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

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

app = FastAPI(title="SMS Spam API (Lab6)")

_model = None
_model_path: Optional[Path] = None

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

@app.on_event("startup")
def startup_event() -> None:
    # Для локального запуска MODEL_DIR может быть относительным и должен существовать
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    load_model()

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

    feats = build_features(inp.text)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    proba = float(_model.predict_proba(X)[0, 1])
    label = "spam" if proba >= 0.5 else "ham"
    return PredictOut(label=label, proba_spam=proba, model_path=str(_model_path) if _model_path else None)
