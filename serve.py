import logging
import os
import random
from pathlib import Path
from typing import Optional
import uvicorn

from fastapi import FastAPI

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ml_service")

app = FastAPI(title="ML Service")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.bin")
_model_path: Optional[Path] = None
_model_loaded: bool = False


def load_model() -> bool:
    global _model_path, _model_loaded

    model_path = MODEL_DIR / MODEL_FILENAME
    if model_path.exists():
        _model_path = model_path
        _model_loaded = True
        logger.info("Model loaded from %s", model_path)
    else:
        _model_path = None
        _model_loaded = False
        logger.warning("Model file not found at %s", model_path)
    return _model_loaded


@app.on_event("startup")
async def startup_event() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    load_model()


@app.get("/predict")
async def predict() -> dict:
    prediction = random.choice([0, 3])
    logger.info("Returning stub prediction %s (model_loaded=%s)", prediction, _model_loaded)
    payload = {"prediction": prediction}
    if _model_path:
        payload["model_path"] = str(_model_path)
    return payload


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model_loaded}


if __name__ == "__main__":

    uvicorn.run("serve:app", host="0.0.0.0", port=8080, log_level=LOG_LEVEL.lower())
