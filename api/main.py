"""API FastAPI de prédiction du churn."""
import json
import logging
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.pkl"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
FRONTEND_DIR = ROOT / "frontend"

FEATURES = [
    "age",
    "anciennete",
    "revenu",
    "nb_reclamations",
    "montant_mensuel",
    "retards_paiement",
]


# ---------------------------------------------------------------------------
# Logging : une ligne JSON par événement (console + fichier lu par Logstash)
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("churn_api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        log.addHandler(console)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            log.addHandler(file_handler)
        except OSError:
            log.warning("Fichier de log indisponible, journalisation console uniquement")
    log.propagate = False
    return log


logger = setup_logging()

# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
else:
    metadata = {"model_name": "Random Forest", "model_version": "1.0.0"}

MODEL_NAME = metadata.get("model_name", "Random Forest")
MODEL_VERSION = metadata.get("model_version", "1.0.0")

logger.info(
    "Model loaded",
    extra={"extra_fields": {"event_type": "model_loaded", "model": MODEL_NAME,
                            "model_version": MODEL_VERSION}},
)

# ---------------------------------------------------------------------------
# Métriques métier (en plus des métriques HTTP de l'Instrumentator)
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total", "Nombre de prédictions réalisées",
    ["result", "model_version"],
)
PREDICTION_ERRORS_TOTAL = Counter(
    "churn_prediction_errors_total", "Nombre d'erreurs de prédiction",
    ["model_version"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_duration_seconds", "Durée d'inférence du modèle",
    ["model_version"],
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Churn Prediction API", version=MODEL_VERSION)


class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[35])
    anciennete: int = Field(..., ge=0, description="Ancienneté en mois", examples=[5])
    revenu: float = Field(..., ge=0, examples=[450000])
    nb_reclamations: int = Field(..., ge=0, examples=[2])
    montant_mensuel: float = Field(..., ge=0, examples=[25000])
    retards_paiement: int = Field(..., ge=0, examples=[2])


@app.get("/")
def root():
    return {"application": "Churn Prediction API", "version": MODEL_VERSION,
            "status": "running"}


@app.get("/health")
def health():
    return {"status": "UP", "model_loaded": model is not None}


@app.get("/model/info")
def model_info():
    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": FEATURES,
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics", {}).get(metadata.get("selected", ""), {}),
    }


@app.post("/predict")
def predict(data: ClientData, request: Request):
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    start = time.perf_counter()

    try:
        # DataFrame avec les noms de colonnes utilisés à l'entraînement
        values = pd.DataFrame([data.model_dump()], columns=FEATURES)
        # Un seul passage dans la forêt : predict() recalculerait les mêmes probabilités
        probability = float(model.predict_proba(values)[0][1])
        prediction = 1 if probability > 0.5 else 0  # même règle que model.predict()
    except Exception as exc:  # noqa: BLE001
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        PREDICTION_ERRORS_TOTAL.labels(MODEL_VERSION).inc()
        logger.error(
            "Prediction error",
            extra={"extra_fields": {
                "event_type": "prediction_error", "request_id": request_id,
                "client_ip": client_ip, "model_version": MODEL_VERSION,
                "duration_ms": duration_ms, "error": str(exc),
            }},
        )
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction") from exc

    duration = time.perf_counter() - start
    duration_ms = round(duration * 1000, 2)
    result = "CHURN" if prediction == 1 else "NO_CHURN"

    PREDICTIONS_TOTAL.labels(result, MODEL_VERSION).inc()
    PREDICTION_LATENCY.labels(MODEL_VERSION).observe(duration)

    logger.info(
        "Prediction done",
        extra={"extra_fields": {
            "event_type": "prediction", "request_id": request_id,
            "client_ip": client_ip, "operation": "predict", "model": MODEL_NAME,
            "model_version": MODEL_VERSION, "prediction": result,
            "probability": round(probability, 4), "response_time_ms": duration_ms,
        }},
    )

    return {
        "request_id": request_id,
        "prediction": result,
        "probability": round(probability, 4),
        "model_version": MODEL_VERSION,
        "duration_ms": duration_ms,
    }


# Interface Web minimale (frontend/index.html) servie sur /ui
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/app", include_in_schema=False)
    def ui_redirect():
        return RedirectResponse(url="/ui/")


Instrumentator().instrument(app).expose(app, include_in_schema=False)
