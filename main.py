import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "pipeline.joblib"
FEATURE_COLUMNS = ["age", "bmi", "children", "sex", "smoker", "region"]
logger = logging.getLogger(__name__)


class InsuranceInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    age: int = Field(ge=0, le=120)
    sex: Literal["male", "female"]
    bmi: float = Field(gt=0, le=100)
    children: int = Field(ge=0, le=20)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[InsuranceInput] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    predictions: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = joblib.load(MODEL_PATH)
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError("El archivo debe contener un Pipeline.")
    if list(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise RuntimeError("Las columnas del pipeline no coinciden.")

    app.state.pipeline = pipeline
    yield
    del app.state.pipeline


app = FastAPI(
    title="Taller 3 - API de prediccion",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    frame = pd.DataFrame(
        [instance.model_dump() for instance in request.instances],
        columns=FEATURE_COLUMNS,
    )

    try:
        predictions = app.state.pipeline.predict(frame)
        if not np.isfinite(predictions).all():
            raise ValueError("El modelo produjo valores no finitos.")
        return PredictionResponse(predictions=predictions.tolist())
    except Exception:
        logger.exception("Error al generar las predicciones")
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar las predicciones.",
        )
