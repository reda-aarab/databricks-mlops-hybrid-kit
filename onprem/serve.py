"""Étape 4 — Inférence on-prem 100% locale.

Réfère au kicker deck **`05 · Phase aval — Étape 4 — Inférence locale`**.

Contrat d'interface (option A) — l'app source envoie un **window de mesures brutes** :
    POST /predict
    {
      "window": [
        {"timestamp": "2026-05-11T12:00:00Z", "u10": 8.0, "v10": 4.0, "t2m": 285, "sp": 101325},
        ...
        (5 entrées au total, ordonnées du plus ancien au plus récent)
      ]
    }
    -> {"prediction_mw": 123.45, "for_timestamp": "2026-05-11T13:04:00Z"}

Le **wrapper pyfunc** chargé depuis `./local_model` extrait le pas courant (le 5ᵉ),
calcule les lags depuis les 4 précédents, calcule les cycliques depuis le timestamp courant,
puis appelle la pipeline sklearn. L'on-prem n'a pas à connaître la mécanique des lags.

Aucune dépendance Databricks au runtime — `mlflow.pyfunc.load_model` lit depuis le disque.
"""

from __future__ import annotations

import os
from datetime import datetime

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

WINDOW_SIZE = 5  # doit matcher features.WINDOW_SIZE côté training
# `MODEL_PATH` est posé par le Dockerfile (= /var/cache/model). En venv local : pas posé,
# fallback sur ./local_model. Pas d'autre changement nécessaire pour le conteneur.
LOAD_PATH = os.environ.get("MODEL_PATH", "./local_model")


class WindowEntry(BaseModel):
    timestamp: datetime
    u10: float
    v10: float
    t2m: float
    sp: float


class WindowRequest(BaseModel):
    window: list[WindowEntry] = Field(min_length=WINDOW_SIZE, max_length=WINDOW_SIZE)


class PredictResponse(BaseModel):
    prediction_mw: float
    for_timestamp: datetime


app = FastAPI(
    title="Wind Forecast on-prem",
    description="Inférence éolienne 100% locale — option A (window de mesures brutes).",
    version="2.0.0",
)
model = mlflow.pyfunc.load_model(LOAD_PATH)
print(f"Modèle chargé depuis {LOAD_PATH}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "load_path": LOAD_PATH, "window_size": str(WINDOW_SIZE)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: WindowRequest) -> PredictResponse:
    df = pd.DataFrame([e.model_dump() for e in req.window])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(str)
    # Garantie d'ordre — le wrapper attend window ascendant
    df = df.sort_values("timestamp").reset_index(drop=True)
    try:
        y = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict failed: {e}")
    return PredictResponse(
        prediction_mw=float(y[0]),
        for_timestamp=req.window[-1].timestamp,
    )
