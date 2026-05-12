"""Feature engineering pour la prévision éolienne — source unique de vérité.

Importé en batch côté training par `01_prep_data.py` et en streaming côté
inférence par le wrapper pyfunc (`02_train.py:WindForecaster`). Les deux
chemins DOIVENT produire des features identiques pour un même point —
contrôlé par `03_validate.py`.

Convention météo ARPEGE :
    u10 = composante zonale du vent à 10 m (m/s, positif vers l'est)
    v10 = composante méridienne du vent à 10 m (m/s, positif vers le nord)
    t2m = température à 2 m (Kelvin)
    sp  = pression de surface (Pascal)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (1, 2, 3, 4)
RAW_COLS: tuple[str, ...] = ("u10", "v10", "t2m", "sp")
LAG_COLS_FOR: tuple[str, ...] = ("u10", "v10", "t2m")  # pas de lag sur sp

FEATURE_COLS: list[str] = (
    list(RAW_COLS)
    + [f"{c}_lag{lag}" for c in LAG_COLS_FOR for lag in LAGS]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)
WINDOW_SIZE: int = max(LAGS) + 1  # 5 : 4 lags + le pas courant


def cyclical_from_timestamp(ts: pd.Timestamp) -> dict[str, float]:
    """Encode l'heure de la journée et le jour de l'année comme (sin, cos)
    de manière à préserver la cyclicité (23h <-> 1h sont contigus)."""
    hour = ts.hour + ts.minute / 60
    doy = ts.dayofyear
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
    }


def features_from_window(window_df: pd.DataFrame) -> dict[str, float]:
    """À partir d'une fenêtre de `WINDOW_SIZE` mesures brutes ordonnées
    croissantes, retourne les 20 features pour la dernière ligne (le présent).

    `window_df` doit contenir les colonnes ['timestamp', 'u10', 'v10', 't2m', 'sp'].
    """
    if len(window_df) != WINDOW_SIZE:
        raise ValueError(
            f"window doit contenir exactement {WINDOW_SIZE} lignes, reçu {len(window_df)}"
        )
    w = window_df.sort_values("timestamp").reset_index(drop=True)
    current = w.iloc[-1]

    out: dict[str, float] = {col: float(current[col]) for col in RAW_COLS}
    for col in LAG_COLS_FOR:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = float(w.iloc[-1 - lag][col])
    out.update(cyclical_from_timestamp(pd.Timestamp(current["timestamp"])))
    return out


def features_from_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée pour 01_prep_data — applique les shifts en batch
    pour calculer les lags, puis les cycliques. Les premières `max(LAGS)`
    lignes sortent avec NaN sur les lags ; à dropper par l'appelant.

    `df` doit contenir 'timestamp', 'u10', 'v10', 't2m', 'sp' (au minimum).
    """
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    for col in LAG_COLS_FOR:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    ts = pd.to_datetime(out["timestamp"])
    hr = ts.dt.hour + ts.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    out["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    return out


def make_test_window(
    timestamps: Iterable[pd.Timestamp],
    u10: Iterable[float],
    v10: Iterable[float],
    t2m: Iterable[float],
    sp: Iterable[float],
) -> pd.DataFrame:
    """Helper pour les tests : construit un window à partir de listes parallèles."""
    return pd.DataFrame(
        {"timestamp": list(timestamps), "u10": list(u10), "v10": list(v10), "t2m": list(t2m), "sp": list(sp)}
    )
