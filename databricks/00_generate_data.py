# Databricks notebook source
# MAGIC %md
# MAGIC # Étape 0 — Generate Data
# MAGIC
# MAGIC **Hors flow de démo réelle.** En conditions de production, cette table est alimentée par le
# MAGIC pipeline d'ingestion existant (Lakeflow/Spark Structured Streaming → ADLS →
# MAGIC Delta) à partir de ARPEGE + télémétrie parcs. Ici on synthétise pour avoir une
# MAGIC source.
# MAGIC
# MAGIC Écrit `workspace.ml.wind_measurements` :
# MAGIC - 30 jours × cadence 16 min ≈ 2 700 lignes
# MAGIC - colonnes : `timestamp`, `u10`, `v10`, `t2m`, `sp`, `production_mw`
# MAGIC - **Mesures brutes uniquement** — pas de lag ni cyclique (ces features sont
# MAGIC   ajoutées en `01_prep_data.py` via le module partagé `features.py`)
# MAGIC
# MAGIC Variables (convention ARPEGE / ERA5) :
# MAGIC - `u10` : vent zonal à 10 m (m/s, positif vers l'est)
# MAGIC - `v10` : vent méridien à 10 m (m/s, positif vers le nord)
# MAGIC - `t2m` : température à 2 m (Kelvin)
# MAGIC - `sp`  : pression au sol (Pascal)

# COMMAND ----------

# MAGIC %pip install -q numpy==1.26.4 pandas==2.2.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import math

import numpy as np
import pandas as pd

CATALOG = "workspace"
SCHEMA = "ml"
TABLE = f"{CATALOG}.{SCHEMA}.wind_measurements"

N_DAYS = 30
CADENCE_MIN = 16
SEED = 20260512
CAPACITY_MW = 1000.0  # capacité totale France simplifiée

# COMMAND ----------

rng = np.random.default_rng(SEED)
n = N_DAYS * 24 * 60 // CADENCE_MIN
ts = pd.date_range("2026-04-01", periods=n, freq=f"{CADENCE_MIN}min", tz="UTC")
dt = CADENCE_MIN / 60.0


def ou(mean: float, std: float, n: int) -> np.ndarray:
    """Ornstein-Uhlenbeck mean-reverting process pour la cinématique du vent."""
    x = np.zeros(n)
    x[0] = mean + rng.normal(0, std)
    for t in range(1, n):
        x[t] = x[t - 1] + 0.4 * (mean - x[t - 1]) * dt + std * math.sqrt(dt) * rng.normal()
    return x


u10 = ou(3.0, 4.5, n)
v10 = ou(1.0, 4.5, n)
t2m = (
    285.0
    + 5.0 * np.cos(2 * np.pi * (ts.dayofyear - 200) / 365.25)
    + 4.0 * np.cos(2 * np.pi * (ts.hour - 14) / 24)
    + rng.normal(0, 0.8, n)
)
sp = 101325.0 + rng.normal(0, 50, n).cumsum()

# Power curve France : cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s
wind = np.sqrt(u10**2 + v10**2)
prod = np.where(
    (wind >= 3) & (wind < 12), CAPACITY_MW * ((wind - 3) / 9) ** 3,
    np.where((wind >= 12) & (wind < 25), CAPACITY_MW, 0.0),
)
production_mw = np.clip(prod * 0.30 + rng.normal(0, 20, n), 0, CAPACITY_MW)

df = pd.DataFrame(
    {
        "timestamp": ts,
        "u10": u10,
        "v10": v10,
        "t2m": t2m,
        "sp": sp,
        "production_mw": production_mw,
    }
)
print(f"{len(df):,} lignes générées · production moyenne : {df['production_mw'].mean():.1f} MW")

# COMMAND ----------

# MAGIC %md ## Écriture Delta

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
(
    spark.createDataFrame(df)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE)
)
print(f"Écrit : {TABLE}")
display(spark.table(TABLE).limit(10))
