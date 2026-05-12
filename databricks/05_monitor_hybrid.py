# Databricks notebook source
# MAGIC %md
# MAGIC # Étape 5 — Monitor (Hybrid)
# MAGIC
# MAGIC Réfère au kicker deck **`08 · Surveillance — Étape 5`** : *"Surveiller un modèle inféré
# MAGIC on-prem"*. Pattern Annexe C du storyline : la vérité terrain (production éolienne réelle)
# MAGIC arrive **déjà côté cloud** via le pipeline d'ingestion. L'évaluation Champion vs
# MAGIC Challenger se fait donc entièrement côté Databricks, **sans aucune télémétrie remontée
# MAGIC du on-prem**.
# MAGIC
# MAGIC Ce notebook :
# MAGIC 1. Charge `@champion` et — s'il existe — `@challenger`
# MAGIC 2. Sur la fenêtre des 7 derniers jours de `wind_measurements`, construit chaque
# MAGIC    window de 5 timestamps consécutifs, appelle les deux modèles
# MAGIC 3. Compare RMSE vs vérité terrain (`production_mw` observée)
# MAGIC 4. Append une ligne dans `monitoring_runs`` (table Delta)
# MAGIC 5. Affiche le delta — décision de promotion laissée à un humain (ou à un job downstream)

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]==2.16.2" scikit-learn==1.5.1 numpy==1.26.4 pandas==2.2.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, shutil, sys
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
nb_dir = "/Workspace" + os.path.dirname(nb_path)
shutil.copy(f"{nb_dir}/features.py", "/tmp/features.py")
sys.path.insert(0, "/tmp")
import features  # noqa: E402

# COMMAND ----------

from datetime import datetime, timedelta, timezone

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType, DoubleType, LongType

CATALOG = "workspace"
SCHEMA = "ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wind_forecast_model"
RAW_TABLE = f"{CATALOG}.{SCHEMA}.wind_measurements"
MONITORING_TABLE = f"{CATALOG}.{SCHEMA}.monitoring_runs"

LOOKBACK_DAYS = 7

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md ## Récupération des alias

# COMMAND ----------

def alias_or_none(alias: str):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception as e:
        print(f"@{alias} introuvable : {e}")
        return None


champ_mv = alias_or_none("champion")
chall_mv = alias_or_none("challenger")
if champ_mv is None:
    raise RuntimeError(f"@champion manquant pour {MODEL_NAME} — rien à monitorer.")
print(f"Champion   : v{champ_mv.version}")
print(f"Challenger : v{chall_mv.version}" if chall_mv else "Challenger : aucun")

# COMMAND ----------

# MAGIC %md ## Vérité terrain — 7 derniers jours

# COMMAND ----------

cutoff = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
raw_pd = (
    spark.table(RAW_TABLE)
    .filter(F.col("timestamp") >= F.lit(cutoff))
    .orderBy("timestamp")
    .toPandas()
)
if len(raw_pd) <= features.WINDOW_SIZE:
    # Données générées dans le passé — fallback : prendre les derniers WINDOW_SIZE+200 timestamps
    raw_pd = spark.table(RAW_TABLE).orderBy(F.col("timestamp").desc()).limit(features.WINDOW_SIZE + 200).toPandas()
    raw_pd = raw_pd.sort_values("timestamp").reset_index(drop=True)
print(f"Fenêtre d'évaluation : {len(raw_pd):,} timestamps")

# COMMAND ----------

# MAGIC %md ## Construction des windows et évaluation

# COMMAND ----------

def evaluate(version: str) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (y_true, y_pred) sur tous les windows de la fenêtre d'évaluation."""
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    y_true, y_pred = [], []
    for i in range(features.WINDOW_SIZE - 1, len(raw_pd)):
        window = raw_pd.iloc[i - features.WINDOW_SIZE + 1 : i + 1][
            ["timestamp", "u10", "v10", "t2m", "sp"]
        ].copy()
        window["timestamp"] = pd.to_datetime(window["timestamp"]).astype(str)
        pred = float(model.predict(window)[0])
        y_pred.append(pred)
        y_true.append(float(raw_pd.iloc[i]["production_mw"]))
    return np.asarray(y_true), np.asarray(y_pred)


print(f"Évaluation @champion (v{champ_mv.version})...")
y_true_c, y_pred_c = evaluate(champ_mv.version)
rmse_champ = float(np.sqrt(np.mean((y_pred_c - y_true_c) ** 2)))
mae_champ = float(np.mean(np.abs(y_pred_c - y_true_c)))
print(f"  RMSE: {rmse_champ:.2f} MW · MAE: {mae_champ:.2f} MW · n={len(y_true_c)}")

if chall_mv is not None:
    print(f"Évaluation @challenger (v{chall_mv.version})...")
    _, y_pred_ch = evaluate(chall_mv.version)
    rmse_chall = float(np.sqrt(np.mean((y_pred_ch - y_true_c) ** 2)))
    mae_chall = float(np.mean(np.abs(y_pred_ch - y_true_c)))
    print(f"  RMSE: {rmse_chall:.2f} MW · MAE: {mae_chall:.2f} MW")
else:
    rmse_chall = None
    mae_chall = None

# COMMAND ----------

# MAGIC %md ## Append à la table monitoring

# COMMAND ----------

monitoring_schema = StructType([
    StructField("event_ts", TimestampType(), False),
    StructField("n_predictions", LongType(), False),
    StructField("champion_version", StringType(), False),
    StructField("challenger_version", StringType(), True),
    StructField("rmse_champion_mw", DoubleType(), False),
    StructField("mae_champion_mw", DoubleType(), False),
    StructField("rmse_challenger_mw", DoubleType(), True),
    StructField("mae_challenger_mw", DoubleType(), True),
])

row = (
    datetime.now(tz=timezone.utc),
    int(len(y_true_c)),
    str(champ_mv.version),
    str(chall_mv.version) if chall_mv else None,
    rmse_champ,
    mae_champ,
    rmse_chall,
    mae_chall,
)
df = spark.createDataFrame([row], schema=monitoring_schema)
df.write.mode("append").saveAsTable(MONITORING_TABLE)
print(f"\nLigne ajoutée à {MONITORING_TABLE}")
display(spark.table(MONITORING_TABLE).orderBy(F.col("event_ts").desc()).limit(10))
