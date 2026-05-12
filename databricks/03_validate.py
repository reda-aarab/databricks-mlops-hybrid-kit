# Databricks notebook source
# MAGIC %md
# MAGIC # Étape 3 — Validate
# MAGIC
# MAGIC Validation pré-promotion. Matérialise la **bascule atomique** mentionnée dans le
# MAGIC kicker deck `04 · Phase amont — Étape 2` : l'alias `@champion` n'est posé qu'après
# MAGIC validation.
# MAGIC
# MAGIC Tests :
# MAGIC 1. **Sanity predict** — predict sur un window connu, output numérique non-NaN, dans [0, capacity]
# MAGIC 2. **Signature compatible** — entrée matche `[timestamp, u10, v10, t2m, sp]`, sortie 1 valeur
# MAGIC 3. **RMSE sous seuil** — métrique loguée par 02_train < 50 MW
# MAGIC 4. **Cohérence batch vs streaming** — la feature vector calculée par `features.features_from_batch`
# MAGIC    (ce qu'utilise le training) doit matcher celle de `features.features_from_window` (ce qu'utilise
# MAGIC    le wrapper) pour un même point. Anti train-serve skew.
# MAGIC
# MAGIC Si tout passe → `set_registered_model_alias("champion", v)`.
# MAGIC Sinon → laisse la version sans alias. Trace les échecs.

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

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

CATALOG = "workspace"
SCHEMA = "ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wind_forecast_model"
RAW_TABLE = f"{CATALOG}.{SCHEMA}.wind_measurements"
FEAT_TABLE = f"{CATALOG}.{SCHEMA}.wind_features"

RMSE_THRESHOLD_MW = 50.0
CAPACITY_MW = 1000.0
SKEW_TOLERANCE = 1e-9

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md ## Récupère la dernière version (toutes versions confondues, READY)

# COMMAND ----------

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
ready_versions = [v for v in versions if v.status == "READY"]
ready_versions.sort(key=lambda v: int(v.version), reverse=True)
if not ready_versions:
    raise RuntimeError(f"Aucune version READY pour {MODEL_NAME}")
latest = ready_versions[0]
print(f"Cible de validation : {MODEL_NAME} v{latest.version} (run {latest.run_id[:8]})")

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest.version}")

# Window de test reconstruit depuis la table raw
raw_sample = spark.table(RAW_TABLE).orderBy("timestamp").limit(features.WINDOW_SIZE).toPandas()
window = raw_sample[["timestamp", "u10", "v10", "t2m", "sp"]].copy()
window["timestamp"] = pd.to_datetime(window["timestamp"]).astype(str)

# COMMAND ----------

# MAGIC %md ## Test 1 — Sanity predict

# COMMAND ----------

results: dict[str, tuple[bool, str]] = {}

try:
    pred = model.predict(window)
    val = float(pred[0])
    ok = (not np.isnan(val)) and (0.0 <= val <= CAPACITY_MW)
    results["sanity_predict"] = (ok, f"prediction = {val:.2f} MW (attendu dans [0, {CAPACITY_MW}])")
except Exception as e:
    results["sanity_predict"] = (False, f"exception : {e}")

# COMMAND ----------

# MAGIC %md ## Test 2 — Signature compatible

# COMMAND ----------

sig = model.metadata.get_input_schema()
expected_cols = {"timestamp", "u10", "v10", "t2m", "sp"}
got_cols = {c.name for c in sig.inputs} if sig else set()
sig_ok = expected_cols == got_cols
results["signature"] = (sig_ok, f"input cols attendues={sorted(expected_cols)} got={sorted(got_cols)}")

# COMMAND ----------

# MAGIC %md ## Test 3 — RMSE sous seuil

# COMMAND ----------

run = client.get_run(latest.run_id)
rmse_metric = run.data.metrics.get("test_rmse_mw")
if rmse_metric is None:
    results["rmse"] = (False, "test_rmse_mw absent du run")
else:
    ok = rmse_metric < RMSE_THRESHOLD_MW
    results["rmse"] = (ok, f"RMSE={rmse_metric:.2f} MW seuil<{RMSE_THRESHOLD_MW}")

# COMMAND ----------

# MAGIC %md ## Test 4 — Cohérence batch vs streaming
# MAGIC On prend une ligne de `wind_features` (calcul batch via features_from_batch),
# MAGIC on reconstruit le window correspondant depuis `wind_measurements`, on appelle
# MAGIC features_from_window, on vérifie que les 20 features matchent.

# COMMAND ----------

feat_df = spark.table(FEAT_TABLE).orderBy("timestamp").limit(features.WINDOW_SIZE + 5).toPandas()
raw_df = spark.table(RAW_TABLE).orderBy("timestamp").limit(features.WINDOW_SIZE * 3).toPandas()

# Synchronisation par timestamp — la table de features a déjà sauté les premières
# lignes (NaN après shift), donc on ne peut pas matcher par position d'index.
feat_df["_ts"] = pd.to_datetime(feat_df["timestamp"])
raw_df["_ts"] = pd.to_datetime(raw_df["timestamp"])
sample_ts = feat_df["_ts"].iloc[0]
batch_row = feat_df.iloc[0]

raw_idx_candidates = raw_df.index[raw_df["_ts"] == sample_ts]
if len(raw_idx_candidates) == 0 or raw_idx_candidates[0] < features.WINDOW_SIZE - 1:
    raise RuntimeError(f"Timestamp {sample_ts} introuvable dans raw_df ou trop tôt pour reconstruire le window")
raw_idx = int(raw_idx_candidates[0])
window_for_streaming = raw_df.iloc[
    raw_idx - features.WINDOW_SIZE + 1 : raw_idx + 1
][["timestamp", "u10", "v10", "t2m", "sp"]].copy().reset_index(drop=True)
streaming_features = features.features_from_window(window_for_streaming)

diffs = {}
for col in features.FEATURE_COLS:
    batch_val = float(batch_row[col])
    stream_val = float(streaming_features[col])
    if abs(batch_val - stream_val) > SKEW_TOLERANCE:
        diffs[col] = (batch_val, stream_val)

skew_ok = len(diffs) == 0
results["batch_vs_streaming"] = (
    skew_ok,
    "features identiques" if skew_ok else f"divergences : {diffs}",
)

# COMMAND ----------

# MAGIC %md ## Décision finale

# COMMAND ----------

all_ok = all(ok for ok, _ in results.values())
print("=" * 70)
for name, (ok, msg) in results.items():
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark}  {name:25s}  {msg}")
print("=" * 70)

if all_ok:
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    print(f"\nPROMOTION : {MODEL_NAME} v{latest.version} -> @champion")
else:
    print(f"\nVALIDATION ÉCHOUÉE : alias @champion inchangé pour {MODEL_NAME}")
    print("La version v" + latest.version + " reste sans alias (shadow).")

# COMMAND ----------

import json as _json
_summary = {
    "version": latest.version,
    "all_ok": all_ok,
    "results": {k: {"ok": ok, "msg": msg} for k, (ok, msg) in results.items()},
}
dbutils.notebook.exit(_json.dumps(_summary))
