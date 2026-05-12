# Databricks notebook source
# MAGIC %md
# MAGIC # Étape 1 — Prep Data
# MAGIC
# MAGIC Lit la table de mesures brutes (`wind_measurements`) et applique le feature
# MAGIC engineering centralisé dans `features.py`. Écrit la table d'entraînement
# MAGIC `wind_features` (20 features + target).
# MAGIC
# MAGIC **Important** : `features.py` est le module **partagé** avec le wrapper pyfunc côté
# MAGIC training (cf. `02_train.py`). Une seule source de vérité pour la formule des lags
# MAGIC et des cycliques — pas de risque de train-serve skew.

# COMMAND ----------

# MAGIC %pip install -q numpy==1.26.4 pandas==2.2.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Copie features.py (qui vit dans le même dossier workspace que ce notebook) vers /tmp,
# puis l'importe. Marche quel que soit l'emplacement du dossier databricks-mlops-hybrid-kit/databricks/.
import os, shutil, sys
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
nb_dir = "/Workspace" + os.path.dirname(nb_path)
shutil.copy(f"{nb_dir}/features.py", "/tmp/features.py")
sys.path.insert(0, "/tmp")
import features  # noqa: E402

# COMMAND ----------

import pandas as pd

CATALOG = "workspace"
SCHEMA = "ml"
SRC_TABLE = f"{CATALOG}.{SCHEMA}.wind_measurements"
DST_TABLE = f"{CATALOG}.{SCHEMA}.wind_features"

# COMMAND ----------

raw = spark.table(SRC_TABLE).orderBy("timestamp").toPandas()
print(f"Lu : {len(raw):,} mesures brutes")

# COMMAND ----------

# MAGIC %md ## Application de `features.features_from_batch`

# COMMAND ----------

featurized = features.features_from_batch(raw).dropna().reset_index(drop=True)
print(f"Featurized : {len(featurized):,} lignes après dropna (premiers {features.WINDOW_SIZE - 1} timesteps rejetés)")
print(f"Colonnes : {list(featurized.columns)}")

# COMMAND ----------

# MAGIC %md ## Écriture Delta

# COMMAND ----------

(
    spark.createDataFrame(featurized)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DST_TABLE)
)
print(f"Écrit : {DST_TABLE}")
display(spark.table(DST_TABLE).limit(10))
