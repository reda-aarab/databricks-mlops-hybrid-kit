# Databricks notebook source
# MAGIC %md
# MAGIC # Étape 2 — Train
# MAGIC
# MAGIC Réfère aux kickers deck **`04 · Phase amont — Étape 2`** (Entraîner et enregistrer)
# MAGIC et **`04 · Phase amont — Anatomie de l'artefact`** (pipeline sklearn + wrapper pyfunc).
# MAGIC
# MAGIC 1. Lit `wind_features` (préparée par `01_prep_data.py`)
# MAGIC 2. Fit `Pipeline(StandardScaler + HistGradientBoostingRegressor)` sur les 20 features
# MAGIC 3. Définit le wrapper `WindForecaster(mlflow.pyfunc.PythonModel)` qui :
# MAGIC    - reçoit un **window de 5 mesures brutes** (timestamp + u10/v10/t2m/sp)
# MAGIC    - appelle `features.features_from_window` pour calculer les 20 features
# MAGIC    - délègue à la pipeline sklearn
# MAGIC 4. `mlflow.pyfunc.log_model(python_model=WindForecaster(), code_paths=["features.py"], ...)`
# MAGIC 5. `register_model` dans Unity Catalog **sans poser d'alias** — c'est `03_validate.py`
# MAGIC    qui décide de la promotion `@champion`

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]==2.16.2" scikit-learn==1.5.1 numpy==1.26.4 pandas==2.2.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Copie features.py (qui vit dans le même dossier workspace que ce notebook) vers /tmp,
# puis l'importe. Le chemin /tmp/features.py sera également passé à code_paths dans log_model.
import os, shutil, sys
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
nb_dir = "/Workspace" + os.path.dirname(nb_path)
shutil.copy(f"{nb_dir}/features.py", "/tmp/features.py")
sys.path.insert(0, "/tmp")
import features  # noqa: E402

# COMMAND ----------

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "ml"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.wind_features"
RAW_TABLE = f"{CATALOG}.{SCHEMA}.wind_measurements"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.wind_forecast_model"
# Expérience MLflow rangée dans le home de l'utilisateur courant (portable)
_CURRENT_USER = spark.sql("SELECT current_user() AS u").collect()[0]["u"]
EXPERIMENT_NAME = f"/Users/{_CURRENT_USER}/wind_forecast_demo"

# Contrat de release — versions pinnées installées on-prem
RUNTIME_LOCK = [
    "mlflow-skinny==2.16.2",
    "scikit-learn==1.5.1",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scipy==1.13.1",
]

# COMMAND ----------

# MAGIC %md ## Train la pipeline sklearn sur les features pré-calculées

# COMMAND ----------

df = spark.table(SOURCE_TABLE).orderBy("timestamp").toPandas()
print(f"{len(df):,} lignes lues depuis {SOURCE_TABLE}")

X = df[features.FEATURE_COLS]
y = df["production_mw"]
train_idx, test_idx = list(TimeSeriesSplit(n_splits=5).split(X))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train : {len(X_train):,} · Test : {len(X_test):,}")

pipe = Pipeline(
    steps=[
        ("scaler", ColumnTransformer([("num", StandardScaler(), features.FEATURE_COLS)], remainder="drop")),
        ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=0)),
    ]
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
print(f"RMSE: {rmse:.1f} MW · MAE: {mae:.1f} MW")

# COMMAND ----------

# MAGIC %md ## Wrapper pyfunc — option A
# MAGIC Reçoit un DataFrame de `WINDOW_SIZE` lignes brutes, applique features_from_window
# MAGIC en interne, retourne 1 prédiction.

# COMMAND ----------

PIPELINE_PATH = "/tmp/sklearn_pipeline.joblib"
joblib.dump(pipe, PIPELINE_PATH)


class WindForecaster(mlflow.pyfunc.PythonModel):
    """Wrapper qui embarque le feature engineering custom.

    Input attendu : DataFrame avec WINDOW_SIZE lignes ordonnées croissantes,
    colonnes ['timestamp', 'u10', 'v10', 't2m', 'sp'].
    Output : array numpy de longueur 1 — la prédiction pour le timestamp le plus récent.
    """

    def load_context(self, context):
        import joblib
        self.pipeline = joblib.load(context.artifacts["sklearn_pipeline"])
        # `features` est bundlé via code_paths — importable au runtime
        import features as _features
        self._features = _features

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        feature_row = self._features.features_from_window(model_input)
        X = pd.DataFrame([feature_row])[self._features.FEATURE_COLS]
        return self.pipeline.predict(X)


# Exemple d'input pour la signature MLflow + input_example
raw = spark.table(RAW_TABLE).orderBy("timestamp").limit(features.WINDOW_SIZE).toPandas()
example_window = raw[["timestamp", "u10", "v10", "t2m", "sp"]].copy()
# La colonne timestamp doit être sérialisable côté MLflow ; on la passe en string ISO
example_window["timestamp"] = pd.to_datetime(example_window["timestamp"]).astype(str)

# COMMAND ----------

# MAGIC %md ## log_model + register (pas d'alias ici — c'est 03_validate qui décide)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="wind_forecast_model_pyfunc") as run:
    mlflow.log_metric("test_rmse_mw", rmse)
    mlflow.log_metric("test_mae_mw", mae)

    # Pour tester localement la signature avant log_model
    wrapper = WindForecaster()

    class _LocalCtx:
        artifacts = {"sklearn_pipeline": PIPELINE_PATH}

    wrapper.load_context(_LocalCtx())
    sample_pred = wrapper.predict(None, example_window)
    print(f"Test wrapper en local : prédiction = {float(sample_pred[0]):.2f} MW")

    signature = infer_signature(example_window, sample_pred)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WindForecaster(),
        artifacts={"sklearn_pipeline": PIPELINE_PATH},
        code_paths=["/tmp/features.py"],
        signature=signature,
        input_example=example_window,
        pip_requirements=RUNTIME_LOCK,
    )
    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"log_model OK -> {model_uri}")

# COMMAND ----------

result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"\nRegistered : {MODEL_NAME} v{result.version}")
print("Alias @champion NON posé — étape 03_validate.py validera puis promouvera.")
print(f"\nRécap du contrat de release :")
print(f"  - input            : DataFrame {features.WINDOW_SIZE} lignes (timestamp + 4 raw)")
print(f"  - feature engineering : embarqué via code_paths=features.py")
print(f"  - sklearn pipeline : ColumnTransformer + StandardScaler + HGBR")
print(f"  - signature        : {len(features.RAW_COLS) + 1} colonnes en entrée, 1 prédiction en sortie")
print(f"  - pip_requirements : {len(RUNTIME_LOCK)} lignes pinnées (identique à venv on-prem)")
