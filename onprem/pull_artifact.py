"""Pull de l'artefact MLflow @champion depuis Unity Catalog vers ./local_model/.

Référence l'API MLflow registry directement, avec authentification via le profile
databricks-cli configuré localement. Pour la production OIV, ce pattern serait
remplacé par un export cloud-side vers un stockage intermédiaire (Nexus, ADLS+SAS,
ACR, NFS) puis un pull depuis ce stockage.

Pré-requis :
    1. Installer databricks-cli : pip install databricks-cli
    2. S'authentifier : databricks auth login --host https://<votre-workspace>.cloud.databricks.com
    3. Adapter MODEL_NAME ci-dessous au catalog/schema/modèle de votre workspace

Usage :
    python pull_artifact.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import mlflow

# ============================================================
# CONFIGURATION — à éditer avant la première exécution
# ============================================================

# Nom complet du modèle dans Unity Catalog (catalog.schema.model_name)
# Sur Free Edition, le catalog par défaut s'appelle souvent "workspace"
MODEL_NAME = "workspace.ml.wind_forecast_model"
ALIAS = "champion"

# Profile databricks-cli à utiliser (configuré via `databricks auth login`)
DATABRICKS_PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")

# Dossier local où l'artefact sera téléchargé (sera créé / écrasé à chaque run)
LOCAL_CACHE = Path("./local_model")

# ============================================================

os.environ["DATABRICKS_CONFIG_PROFILE"] = DATABRICKS_PROFILE
os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

if LOCAL_CACHE.exists():
    shutil.rmtree(LOCAL_CACHE)
LOCAL_CACHE.mkdir(parents=True)

uri = f"models:/{MODEL_NAME}@{ALIAS}"
print(f"Pull artefact : {uri} -> {LOCAL_CACHE}")
dst = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(LOCAL_CACHE))

contents = sorted(p.name for p in Path(dst).iterdir())
print(f"\nOK — artefact local : {dst}")
print("Contenu :")
for name in contents:
    print(f"  - {name}")

print("\nProchaine étape :")
print("  Linux / macOS : bash run_container.sh")
print("  Windows       : .\\run_container.ps1")
