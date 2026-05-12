# Kit MLOps Hybride — Entraînement Cloud, Inférence On-Prem

Démonstration end-to-end du pattern « entraînement Databricks Cloud, inférence on-prem »,
matérialisé sous forme de **cycle MLOps en 5 étapes**, avec garanties de **continuité de service**
(coupure réseau acceptée), **anti train-serve skew** (formule de feature engineering unique),
et **architecture pull-only** (aucun flux entrant initié depuis le cloud).

Code et mode opératoire conçus pour être rejoués sur **Databricks Free Edition** côté cloud,
et un laptop **Windows ou macOS / Linux** côté on-prem.

---

## Ce que vous allez tester

Le cycle MLOps complet, en 5 étapes :

| # | Étape | Où | Fichier |
|---|---|---|---|
| 0 | Generate (synthétique, hors démo) | Cloud | `databricks/00_generate_data.py` |
| 1 | Prep — feature engineering | Cloud | `databricks/01_prep_data.py` |
| 2 | Train + Register | Cloud | `databricks/02_train.py` |
| 3 | Validate (pré-promotion) | Cloud | `databricks/03_validate.py` |
| 4 | Deploy on-prem | Laptop | `onprem/pull_artifact.py` + `onprem/run_container.*` |
| 5 | Monitor hybride | Cloud | `databricks/05_monitor_hybrid.py` |

Le modèle utilisé est un `HistGradientBoostingRegressor` scikit-learn entraîné sur des données
synthétiques de météo et de production éolienne, prédisant la production agrégée France à 16 min.

---

## Prérequis (15 min de setup une fois)

Voir [PREREQUISITES.md](PREREQUISITES.md) pour les détails. En bref :

- **Compte Databricks Free Edition** : https://www.databricks.com/learn/free-edition
- **Python 3.10** (impératif, pas 3.11+) côté laptop
- **Docker Desktop** côté laptop
- **Databricks CLI** côté laptop
- **Git** côté laptop pour cloner ce kit

---

## Setup en 5 étapes

### 1. Récupérer le kit

```bash
git clone https://github.com/reda-aarab/databricks-mlops-hybrid-kit.git
cd databricks-mlops-hybrid-kit
```

### 2. Importer les notebooks dans Databricks

Dans votre workspace Databricks :
1. Ouvrir l'onglet **Workspace** dans la barre latérale
2. Aller dans votre dossier utilisateur (`/Users/votre.email@example.com/`)
3. Clic droit → **Import** → sélectionner le dossier `databricks/` complet du kit

Vous devez voir 6 fichiers Python apparaître :
`features.py`, `00_generate_data`, `01_prep_data`, `02_train`, `03_validate`, `05_monitor_hybrid`.

### 3. Vérifier le catalog par défaut

Sur **Databricks Free Edition**, le catalog par défaut s'appelle `workspace`. Si vous utilisez
une édition payante, vérifiez le nom de votre catalog dans **Catalog Explorer** et adaptez-le
en haut de chaque notebook (variable `CATALOG`).

Le schema `ml` sera créé automatiquement à la première exécution.

### 4. Authentifier le Databricks CLI localement

```bash
databricks auth login --host https://<votre-workspace>.cloud.databricks.com --profile demo
```

Suivez le flow SSO dans votre navigateur. À la fin, vérifiez :

```bash
databricks current-user me --profile demo
```

### 5. Préparer le venv Python 3.10 local

**macOS / Linux** :
```bash
cd onprem
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell)** :
```powershell
cd onprem
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> ⚠️ **Important** : Python 3.10 est obligatoire (pas 3.11+). Voir [TROUBLESHOOTING.md](TROUBLESHOOTING.md) pour la raison.

---

## Dérouler la démo

### Étape 0 — Générer les données (1 fois, ~30 s)

Dans Databricks : ouvrir `00_generate_data` → **Run all**.

Crée la table `workspace.ml.wind_measurements` avec ~2700 mesures synthétiques.

### Étape 1 — Feature engineering (~30 s)

Ouvrir `01_prep_data` → **Run all**.

Lit la table précédente, applique lags et features cycliques via `features.py`, écrit
`workspace.ml.wind_features`.

### Étape 2 — Entraîner et enregistrer (~90 s)

Ouvrir `02_train` → **Run all**.

Entraîne un `HistGradientBoostingRegressor`, l'emballe dans un wrapper pyfunc, et l'enregistre
dans Unity Catalog **sans alias**.

À la fin, vérifier dans **Catalog Explorer** :
- Le modèle `workspace.ml.wind_forecast_model` existe
- Une nouvelle version est créée, status `READY`
- **Aucun alias n'est posé** encore

### Étape 3 — Valider (~30 s)

Ouvrir `03_validate` → **Run all**.

Exécute 4 contrôles pré-promotion sur la dernière version :
1. Plausibilité de la prédiction
2. Conformité de la signature
3. RMSE sous seuil
4. Parité batch / streaming (anti train-serve skew)

Si tous passent, pose l'alias `@champion` sur la version. Sinon, laisse la version sans alias.

Vérifier dans **Catalog Explorer** : la version récente a maintenant l'alias `@champion`.

### Étape 4a — Pull on-prem (~5 s)

Sur votre laptop :

**macOS / Linux** :
```bash
cd onprem
.venv/bin/python pull_artifact.py
```

**Windows (PowerShell)** :
```powershell
cd onprem
.venv\Scripts\python.exe pull_artifact.py
```

> ⚠️ Avant la première exécution, ouvrir `pull_artifact.py` et adapter la ligne `MODEL_NAME`
> si votre catalog n'est pas `workspace`.

Sortie attendue : `./local_model/` se remplit avec `MLmodel`, `artifacts/`, `code/`, etc.

### Étape 4b — Lancer le container (~30 s première fois, 2 s ensuite)

**macOS / Linux** :
```bash
bash run_container.sh
```

**Windows (PowerShell)** :
```powershell
.\run_container.ps1
```

Sortie attendue :
```
Modèle chargé depuis /var/cache/model
Uvicorn running on http://0.0.0.0:8000
```

### Étape 4c — Tester /predict

Dans un autre terminal :

**macOS / Linux** :
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"window":[
    {"timestamp":"2026-05-11T12:00:00Z","u10":12,"v10":5,"t2m":285,"sp":101325},
    {"timestamp":"2026-05-11T12:16:00Z","u10":12,"v10":5,"t2m":285,"sp":101325},
    {"timestamp":"2026-05-11T12:32:00Z","u10":12,"v10":5,"t2m":285,"sp":101325},
    {"timestamp":"2026-05-11T12:48:00Z","u10":12,"v10":5,"t2m":285,"sp":101325},
    {"timestamp":"2026-05-11T13:04:00Z","u10":12,"v10":5,"t2m":285,"sp":101325}
  ]}'
```

**Windows (PowerShell)** :
```powershell
curl http://localhost:8000/health

$body = @{
  window = @(
    @{ timestamp="2026-05-11T12:00:00Z"; u10=12; v10=5; t2m=285; sp=101325 },
    @{ timestamp="2026-05-11T12:16:00Z"; u10=12; v10=5; t2m=285; sp=101325 },
    @{ timestamp="2026-05-11T12:32:00Z"; u10=12; v10=5; t2m=285; sp=101325 },
    @{ timestamp="2026-05-11T12:48:00Z"; u10=12; v10=5; t2m=285; sp=101325 },
    @{ timestamp="2026-05-11T13:04:00Z"; u10=12; v10=5; t2m=285; sp=101325 }
  )
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType 'application/json'
```

Réponse attendue : un `prediction_mw` entre 250 et 350 (vent fort, proche du rated power).

### Étape 5 — Surveillance hybride (~60 s)

Ouvrir `05_monitor_hybrid` → **Run all**.

Compare les prédictions du modèle `@champion` à la vérité terrain qui arrive déjà côté cloud
via le pipeline d'ingestion. **Aucune télémétrie remontée du on-prem.**

Crée la table `workspace.ml.monitoring_runs` avec RMSE et MAE.

---

## Ce que la démo prouve

| Contrainte du pattern | Comment c'est démontré |
|---|---|
| **Pull-only** | Le laptop initie le pull, le cloud ne push jamais |
| **Données on-prem** | Le payload `/predict` ne sort jamais du laptop |
| **Continuité de service** | Une fois `./local_model/` rempli, `/predict` marche sans Internet |
| **Pas de runtime cloud on-prem** | Le container Docker ne contient que Python + sklearn + FastAPI |
| **Anti train-serve skew** | `features.py` est embarqué dans l'artefact via `code_paths` |
| **Bascule atomique** | L'alias `@champion` est posé/déposé en une seule opération UC |
| **Audit cloud-side sans télémétrie on-prem** | `05_monitor_hybrid` rejoue les prédictions sur la vérité terrain déjà ingérée |

---

## Structure du kit

```
databricks-mlops-hybrid-kit/
├── README.md                          ← ce fichier
├── PREREQUISITES.md                   ← détails installation
├── TROUBLESHOOTING.md                 ← erreurs courantes
├── LICENSE
│
├── databricks/                        ← à importer dans votre workspace
│   ├── features.py                    ← module partagé (anti train-serve skew)
│   ├── 00_generate_data.py            ← Étape 0 : données synthétiques
│   ├── 01_prep_data.py                ← Étape 1 : feature engineering
│   ├── 02_train.py                    ← Étape 2 : train + register UC
│   ├── 03_validate.py                 ← Étape 3 : 4 checks pré-promotion
│   └── 05_monitor_hybrid.py           ← Étape 5 : Champion vs Challenger cloud-side
│
└── onprem/                            ← à exécuter sur votre laptop
    ├── pull_artifact.py               ← Étape 4a : télécharge le modèle
    ├── serve.py                       ← serveur FastAPI
    ├── Dockerfile                     ← image stable (runtime sans modèle)
    ├── requirements.txt               ← dépendances pinnées
    ├── run_container.sh               ← Étape 4b : Linux / macOS
    ├── run_container.ps1              ← Étape 4b : Windows PowerShell
    └── run_container.bat              ← Étape 4b : Windows CMD (fallback)
```

---

## Limites du kit pour la prod

Ce kit est **volontairement simplifié** pour la prise en main. Pour une production OIV-grade, il
faudrait ajouter :

- **Authentification OIDC** (federation policy au lieu de profile CLI)
- **Stockage intermédiaire** dédié (ADLS+SAS, Nexus, ACR…) au lieu du pull direct API MLflow
- **CI/CD** pour automatiser la chaîne build image + push artefact
- **Signature HMAC** du manifest pour l'intégrité bout-en-bout
- **Audit OIV** structuré (JSONL append-only, rétention 7 ans)
- **Orchestration** systemd / Kubernetes pour le pull périodique automatique

Les patterns associés sont détaillés dans le deck de cadrage qui accompagne ce kit.

---

## Contact

Pour les questions techniques sur ce kit ou pour discuter de l'industrialisation,
ouvrir une issue GitHub ou contacter directement l'équipe Databricks.
