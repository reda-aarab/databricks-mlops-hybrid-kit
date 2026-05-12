# Troubleshooting — Erreurs courantes

Erreurs rencontrées en pratique lors du déroulé du kit, avec leur cause et leur solution.

---

## Côté Databricks (cloud)

### `CREATE CATALOG IF NOT EXISTS ... failed: Metastore storage root URL does not exist`

**Cause** : votre workspace Free Edition n'autorise pas la création de nouveaux catalogs.

**Solution** : ne pas créer de catalog. Utiliser le catalog par défaut `workspace`. Adapter la
ligne `CATALOG = "workspace"` en haut de chaque notebook si ce n'est pas déjà le cas.

---

### `RESOURCE_DOES_NOT_EXIST: Parent directory /Users/votre.email/wind_forecast_demo does not exist`

**Cause** : le notebook `02_train.py` essaie de créer une expérience MLflow dans un chemin qui
n'existe pas.

**Solution** : la version corrigée du notebook utilise `current_user()` pour ranger l'expérience
dans votre home (`/Users/votre.email/wind_forecast_demo`). Si vous obtenez cette erreur, vérifier que le
notebook contient bien :

```python
_CURRENT_USER = spark.sql("SELECT current_user() AS u").collect()[0]["u"]
EXPERIMENT_NAME = f"/Users/{_CURRENT_USER}/wind_forecast_demo"
```

---

### `Unable to import necessary dependencies to access model version files in Unity Catalog`

**Cause** : la lib `mlflow-skinny` est installée sans l'extra `[databricks]`.

**Solution** : la cellule `%pip install` de chaque notebook inclut `mlflow-skinny[databricks]==2.16.2`.
Vérifier que c'est bien la syntaxe utilisée et que `dbutils.library.restartPython()` est appelé
ensuite pour recharger l'environnement.

---

### `Cannot determine type for column previous_version`

**Cause** : Spark Connect ne peut pas inférer le type d'une colonne quand toutes ses valeurs
sont `None` dans le batch.

**Solution** : la version corrigée du notebook `04_export_to_adls.py` (si vous utilisez la
variante OIV) déclare un schéma explicite avec `StructType`. Pour le kit Free Edition, ce
notebook n'est pas inclus, donc l'erreur ne devrait pas survenir.

---

### Les versions du modèle restent en `PENDING_REGISTRATION`

**Cause** : un run précédent a planté pendant la copie des artefacts vers le storage UC. La
version est créée mais ne devient jamais `READY`.

**Solution** : ce sont des zombies orphelins, sans impact sur la dernière version. Vous pouvez
les ignorer ou les supprimer manuellement via Catalog Explorer. Une fois supprimées, les
versions suivantes seront numérotées correctement.

---

## Côté laptop (on-prem)

### `TypeError: code() argument 13 must be str, not int` au démarrage du container

**Cause** : vous tournez Python 3.11+ alors que le wrapper pyfunc a été sérialisé en Python 3.10
côté Databricks Serverless. Le format des objets `code` Python a changé entre 3.10 et 3.11.

**Solution** : utiliser **strictement Python 3.10** localement. Recréer le venv :

```bash
# Supprimer l'ancien venv
rm -rf .venv

# Recréer avec Python 3.10
python3.10 -m venv .venv      # macOS / Linux
py -3.10 -m venv .venv        # Windows avec py launcher

# Réinstaller
.venv/bin/pip install -r requirements.txt   # macOS / Linux
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
```

Si vous n'avez pas Python 3.10 :
- **Windows** : https://www.python.org/downloads/release/python-31013/
- **macOS** : `brew install python@3.10`
- **Linux** : `sudo apt install python3.10 python3.10-venv`

---

### `docker: Error response from daemon: dial unix /var/run/docker.sock: no such file or directory`

**Cause** : Docker Desktop n'est pas démarré.

**Solution** : lancer Docker Desktop (icône application) et attendre que l'icône baleine soit
stable dans la barre menu / barre de tâches. Puis vérifier :

```bash
docker version
```

Sur Windows, Docker Desktop peut prendre 30-60 secondes à démarrer la première fois.

---

### `Sign in to continue using Docker Desktop`

**Cause** : politique d'entreprise qui impose un sign-in périodique dans Docker Desktop.

**Solution** : cliquer sur l'icône Docker dans la barre menu → **Sign in** → suivre le flow SSO
de votre organisation. Une fois signé, relancer `docker build` / `docker run`.

---

### `Could not find a version that satisfies the requirement mlflow-skinny`

**Cause** : votre proxy d'entreprise bloque PyPI public, et pip n'est pas configuré pour utiliser
le mirror interne.

**Solution Windows** : configurer le mirror dans pip :

```powershell
pip config set global.index-url https://<votre-mirror>/simple
```

Puis re-tester. Si vous ne savez pas l'URL du mirror, demander à votre équipe DevOps.

---

### Le container démarre mais `/predict` retourne 500

**Cause possible 1** : le modèle pullé est corrompu ou incomplet.

**Solution** : refaire le pull :
```bash
rm -rf local_model
python pull_artifact.py
```

**Cause possible 2** : le payload est mal formé.

**Solution** : vérifier que `window` contient exactement **5 entrées**, chacune avec **toutes
les 5 colonnes** (`timestamp`, `u10`, `v10`, `t2m`, `sp`).

---

### Le port 8000 est déjà utilisé

**macOS / Linux** :
```bash
lsof -ti :8000 | xargs kill
```

**Windows (PowerShell)** :
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

Puis relancer le container.

---

### `pull_artifact.py` : `Profile DEFAULT does not exist`

**Cause** : le `DATABRICKS_CONFIG_PROFILE` pointe sur un profile qui n'a pas été créé.

**Solution** : adapter en haut de `pull_artifact.py` :

```python
DATABRICKS_PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "demo")
```

ou exporter la variable d'environnement avant de lancer :

```bash
export DATABRICKS_CONFIG_PROFILE=demo    # macOS / Linux
$env:DATABRICKS_CONFIG_PROFILE = "demo"  # Windows PowerShell
```

---

## Côté Windows spécifiquement

### Le script `run_container.ps1` est bloqué par l'execution policy

**Cause** : Windows refuse par défaut d'exécuter des scripts PowerShell non signés.

**Solution** : autoriser les scripts dans la session courante :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\run_container.ps1
```

L'autorisation ne dure que pour la session PowerShell ouverte.

---

### `bash: command not found` quand vous essayez `bash run_container.sh`

**Cause** : `bash` n'est pas installé nativement sur Windows.

**Solution** : utiliser le script PowerShell `run_container.ps1` à la place. Ou installer Git
Bash (livré avec Git pour Windows) et l'utiliser pour `bash run_container.sh`.

---

### Le volume Docker ne fonctionne pas avec un chemin Windows

**Cause** : le `bind mount` Docker attend un format spécifique selon le shell.

**Solution** : dans PowerShell, utiliser `${PWD}` (avec accolades) au lieu de `$PWD`. Le
`run_container.ps1` fait déjà ça correctement. Si vous lancez manuellement :

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/local_model:/var/cache/model:ro" wind-onprem
```

---

## Encore bloqué ?

Ouvrir une issue sur le repo GitHub avec :
- Le message d'erreur exact (copier-coller, pas une capture d'écran si possible)
- Votre OS et version de Python
- À quelle étape du README l'erreur est survenue
- La sortie de `docker version` et `python --version`
