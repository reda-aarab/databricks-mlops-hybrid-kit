# Prérequis — Setup avant de rejouer la démo

Tout ce qu'il faut installer une fois pour faire tourner le kit. Compte environ **15 minutes**
si vous partez de zéro sous Windows, **5 minutes** sous macOS / Linux.

---

## 1. Compte Databricks Free Edition

https://www.databricks.com/learn/free-edition

Gratuit, sans carte bancaire. Donne accès à :
- Un workspace Databricks complet
- Unity Catalog (catalog `workspace` par défaut)
- Compute Serverless (limité mais suffisant pour la démo)
- MLflow Tracking + Registry

À la première connexion, notez votre **URL de workspace** sous la forme
`https://<random-id>.cloud.databricks.com`. Vous en aurez besoin pour le CLI.

---

## 2. Python 3.10

**Impératif : Python 3.10, pas 3.11 ou plus récent.**

La raison : le wrapper pyfunc est sérialisé par cloudpickle côté Databricks Serverless qui
tourne en Python 3.10. Le format des objets `code` Python a changé entre 3.10 et 3.11, donc
charger l'artefact dans un Python 3.11+ échoue avec une `TypeError`. Voir
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) pour le détail.

### Installation par OS

**Windows**
1. Télécharger l'installeur officiel : https://www.python.org/downloads/release/python-31013/
2. Choisir *Windows installer (64-bit)*
3. Pendant l'install : **cocher "Add python.exe to PATH"** (important)
4. Vérifier dans une nouvelle fenêtre PowerShell :
   ```powershell
   python --version
   ```
   Doit afficher `Python 3.10.x`.

**macOS** (via Homebrew)
```bash
brew install python@3.10
python3.10 --version
```

**Linux Ubuntu / Debian**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv
python3.10 --version
```

---

## 3. Docker Desktop

Pour le container d'inférence on-prem.

**Windows**
1. Télécharger : https://www.docker.com/products/docker-desktop/
2. Pendant l'install : laisser cochée l'option **"Use WSL 2 instead of Hyper-V"**
3. Redémarrer le PC si demandé
4. Lancer Docker Desktop, attendre que l'icône baleine soit stable

**macOS**
1. Télécharger : https://www.docker.com/products/docker-desktop/ (version Apple Silicon ou Intel selon votre Mac)
2. Glisser dans Applications
3. Lancer Docker Desktop, autoriser les permissions

**Vérifier**
```bash
docker version
```
Doit afficher Client + Server versions.

---

## 4. Databricks CLI

Pour s'authentifier depuis le laptop vers le workspace cloud.

**Installation universelle** (recommandée)
```bash
pip install databricks-cli
databricks --version
```

**Alternative Windows / Mac via installeur** : https://docs.databricks.com/dev-tools/cli/install.html

### Authentification au workspace

```bash
databricks auth login --host https://<votre-workspace>.cloud.databricks.com --profile demo
```

- Remplacer `<votre-workspace>` par l'URL réelle de votre Free Edition
- `--profile demo` peut être n'importe quel nom (sera réutilisé dans `pull_artifact.py`)
- Un navigateur s'ouvre, login via votre compte Databricks

Vérifier :
```bash
databricks current-user me --profile demo
```

Doit afficher votre nom et email.

---

## 5. Git

Pour cloner le kit.

**Windows**
1. Télécharger : https://git-scm.com/download/win
2. Installer avec les options par défaut (Git Bash inclus)

**macOS**
```bash
brew install git
```
ou installé via Xcode Command Line Tools : `xcode-select --install`

**Linux**
```bash
sudo apt install git
```

---

## Récapitulatif des versions

| Outil | Version requise | Comment vérifier |
|---|---|---|
| Python | **3.10.x** (strict) | `python --version` ou `python3.10 --version` |
| Docker | 20.x ou plus récent | `docker version` |
| Databricks CLI | 0.18+ ou v2 | `databricks --version` |
| Git | n'importe quelle version récente | `git --version` |

---

## Espace disque requis

- Python 3.10 + venv + dépendances : ~500 MB
- Docker image `wind-onprem` : ~700 MB
- Modèle pullé local : ~50 KB
- **Total** : ~1.5 GB de libre sur la machine

---

## Configuration réseau

Le kit nécessite des connexions sortantes HTTPS vers :
- `*.cloud.databricks.com` (votre workspace)
- `pypi.org` ou votre mirror PyPI corporate (pour `pip install`)
- `hub.docker.com` ou registre Docker corporate (pour `docker pull python:3.10-slim`)

Si vous êtes derrière un proxy d'entreprise, configurer :

**pip** : `pip config set global.index-url https://<votre-mirror>/simple`

**Docker** : Docker Desktop → Settings → Resources → Proxies

**Databricks CLI** : variables d'environnement `HTTP_PROXY` / `HTTPS_PROXY`

---

## Vous êtes prêt ?

Retournez au [README.md](README.md) section *Setup en 5 étapes*.
