#!/usr/bin/env bash
# Build + run du conteneur on-prem.
# Le modèle local (./local_model rempli par pull_artifact.py) est monté en read-only
# dans le conteneur à /var/cache/model. L'image elle-même ne contient pas le modèle —
# c'est la séparation "image stable / modèle mouvant" du kicker deck "L'image Docker on-prem".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Si pip est configuré avec un mirror local (corporate proxy), on le passe au build.
PIP_INDEX_URL="$(pip config get global.index-url 2>/dev/null || echo 'https://pypi.org/simple')"
docker build --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" -t wind-onprem .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/local_model:/var/cache/model:ro" \
  -e MODEL_PATH=/var/cache/model \
  wind-onprem
