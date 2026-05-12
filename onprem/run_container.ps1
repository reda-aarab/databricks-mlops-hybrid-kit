# Build + run du container on-prem (Windows PowerShell).
# Équivalent de run_container.sh.
#
# Pré-requis : Docker Desktop installé et démarré, pull_artifact.py exécuté
# (donc le dossier ./local_model/ existe à côté de ce script).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Détecte si un mirror PyPI corporate est configuré (via pip config)
try {
    $pipIndexUrl = (pip config get global.index-url 2>$null)
    if (-not $pipIndexUrl) { $pipIndexUrl = "https://pypi.org/simple" }
} catch {
    $pipIndexUrl = "https://pypi.org/simple"
}

Write-Host "Build de l'image avec PIP_INDEX_URL=$pipIndexUrl" -ForegroundColor Cyan
docker build --build-arg PIP_INDEX_URL=$pipIndexUrl -t wind-onprem .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build a échoué." -ForegroundColor Red
    exit 1
}

Write-Host "Lancement du container sur le port 8000" -ForegroundColor Cyan
docker run --rm -p 8000:8000 `
    -v "${PWD}/local_model:/var/cache/model:ro" `
    -e MODEL_PATH=/var/cache/model `
    wind-onprem
