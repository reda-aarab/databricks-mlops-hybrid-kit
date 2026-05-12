@echo off
REM Build + run du container on-prem (Windows CMD fallback).
REM Pour PowerShell, préférer run_container.ps1.
REM
REM Pré-requis : Docker Desktop installé et démarré, pull_artifact.py exécuté.

cd /d "%~dp0"

echo Build de l'image wind-onprem...
docker build -t wind-onprem .
if errorlevel 1 (
    echo Build a echoue.
    exit /b 1
)

echo Lancement du container sur le port 8000...
docker run --rm -p 8000:8000 ^
    -v "%CD%/local_model:/var/cache/model:ro" ^
    -e MODEL_PATH=/var/cache/model ^
    wind-onprem
