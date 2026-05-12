FROM python:3.10-slim

WORKDIR /app

# Install dépendances en premier — layer cacheable, rebuild seulement si requirements.txt change
# PIP_INDEX_URL override pour les environnements avec mirror corporate (Databricks, etc.)
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt /app/
RUN pip install --no-cache-dir --index-url=${PIP_INDEX_URL} -r /app/requirements.txt

# Copie uniquement le code de service — le modèle est monté en volume au runtime
COPY serve.py /app/

ENV MODEL_PATH=/var/cache/model
EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
