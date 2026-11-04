FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    WEB_CONCURRENCY=3 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non-root + dossier de logs accessible en écriture
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Plusieurs workers uvicorn (WEB_CONCURRENCY) : un processus par cœur, car un seul
# processus Python n'utilise qu'un cœur. Les métriques Prometheus des workers sont
# agrégées via PROMETHEUS_MULTIPROC_DIR, vidé à chaque démarrage.
CMD ["sh", "-c", "rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers \"$WEB_CONCURRENCY\""]
