FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    python -m venv "$VIRTUAL_ENV" && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    APP_PORT=8000 \
    APP_WORKERS=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    groupadd --system appgroup && \
    useradd --system --gid appgroup --uid 10001 --create-home appuser && \
    mkdir -p /app/.eshop_cache /tmp/sre-agent && \
    chown -R appuser:appgroup /app /tmp/sre-agent && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup app ./app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --workers ${APP_WORKERS:-1}"]
