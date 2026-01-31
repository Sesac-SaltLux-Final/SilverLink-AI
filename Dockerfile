# Builder stage
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.0.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /code
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

# 여기서 spaCy 모델 다운로드
RUN python -m spacy download ko_core_news_lg

# Final stage (슬림 + non-root)
FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /code
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 기존
# COPY . .
# RUN chown -R appuser:appuser /code
# 변경 후 (더 효율적)
COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]