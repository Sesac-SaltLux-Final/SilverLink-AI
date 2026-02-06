# syntax=docker/dockerfile:1
# 👆 BuildKit 활성화

# ✅ Base Image: OS 버전을 'bookworm'으로 명시
FROM python:3.12-slim-bookworm AS builder

# 1. 환경 변수
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.0.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_INDEX_URL=https://mirror.kakao.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    POETRY_REQUESTS_TIMEOUT=300

# 2. 빌드 도구 설치 (pkg-config 추가됨!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake ninja-build git \
    libopenblas-dev \
    pkg-config \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /code

# 3. 일반 의존성 설치
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry config installer.max-workers 1 \
 && poetry install --no-root --only main

# 4. llama-cpp-python 0.3.16 빌드 (GGML_BLAS 적용)
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN --mount=type=cache,target=/root/.cache/pip \
    poetry run pip install --no-cache-dir --upgrade pip \
 && poetry run pip install --no-cache-dir --no-binary :all: "llama-cpp-python==0.3.16"

# 5. Spacy 모델 설치
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry run pip install "https://github.com/explosion/spacy-models/releases/download/ko_core_news_lg-3.7.0/ko_core_news_lg-3.7.0-py3-none-any.whl"


# ==========================================
# Final stage
# ==========================================
FROM python:3.12-slim-bookworm

# 1. 런타임 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /code

# 3. 빌드 아티팩트 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 4. 소스 코드 복사
COPY --chown=appuser:appuser . .

# 5. logs 폴더 생성 (Final 단계에서!)
RUN mkdir -p /code/logs && chown -R appuser:appuser /code/logs

# 6. 실행
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]