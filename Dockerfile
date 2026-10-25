# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

# Keep runtime paths independent of the host machine.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    NHABE_PROJECT_ROOT=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /app

# OpenMP is needed at runtime; compilers belong only in the builder below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

# The ARM64 build of Py-ART selected a source distribution and requires gcc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv

# Reuse downloaded archives/wheels when retrying a build. This cache is not copied.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps .

FROM base AS runtime

# Only installed Python packages are copied from the builder, not its compilers.
COPY --from=builder /opt/venv /opt/venv

# Explicit copies prevent observations and private files entering the image.
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/pipeline.yaml configs/forecasting.yaml configs/normalization.yaml ./configs/

# Code/configuration stay root-owned; only outputs and staging are writable.
RUN groupadd --gid 10001 pipeline \
    && useradd --uid 10001 --gid pipeline --create-home pipeline \
    && mkdir -p /app/output /app/data/staging \
    && chown -R pipeline:pipeline /app/output /app/data/staging

USER pipeline

# Environment check during build; this is a batch CLI, not a long-running service.
RUN nhabe-pipeline doctor --quiet

ENTRYPOINT ["nhabe-pipeline"]
CMD ["--help"]
