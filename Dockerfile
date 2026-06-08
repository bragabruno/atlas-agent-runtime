# syntax=docker/dockerfile:1
# atlas-agent-runtime runtime image — multi-stage, non-root, pinned base.
# Base pinned exactly (atlas-docs/02 §2), matching the CI image. Runtime deps are
# the exact-pinned [project.dependencies] from pyproject.toml (no dev deps).

FROM python:3.12.13-slim-bookworm AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY . .
# Install the package + its pinned runtime deps into an isolated venv.
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir .

FROM python:3.12.13-slim-bookworm AS runtime
# Non-root runtime user.
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build /app /app
ENV PATH="/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
# PLACEHOLDER ENTRYPOINT — the FastAPI run-trigger surface (ADR-020) is not yet
# coded: app/api/main.py does not exist, and fastapi/uvicorn are not yet in
# [project.dependencies]. When that surface lands, the deploy/ chart's probes
# (/healthz, /metrics on :8000) expect this exact command. `docker build` never
# executes CMD, so the image builds today; the container will only run once the
# trigger surface and its deps are added. Real provider keys + Key Vault CSI are
# injected per-env at deploy time (atlas-docs/04); the image ships no secrets.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
