"""Runtime configuration (env-driven, no secrets in code) — AGT-16.

The FastAPI trigger surface (ADR-020) reads its collaborators from the
environment with the ``ATLAS_`` prefix. Persistence is **config-gated**: with
no ``ATLAS_DATABASE_URL`` the service still serves runs (POST works), it just
does not persist them (GET /v1/agent/runs/{id} returns 503) — mirroring the
gateway's default-OFF philosophy so the service boots with zero external
dependencies for smoke tests.

Real deployments inject ``ATLAS_*`` per-env via the Key Vault CSI mount
(atlas-docs/04); the image ships no secrets.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    #: Atlas Gateway base URL (OpenAI-compatible /v1/chat/completions).
    gateway_url: str = "http://localhost:8000"

    #: Bearer token presented to the gateway. Local dev default is "dev-key";
    #: real deployments source this from Key Vault.
    gateway_api_key: str = "dev-key"

    #: SQLAlchemy DSN for agent_runs/agent_steps persistence. ``None`` (default)
    #: disables persistence: runs still execute, but are not stored and the
    #: GET status endpoint returns 503.
    database_url: str | None = None

    #: Directory holding agent YAML specs, resolved as ``<dir>/<agent_name>.yaml``.
    agents_dir: str = "agents"

    #: MCP tool-server URLs (wired for parity with the Helm chart; tool execution
    #: lives outside the runner, so these are not required for the trigger surface).
    mcp_doc_search_url: str | None = None
    mcp_citations_url: str | None = None


def get_settings() -> Settings:
    return Settings()
