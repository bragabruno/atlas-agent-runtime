"""FastAPI dependency providers — read collaborators from app.state (AGT-16).

The composition root (`app.main`) builds the gateway client, settings, and
(optional) session factory once at startup and stores them on ``app.state``.
These providers expose them to the routes via ``Depends`` so tests can override
each one independently with ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from app.loop.gateway_client import GatewayClient


def settings_dep(request: Request) -> Settings:
    return request.app.state.settings


# Declared once so FastAPI emits a single `BearerAuth` entry under
# `components.securitySchemes` in the OpenAPI spec; the two /v1/agent/runs
# routes that depend on `require_api_key` then advertise
# `security: [{ BearerAuth: [] }]` and Swagger UI renders the Authorize button.
# `auto_error=False` keeps the 401 semantics owned by `require_api_key`, mirroring
# the gateway's deps (the detail strings are part of the contract tests assert).
bearer_scheme = HTTPBearer(
    scheme_name="BearerAuth",
    bearerFormat="Atlas API key",
    description=(
        "Atlas API key sent as `Authorization: Bearer <key>`. Local dev "
        "default is `dev-key`; real deployments source keys from Key Vault."
    ),
    auto_error=False,
)


def require_api_key(
    settings: Annotated[Settings, Depends(settings_dep)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> str:
    """Enforce ``Authorization: Bearer <key>`` against ``settings.api_keys``.

    Raises 401 on a missing/non-bearer header or a key not in the allowlist.
    Applied to the agent-trigger routes only; ``/healthz`` stays open (BRA-878).
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing bearer token")
    key = credentials.credentials.strip()
    if key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid api key")
    return key


def gateway_client_dep(request: Request) -> GatewayClient:
    return request.app.state.gateway_client


def session_factory_dep(request: Request) -> sessionmaker[Session] | None:
    """Return the session factory, or ``None`` when persistence is disabled."""
    return request.app.state.session_factory
