"""FastAPI dependency providers — read collaborators from app.state (AGT-16).

The composition root (`app.main`) builds the gateway client, settings, and
(optional) session factory once at startup and stores them on ``app.state``.
These providers expose them to the routes via ``Depends`` so tests can override
each one independently with ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from app.config import Settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from app.loop.gateway_client import GatewayClient


def settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def gateway_client_dep(request: Request) -> GatewayClient:
    return request.app.state.gateway_client


def session_factory_dep(request: Request) -> sessionmaker[Session] | None:
    """Return the session factory, or ``None`` when persistence is disabled."""
    return request.app.state.session_factory
