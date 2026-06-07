"""SQLAlchemy 2.0 declarative base for the agent-runtime persistence schema.

`Base` is the single declarative registry every ORM model inherits from, so
`Base.metadata` is the schema source of truth Alembic diffs against (ADR-010).
Per ADR-016 all DB types live in `app.persistence`; the service layer never
imports SQLAlchemy.
See atlas-docs/03 §1.7.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base; `Base.metadata` is the authoritative schema."""
