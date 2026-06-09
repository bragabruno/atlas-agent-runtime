"""HTTP API layer — FastAPI trigger surface for agent runs (AGT-16, ADR-020).

Thin controllers (`app.api.v1`) map HTTP ↔ the service layer; DI wiring lives
in `app.api.deps`. Per ADR-016 nothing here imports `AgentRunner` or SQLAlchemy
directly — that is the service layer's job.
"""
