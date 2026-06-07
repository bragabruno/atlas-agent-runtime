"""Loop-level domain errors (AGT-3).

`CapBreachError` is raised by `AgentRunner` when any hard cap is exceeded.
The message always names the breached cap and its configured limit so callers
can log and persist the exact reason without inspecting the spec again.
"""

from __future__ import annotations


class CapBreachError(Exception):
    """Raised when the agent loop breaches a hard cap.

    The `cap` attribute names the breached field from `AgentSpec`
    (``"max_iterations"``, ``"token_budget"``, or ``"timeout_s"``).
    The human-readable `message` is also stored in `args[0]`.
    """

    def __init__(self, cap: str, limit: int | float, unit: str = "") -> None:
        self.cap = cap
        self.limit = limit
        suffix = f" {unit}" if unit else ""
        msg = f"CapBreachError: {cap} ({limit}{suffix}) exceeded"
        super().__init__(msg)
