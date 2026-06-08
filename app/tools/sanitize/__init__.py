"""Tool result sanitizer (AGT-5).

Public surface:
- `ToolSanitizer` — sanitizes a single tool output, returns `SanitizedToolOutput`.
- `SanitizedToolOutput` — value object: sanitized text + ``fenced`` flag.
- `sanitize_tool_output` — convenience wrapper; returns the sanitized string directly.
- `FENCE_OPEN` / `FENCE_CLOSE` — untrusted-content fence delimiters (plain ASCII).
- `strip_hidden_chars` / `defuse_forged_roles` / `neutralize_injections` — individual
  transform steps; exposed so callers and tests can exercise each step in isolation.
"""

from __future__ import annotations

from app.tools.sanitize.sanitizer import (
    FENCE_CLOSE,
    FENCE_OPEN,
    SanitizedToolOutput,
    ToolSanitizer,
    defuse_forged_roles,
    neutralize_injections,
    sanitize_tool_output,
    strip_hidden_chars,
)

__all__ = [
    "FENCE_CLOSE",
    "FENCE_OPEN",
    "SanitizedToolOutput",
    "ToolSanitizer",
    "defuse_forged_roles",
    "neutralize_injections",
    "sanitize_tool_output",
    "strip_hidden_chars",
]
