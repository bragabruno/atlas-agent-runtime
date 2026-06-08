"""AGT-5 — Tool/MCP output sanitization before context re-entry.

Defense against *injection-via-tool-results*: a tool or MCP server can return
attacker-influenced text (a poisoned document chunk, a crafted API response).
If that text re-enters the agent's context verbatim it becomes an injection
vector — "ignore your instructions", a forged ``System:`` turn, or a hidden
directive smuggled in a zero-width / control-char payload.

This module is the agent-runtime equivalent of the gateway's GRD-6 sanitizer.
The two repos do not share a package — this is an intentional re-implementation
that mirrors GRD-6 behavior exactly. The same four-step transform applies:

1. Strip control / zero-width characters used to hide instructions.
2. Defuse forged role turns (``System:`` / ``Assistant:`` etc.) so the model
   cannot mistake tool data for a privileged conversation turn.
3. Neutralize known injection imperatives (``ignore previous instructions``
   and friends) by wrapping the trigger verb — the phrase loses its imperative
   force while the text stays human-readable.
4. Fence the whole result in an explicit untrusted-content delimiter so the
   downstream prompt assembler keeps tool data lexically separate from trusted
   instructions.

Stdlib ``re`` / ``unicodedata`` only — no external deps, no network.
Fail-fast on a non-``str`` payload (a programming error); a *clean* payload
returns defanged, never rejected, so the agent loop keeps making progress.
See AGT-5 + ADR-016 (capability module pattern).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fence labels wrapping a sanitized tool result. The downstream prompt
#: assembler treats everything between these markers as untrusted data, never
#: as instructions. Plain ASCII so they survive any later transport.
#: Exposed as public constants so downstream code and tests can reference them
#: without duplicating the strings.
FENCE_OPEN = "[BEGIN UNTRUSTED TOOL OUTPUT]"
FENCE_CLOSE = "[END UNTRUSTED TOOL OUTPUT]"

#: Unicode "format" (Cf) and control (Cc) characters are removed outright.
#: Structural whitespace is kept — tab, newline, carriage return.
_KEEP_CONTROL = frozenset({"\t", "\n", "\r"})

#: Forged role turn at the start of a line — ``System:`` / ``Assistant:`` etc.
_FORGED_ROLE_RE = re.compile(
    r"(?im)^[ \t]*(system|assistant|user|developer|tool)[ \t]*:",
)

#: How a forged leading role token is rewritten.
_NEUTRALIZED_ROLE = r"(defanged-role: \1)"

#: Known injection imperatives.  The captured trigger verb is bracketed in
#: place; surrounding text is preserved for auditability.
_INJECTION_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore-previous-instructions",
        re.compile(r"(?i)\b(ignore|disregard|forget)\b(?=[\s\S]{0,40}\binstructions?\b)"),
    ),
    (
        "override-system-prompt",
        re.compile(r"(?i)\b(override|bypass)\b(?=[\s\S]{0,40}\b(?:system\s+)?prompt\b)"),
    ),
    (
        "reveal-system-prompt",
        re.compile(r"(?i)\b(reveal|leak)\b(?=[\s\S]{0,40}\b(?:system\s+)?prompt\b)"),
    ),
)


# ---------------------------------------------------------------------------
# Step helpers (public for unit-testing each step in isolation)
# ---------------------------------------------------------------------------


def strip_hidden_chars(text: str) -> str:
    """Drop zero-width/format and control chars used to smuggle directives.

    Removes every Unicode ``Cf`` (format) char and every ``Cc`` (control)
    char except the structural whitespace in `_KEEP_CONTROL`. Visible text is
    untouched so a clean payload reads identically afterward.
    """
    out: list[str] = []
    for ch in text:
        if ch in _KEEP_CONTROL:
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in ("Cf", "Cc"):
            continue
        out.append(ch)
    return "".join(out)


def defuse_forged_roles(text: str) -> str:
    """Rewrite leading ``System:``-style role tokens so they cannot impersonate a turn."""
    return _FORGED_ROLE_RE.sub(_NEUTRALIZED_ROLE, text)


def neutralize_injections(text: str) -> str:
    """Wrap known injection trigger verbs so the imperative loses its force.

    The trigger word (e.g. ``ignore``) is bracketed in place; surrounding text
    is preserved so the result stays auditable.
    """
    for _label, pattern in _INJECTION_RES:
        text = pattern.sub(r"[\1]", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SanitizedToolOutput:
    """The result of sanitizing one tool/MCP output.

    ``text`` is the safe-to-re-inject, fenced payload. ``fenced`` is ``True``
    when the untrusted-content delimiters were added (the default). Returned as
    a value object so callers (AgentRunner) can log/trace the transform without
    re-running it; equality is by value for easy assertions in tests.
    """

    text: str
    fenced: bool


class ToolSanitizer:
    """Neutralizes tool/MCP output before it re-enters agent context (AGT-5).

    Reusable capability the agent runtime composes into its loop. The transform
    is deterministic and order-fixed:

        strip hidden chars
        → defuse forged role turns
        → neutralize injection imperatives
        → fence as untrusted

    ``fence`` may be disabled by a caller that applies its own delimiters, but
    it is on by default because lexical separation of tool data from
    instructions is the primary defense.

    Fail-fast: a non-``str`` payload raises ``TypeError`` (a wiring bug that
    must be fixed upstream, not guessed at here).
    """

    def __init__(self, *, fence: bool = True) -> None:
        self._fence = fence

    @property
    def fences(self) -> bool:
        """Whether this sanitizer wraps output in untrusted-content delimiters."""
        return self._fence

    def sanitize(self, output: object) -> SanitizedToolOutput:
        """Return *output* defanged and (by default) fenced as untrusted content.

        Raises:
            TypeError: if *output* is not a ``str``.  A tool that returned a
                non-text payload must be adapted upstream, not coerced here.
        """
        if not isinstance(output, str):
            raise TypeError(f"tool output must be str, got {type(output).__name__}")

        cleaned = strip_hidden_chars(output)
        cleaned = defuse_forged_roles(cleaned)
        cleaned = neutralize_injections(cleaned)

        if self._fence:
            cleaned = f"{FENCE_OPEN}\n{cleaned}\n{FENCE_CLOSE}"

        return SanitizedToolOutput(text=cleaned, fenced=self._fence)


def sanitize_tool_output(output: object, *, fence: bool = True) -> str:
    """Module-level convenience: sanitize one tool output and return the text.

    Thin wrapper over :class:`ToolSanitizer` for the common single-call case.
    Same fail-fast contract: a non-``str`` payload raises ``TypeError``.
    """
    return ToolSanitizer(fence=fence).sanitize(output).text
