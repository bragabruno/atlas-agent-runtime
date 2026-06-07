"""ToolRegistry — whitelist enforcement for agent tool calls (AGT-4).

Maps each agent (by name) to its allowed tool set, taken from
`AgentSpec.tool_whitelist`.  Every tool call must be checked with
`assert_allowed` before execution; a non-whitelisted name raises
`ToolNotAllowedError` — never silently passed through.

Usage::

    registry = ToolRegistry(spec)
    registry.assert_allowed("doc_search")   # OK
    registry.assert_allowed("rm_everything")  # raises ToolNotAllowedError

The registry is immutable after construction (the whitelist is frozen).
"""

from __future__ import annotations

from app.agentspec.model import AgentSpec


class ToolNotAllowedError(Exception):
    """Raised when an agent attempts to call a tool not in its whitelist.

    Attributes:
        agent_name: the agent that attempted the call.
        tool_name:  the disallowed tool name.
    """

    def __init__(self, agent_name: str, tool_name: str, allowed: frozenset[str]) -> None:
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.allowed = allowed
        msg = (
            f"ToolNotAllowedError: agent '{agent_name}' is not permitted to call "
            f"'{tool_name}'. Allowed tools: {sorted(allowed)}"
        )
        super().__init__(msg)


class ToolRegistry:
    """Immutable whitelist guard built from an `AgentSpec`.

    The registry holds the frozenset of allowed tool names for a single agent.
    Build one per agent invocation and inject it into the runner.
    """

    def __init__(self, spec: AgentSpec) -> None:
        self._agent_name: str = spec.agent_name
        self._allowed: frozenset[str] = frozenset(spec.tool_whitelist)

    @property
    def agent_name(self) -> str:
        """The agent this registry is scoped to."""
        return self._agent_name

    @property
    def allowed_tools(self) -> frozenset[str]:
        """Immutable set of permitted tool names."""
        return self._allowed

    def is_allowed(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is in the whitelist, ``False`` otherwise."""
        return tool_name in self._allowed

    def assert_allowed(self, tool_name: str) -> None:
        """Check *tool_name* against the whitelist.

        Raises:
            ToolNotAllowedError: if the tool is not in the whitelist.
        """
        if tool_name not in self._allowed:
            raise ToolNotAllowedError(self._agent_name, tool_name, self._allowed)
