"""AGT-4 ToolRegistry tests — fully offline, no gateway or DB.

Covers:
- ToolRegistry built from AgentSpec has correct allowed set.
- is_allowed returns True for whitelisted tool, False otherwise.
- assert_allowed passes silently for whitelisted tool.
- assert_allowed raises ToolNotAllowedError for non-whitelisted tool.
- ToolNotAllowedError carries agent_name, tool_name, and allowed set.
- ToolNotAllowedError message names the disallowed tool and agent.
- allowed_tools is immutable (frozenset).
- AgentRunner with registry rejects non-whitelisted tool_use response.
- AgentRunner with registry allows whitelisted tool_use response.
- AgentRunner without registry skips whitelist check.
"""

from __future__ import annotations

import pytest

from app.agentspec.model import AgentSpec
from app.loop.errors import CapBreachError
from app.loop.gateway_client import LLMRequest, LLMResponse
from app.loop.runner import AgentRunner
from app.tools.registry import ToolNotAllowedError, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(whitelist: list[str] | None = None) -> AgentSpec:
    return AgentSpec(
        agent_name="test-agent",
        agent_version="1.0.0",
        system_prompt_ref="prompts/test.txt",
        model_alias="atlas-default",
        tool_whitelist=whitelist if whitelist is not None else ["doc_search", "verify_citation"],
        max_iterations=10,
        token_budget=100_000,
        timeout_s=60,
    )


class FakeGatewayClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def chat(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        if self._idx >= len(self._responses):
            raise IndexError("FakeGatewayClient exhausted")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _stop(content: str = "Answer.") -> LLMResponse:
    return LLMResponse(content=content, finish_reason="stop", input_tokens=10, output_tokens=10)


def _tool_use(tool_name: str) -> LLMResponse:
    return LLMResponse(
        content=f"tool:{tool_name}",
        finish_reason="tool_use",
        input_tokens=10,
        output_tokens=10,
    )


# ---------------------------------------------------------------------------
# ToolRegistry unit tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_allowed_tools_matches_whitelist(self) -> None:
        spec = _spec(["doc_search", "verify_citation"])
        registry = ToolRegistry(spec)
        assert registry.allowed_tools == frozenset(["doc_search", "verify_citation"])

    def test_agent_name_matches_spec(self) -> None:
        spec = _spec()
        registry = ToolRegistry(spec)
        assert registry.agent_name == "test-agent"

    def test_is_allowed_true_for_whitelisted(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        assert registry.is_allowed("doc_search") is True

    def test_is_allowed_false_for_unlisted(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        assert registry.is_allowed("rm_everything") is False

    def test_assert_allowed_passes_for_whitelisted(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        registry.assert_allowed("doc_search")  # must not raise

    def test_assert_allowed_raises_for_unlisted(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        with pytest.raises(ToolNotAllowedError):
            registry.assert_allowed("rm_everything")

    def test_error_carries_agent_name(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        with pytest.raises(ToolNotAllowedError) as exc_info:
            registry.assert_allowed("bad_tool")
        assert exc_info.value.agent_name == "test-agent"

    def test_error_carries_tool_name(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        with pytest.raises(ToolNotAllowedError) as exc_info:
            registry.assert_allowed("bad_tool")
        assert exc_info.value.tool_name == "bad_tool"

    def test_error_carries_allowed_set(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        with pytest.raises(ToolNotAllowedError) as exc_info:
            registry.assert_allowed("bad_tool")
        assert exc_info.value.allowed == frozenset(["doc_search"])

    def test_error_message_names_agent_and_tool(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        with pytest.raises(ToolNotAllowedError) as exc_info:
            registry.assert_allowed("bad_tool")
        msg = str(exc_info.value)
        assert "test-agent" in msg
        assert "bad_tool" in msg

    def test_allowed_tools_is_frozenset(self) -> None:
        registry = ToolRegistry(_spec(["doc_search"]))
        assert isinstance(registry.allowed_tools, frozenset)

    def test_empty_whitelist_rejects_everything(self) -> None:
        # AgentSpec requires at least one entry but let's guard defensively
        # by using a spec with one entry and a different tool name
        registry = ToolRegistry(_spec(["only_allowed"]))
        assert not registry.is_allowed("doc_search")


# ---------------------------------------------------------------------------
# AgentRunner + registry integration (AGT-4)
# ---------------------------------------------------------------------------


class TestAgentRunnerRegistry:
    @pytest.mark.asyncio
    async def test_runner_rejects_non_whitelisted_tool(self) -> None:
        """AgentRunner with registry raises ToolNotAllowedError for unlisted tool."""
        spec = _spec(["doc_search"])
        registry = ToolRegistry(spec)
        # LLM calls a tool NOT in the whitelist, then stops
        client = FakeGatewayClient([_tool_use("forbidden_tool"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)
        with pytest.raises(ToolNotAllowedError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.tool_name == "forbidden_tool"

    @pytest.mark.asyncio
    async def test_runner_allows_whitelisted_tool(self) -> None:
        """AgentRunner with registry proceeds when the tool is in the whitelist."""
        spec = _spec(["doc_search"])
        registry = ToolRegistry(spec)
        client = FakeGatewayClient([_tool_use("doc_search"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)
        result = await runner.run(user_message="Q")
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_runner_without_registry_skips_check(self) -> None:
        """AgentRunner without registry never raises ToolNotAllowedError."""
        spec = _spec(["doc_search"])
        client = FakeGatewayClient([_tool_use("forbidden_tool"), _stop()])
        runner = AgentRunner(spec=spec, client=client)  # no registry
        # Should complete without error
        result = await runner.run(user_message="Q")
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_runner_multiple_whitelisted_calls(self) -> None:
        """Multiple tool calls all in the whitelist pass through."""
        spec = _spec(["doc_search", "verify_citation"])
        registry = ToolRegistry(spec)
        client = FakeGatewayClient([_tool_use("doc_search"), _tool_use("verify_citation"), _stop()])
        runner = AgentRunner(spec=spec, client=client, registry=registry)
        result = await runner.run(user_message="Q")
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_runner_with_cap_breach_and_registry(self) -> None:
        """CapBreachError still fires even when a registry is injected."""
        spec = _spec()
        registry = ToolRegistry(spec)
        # 20 continue responses, iteration cap = 10 (from _spec default)
        responses = [_tool_use("doc_search")] * 20
        client = FakeGatewayClient(responses)
        runner = AgentRunner(
            spec=AgentSpec(
                agent_name="test-agent",
                agent_version="1.0.0",
                system_prompt_ref="prompts/test.txt",
                model_alias="atlas-default",
                tool_whitelist=["doc_search"],
                max_iterations=3,
                token_budget=100_000,
                timeout_s=60,
            ),
            client=client,
            registry=registry,
        )
        with pytest.raises(CapBreachError) as exc_info:
            await runner.run(user_message="Q")
        assert exc_info.value.cap == "max_iterations"
