"""AGT-16 — HttpGatewayClient maps requests/responses correctly (httpx mocked)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.gateway_client import HttpGatewayClient
from app.loop.gateway_client import LLMRequest

_RESPONSE_BODY = {
    "id": "cmpl-1",
    "created": 1,
    "model": "mock",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Article 6 requires consent [src:reg-042]"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "object": "chat.completion",
}


@pytest.mark.asyncio
async def test_chat_maps_request_and_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_RESPONSE_BODY)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as ac:
        client = HttpGatewayClient(base_url="http://gw:8000", api_key="dev-key", client=ac)
        resp = await client.chat(
            LLMRequest(
                model_alias="mock",
                system_prompt="SYS",
                messages=[{"role": "user", "content": "q"}],
            )
        )

    assert resp.content == "Article 6 requires consent [src:reg-042]"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
    assert resp.total_tokens == 15
    assert resp.finish_reason == "stop"

    assert captured["url"] == "http://gw:8000/v1/chat/completions"
    body = captured["json"]
    assert body["model"] == "mock"  # type: ignore[index]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}  # type: ignore[index]
    assert body["messages"][1] == {"role": "user", "content": "q"}  # type: ignore[index]
    assert body["stream"] is False  # type: ignore[index]
    assert captured["auth"] == "Bearer dev-key"


@pytest.mark.asyncio
async def test_chat_includes_max_tokens_when_set() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_RESPONSE_BODY)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as ac:
        client = HttpGatewayClient(base_url="http://gw:8000", api_key="k", client=ac)
        await client.chat(
            LLMRequest(model_alias="mock", system_prompt="s", messages=[], max_tokens=256)
        )

    assert captured["json"]["max_tokens"] == 256  # type: ignore[index]


@pytest.mark.asyncio
async def test_chat_raises_on_http_error() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "boom"}))
    async with httpx.AsyncClient(transport=transport) as ac:
        client = HttpGatewayClient(base_url="http://gw:8000", api_key="k", client=ac)
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(LLMRequest(model_alias="mock", system_prompt="s", messages=[]))
