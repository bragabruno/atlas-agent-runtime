"""HttpGatewayClient — GatewayClient backed by httpx (AGT-16).

Implements the `GatewayClient` protocol (`app.loop.gateway_client`) by calling
the Atlas Gateway's OpenAI-compatible ``POST /v1/chat/completions``. Translates
the loop's `LLMRequest` into a chat-completion request and maps the gateway's
`Choice`/`CompletionUsage` response back into the loop's `LLMResponse`.

The `system_prompt` carried on `LLMRequest` is sent as a leading system-role
message. With the local gateway running ``model=mock`` (no provider keys,
MockProvider — ADR-012) the message content is echoed/canned, which is exactly
what the offline dev loop needs.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.loop.gateway_client import LLMRequest, LLMResponse

_CHAT_PATH = "/v1/chat/completions"


class HttpGatewayClient:
    """Async gateway client.

    Parameters
    ----------
    base_url:
        Gateway origin, e.g. ``http://gateway:8000``.
    api_key:
        Bearer token sent in the ``Authorization`` header.
    timeout_s:
        Per-request timeout in seconds.
    client:
        Optional injected ``httpx.AsyncClient`` (tests pass one backed by a
        ``MockTransport``). When ``None`` a client is created per request.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client = client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send one LLM request to the gateway and normalize the response."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": request.system_prompt},
            *request.messages,
        ]
        payload: dict[str, object] = {
            "model": request.model_alias,
            "messages": messages,
            "stream": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}{_CHAT_PATH}"

        if self._client is not None:
            resp = await self._client.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=self._timeout)

        resp.raise_for_status()
        return _to_llm_response(resp.json())


def _to_llm_response(data: dict[str, Any]) -> LLMResponse:
    """Map an OpenAI-compatible chat-completion body to an `LLMResponse`.

    Fails fast (KeyError/IndexError) on a malformed body rather than silently
    returning an empty response — a missing choice is a real upstream error.
    """
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return LLMResponse(
        content=choice["message"]["content"],
        finish_reason=choice.get("finish_reason", "stop") or "stop",
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
