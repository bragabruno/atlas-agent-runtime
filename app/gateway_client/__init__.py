"""Concrete gateway client adapters (AGT-16).

`HttpGatewayClient` is the production implementation of the `GatewayClient`
protocol declared in `app.loop.gateway_client` — it talks to the Atlas Gateway
over HTTP. The agent loop depends only on the protocol, never on this module.
"""

from __future__ import annotations

from app.gateway_client.http_client import HttpGatewayClient

__all__ = ["HttpGatewayClient"]
