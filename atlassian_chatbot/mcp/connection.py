from __future__ import annotations

from typing import Any

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from atlassian_chatbot.auth import AtlassianAuth
from atlassian_chatbot.config import AppConfig


class MCPConnectionError(RuntimeError):
    """Raised when the configured MCP transport cannot be created."""


def httpx_client_factory(
    *, auth: Any = None, headers: dict[str, str] | None = None, timeout: Any = None
) -> httpx.AsyncClient:
    """Create the MCP HTTP client and preserve OAuth auth hooks."""
    return httpx.AsyncClient(
        headers=headers,
        auth=auth,
        timeout=timeout or httpx.Timeout(30.0, read=300.0),
    )


class MCPConnection:
    """Build the configured Atlassian MCP client."""

    def __init__(self, config: AppConfig, auth: AtlassianAuth | None = None):
        self.config = config
        self.auth = auth or AtlassianAuth()

    def create_client(self) -> MultiServerMCPClient:
        if self.config.mcp_transport == "stdio":
            raise MCPConnectionError(
                "The LangChain agent currently expects the hosted Streamable HTTP MCP transport. "
                "Set ATLASSIAN_MCP_TRANSPORT=streamable-http for this Atlassian connection."
            )
        connection: dict[str, Any] = {
            "transport": "streamable_http",
            "url": self.config.mcp_url,
            "headers": {},
            "timeout": self.config.mcp_timeout_seconds,
            "httpx_client_factory": httpx_client_factory,
        }
        connection.update(self.auth.connection_options(self.config))
        return MultiServerMCPClient({"atlassian": connection})


def create_mcp_client(config: AppConfig) -> MultiServerMCPClient:
    return MCPConnection(config).create_client()
