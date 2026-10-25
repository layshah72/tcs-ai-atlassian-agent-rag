"""Backward-compatible imports; implementation lives in atlassian_chatbot.mcp."""

from atlassian_chatbot.mcp.connection import (
    MCPConnection,
    MCPConnectionError,
    create_mcp_client,
    httpx_client_factory,
)

_make_mcp_client = create_mcp_client
_httpx_client_factory = httpx_client_factory
