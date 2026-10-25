"""Legacy import facade; new code should import from atlassian_chatbot."""

from mcp.shared.auth import OAuthToken

from atlassian_chatbot.agent import *
from atlassian_chatbot.agent import (
    _canonical_tool_name,
    _exception_details,
    _find_mcp_tool,
    _inject_site_context,
    _site_aware_args_schema,
    _site_context_value,
    _tool_result_text,
)
from atlassian_chatbot.auth import (
    AtlassianAuth,
    AuthenticationError,
    DPAPIOAuthTokenStorage,
    KeyringOAuthTokenStorage,
    oauth_redirect_handlers,
)
from atlassian_chatbot.config import AppConfig, ConfigError
from atlassian_chatbot.mcp import (
    MCPConnection,
    MCPConnectionError,
    create_mcp_client,
    httpx_client_factory,
)


def _make_mcp_client(config: AppConfig):
    return create_mcp_client(config)


_httpx_client_factory = httpx_client_factory
_oauth_redirect_handlers = oauth_redirect_handlers
_oauth_provider = AtlassianAuth().oauth_provider
