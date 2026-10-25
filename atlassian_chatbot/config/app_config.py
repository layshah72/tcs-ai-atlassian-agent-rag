from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when required application settings are missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    groq_model: str
    mcp_transport: str
    mcp_url: str | None
    atlassian_site_url: str | None
    atlassian_cloud_id: str | None
    mcp_auth_type: str
    mcp_bearer_token: str | None
    mcp_basic_email: str | None
    mcp_personal_api_token: str | None
    mcp_command: str | None
    mcp_args: tuple[str, ...]
    mcp_env: Mapping[str, str]
    mcp_timeout_seconds: float
    max_tool_rounds: int
    mcp_oauth_redirect_uri: str = "http://127.0.0.1:8765/oauth/callback"
    mcp_oauth_storage_key: str = "default"

    @classmethod
    def from_environment(cls) -> "AppConfig":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("GROQ_API_KEY is not configured.")
        if api_key.lower().startswith(("replace-", "your-", "put-", "enter-")):
            raise ConfigError("GROQ_API_KEY is still a placeholder. Replace it in .env with a valid Groq API key.")

        transport = os.getenv("ATLASSIAN_MCP_TRANSPORT", "streamable-http").strip().lower()
        if transport not in {"streamable-http", "stdio"}:
            raise ConfigError("ATLASSIAN_MCP_TRANSPORT must be 'streamable-http' or 'stdio'.")

        url = os.getenv("ATLASSIAN_MCP_URL", "").strip() or None
        command = os.getenv("ATLASSIAN_MCP_COMMAND", "").strip() or None
        site_url = os.getenv("ATLASSIAN_SITE_URL", "").strip().rstrip("/") or None
        cloud_id = os.getenv("ATLASSIAN_CLOUD_ID", "").strip() or None
        cloud_id_source = "ATLASSIAN_CLOUD_ID environment variable" if cloud_id else None
        if site_url and site_url.lower().startswith(("replace-", "your-", "enter-")):
            raise ConfigError("ATLASSIAN_SITE_URL is a placeholder. Set it to your actual https://<site>.atlassian.net URL.")
        if site_url:
            parsed_site_url = urlparse(site_url)
            if parsed_site_url.scheme != "https" or not parsed_site_url.hostname:
                raise ConfigError("ATLASSIAN_SITE_URL must be a full HTTPS URL, e.g. https://my-site.atlassian.net.")
            if parsed_site_url.path not in {"", "/"} or parsed_site_url.query or parsed_site_url.fragment:
                raise ConfigError("ATLASSIAN_SITE_URL must contain only the site origin, without a path or query.")
        if transport == "streamable-http" and not site_url and not cloud_id:
            raise ConfigError("Set ATLASSIAN_SITE_URL or ATLASSIAN_CLOUD_ID to select your Jira/Confluence site.")
        if transport == "streamable-http" and site_url and not cloud_id:
            cloud_id = _resolve_atlassian_cloud_id(site_url)
            cloud_id_source = f"tenant-info endpoint for {site_url}"
        if cloud_id:
            logger.info("Atlassian cloud ID loaded from %s: %s", cloud_id_source, cloud_id)

        auth_type = os.getenv("ATLASSIAN_MCP_AUTH_TYPE", "oauth2").strip().lower()
        if auth_type not in {"oauth2", "bearer", "basic"}:
            raise ConfigError("ATLASSIAN_MCP_AUTH_TYPE must be 'oauth2', 'bearer', or 'basic'.")
        if auth_type == "oauth2" and transport != "streamable-http":
            raise ConfigError("OAuth 2.0 authentication requires ATLASSIAN_MCP_TRANSPORT=streamable-http.")
        if transport == "streamable-http" and not url:
            raise ConfigError("ATLASSIAN_MCP_URL is required for streamable-http transport.")
        if transport == "stdio" and not command:
            raise ConfigError("ATLASSIAN_MCP_COMMAND is required for stdio transport.")

        # Credential loading and validation belong to auth-mode-specific classes.
        from atlassian_chatbot.auth.basic import AuthConfigurationError, BasicAuthConfig
        from atlassian_chatbot.auth.bearer import BearerTokenAuthConfig

        bearer_auth = BearerTokenAuthConfig.from_environment()
        basic_auth = BasicAuthConfig.from_environment()
        bearer_token = bearer_auth.token
        basic_email = basic_auth.email
        personal_api_token = basic_auth.personal_api_token
        oauth_redirect_uri = os.getenv(
            "ATLASSIAN_OAUTH_REDIRECT_URI", "http://127.0.0.1:8765/oauth/callback"
        ).strip()
        if auth_type == "oauth2":
            try:
                parsed_redirect_uri = urlparse(oauth_redirect_uri)
                valid_redirect_uri = (
                    parsed_redirect_uri.scheme == "http"
                    and parsed_redirect_uri.hostname in {"localhost", "127.0.0.1", "::1"}
                    and parsed_redirect_uri.port is not None
                    and parsed_redirect_uri.port > 0
                    and bool(parsed_redirect_uri.path)
                    and not parsed_redirect_uri.query
                    and not parsed_redirect_uri.fragment
                )
            except ValueError:
                valid_redirect_uri = False
            if not valid_redirect_uri:
                raise ConfigError(
                    "ATLASSIAN_OAUTH_REDIRECT_URI must be a localhost HTTP callback URL, "
                    "such as http://127.0.0.1:8765/oauth/callback."
                )
        if transport == "streamable-http" and auth_type == "bearer" and not bearer_token:
            try:
                bearer_auth.authorization_header()
            except AuthConfigurationError as exc:
                raise ConfigError("ATLASSIAN_MCP_BEARER_TOKEN is required for the Atlassian MCP connection.") from exc

        try:
            args = json.loads(os.getenv("ATLASSIAN_MCP_ARGS", "[]"))
            child_env = json.loads(os.getenv("ATLASSIAN_MCP_ENV", "{}"))
            timeout = float(os.getenv("ATLASSIAN_MCP_TIMEOUT_SECONDS", "30"))
            max_rounds = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigError("MCP args/env must be valid JSON; timeout and tool rounds must be numbers.") from exc
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ConfigError("ATLASSIAN_MCP_ARGS must be a JSON array of strings.")
        if not isinstance(child_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in child_env.items()
        ):
            raise ConfigError("ATLASSIAN_MCP_ENV must be a JSON object of string values.")
        if timeout <= 0 or max_rounds < 1:
            raise ConfigError("MCP timeout and AGENT_MAX_TOOL_ROUNDS must be positive.")

        return cls(
            groq_api_key=api_key,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            mcp_transport=transport,
            mcp_url=url,
            atlassian_site_url=site_url,
            atlassian_cloud_id=cloud_id,
            mcp_auth_type=auth_type,
            mcp_bearer_token=bearer_token,
            mcp_basic_email=basic_email,
            mcp_personal_api_token=personal_api_token,
            mcp_command=command,
            mcp_args=tuple(args),
            mcp_env=child_env,
            mcp_timeout_seconds=timeout,
            max_tool_rounds=max_rounds,
            mcp_oauth_redirect_uri=oauth_redirect_uri,
        )

    def atlassian_authorization_header(self) -> str:
        """Compatibility wrapper; authentication header logic lives in AtlassianAuth."""
        from atlassian_chatbot.auth import AtlassianAuth

        return AtlassianAuth.authorization_header(self)


def _resolve_atlassian_cloud_id(site_url: str) -> str:
    """Resolve a site's cloud ID using Atlassian's tenant-info endpoint."""
    try:
        response = httpx.get(f"{site_url}/_edge/tenant_info", timeout=10.0, follow_redirects=False)
        response.raise_for_status()
        tenant_info = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ConfigError(
            "Could not retrieve ATLASSIAN_CLOUD_ID from ATLASSIAN_SITE_URL. "
            "Check the Jira site URL and network access, or set ATLASSIAN_CLOUD_ID directly."
        ) from exc
    cloud_id = tenant_info.get("cloudId") or tenant_info.get("cloud_id") if isinstance(tenant_info, dict) else None
    if not isinstance(cloud_id, str) or not cloud_id.strip():
        raise ConfigError("Atlassian's tenant-info endpoint did not return a cloudId. Set ATLASSIAN_CLOUD_ID directly in .env.")
    return cloud_id.strip()
