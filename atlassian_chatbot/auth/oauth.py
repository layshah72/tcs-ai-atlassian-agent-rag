from __future__ import annotations

import asyncio
from base64 import b64encode
import webbrowser
from urllib.parse import parse_qs, urlparse, urlsplit

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from atlassian_chatbot.config import AppConfig, ConfigError
from .storage import DPAPIOAuthTokenStorage, AuthenticationError, make_oauth_token_storage
from .basic import AuthConfigurationError, BasicAuthConfig
from .bearer import BearerTokenAuthConfig


def oauth_redirect_handlers(redirect_uri: str):
    """Create browser and localhost callback handlers for OAuth authorization-code flow."""
    parsed = urlparse(redirect_uri)
    callback_host, callback_port, callback_path = parsed.hostname, parsed.port, parsed.path
    if not callback_host or callback_port is None:
        raise AuthenticationError("The OAuth callback URL must include a localhost host and explicit port.")

    callback_future: asyncio.Future[tuple[str, str | None]] | None = None
    callback_server: asyncio.Server | None = None

    async def redirect_handler(authorization_url: str) -> None:
        nonlocal callback_future, callback_server
        callback_future = asyncio.get_running_loop().create_future()

        async def handle_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                request_line = await asyncio.wait_for(reader.readline(), timeout=20)
                parts = request_line.decode("ascii", "replace").split()
                target = urlsplit(parts[1]) if len(parts) >= 2 else None
                if target is None or target.path != callback_path:
                    status, text = "HTTP/1.1 404 Not Found", "OAuth callback path not found."
                else:
                    params = parse_qs(target.query)
                    code, state = params.get("code", [""])[0], params.get("state", [""])[0] or None
                    status = "HTTP/1.1 200 OK"
                    text = (
                        "Atlassian authorization was received. You can return to the chatbot."
                        if code and not params.get("error")
                        else "Atlassian authorization was not completed. Return to the chatbot for details."
                    )
                    if callback_future is not None and not callback_future.done():
                        callback_future.set_result((code, state))
                body = text.encode("utf-8")
                writer.write(
                    f"{status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii") + body
                )
                await writer.drain()
            except Exception as exc:
                if callback_future is not None and not callback_future.done():
                    callback_future.set_exception(exc)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

        try:
            callback_server = await asyncio.start_server(handle_callback, callback_host, callback_port)
        except OSError as exc:
            raise AuthenticationError(
                f"Could not start the OAuth callback listener on {callback_host}:{callback_port}. "
                "Close the process using that port or set ATLASSIAN_OAUTH_REDIRECT_URI to another localhost port."
            ) from exc
        if not await asyncio.to_thread(webbrowser.open, authorization_url):
            callback_server.close()
            await callback_server.wait_closed()
            raise AuthenticationError("Could not open a browser for Atlassian sign-in. Check browser configuration and retry.")

    async def callback_handler() -> tuple[str, str | None]:
        if callback_future is None:
            raise AuthenticationError("The Atlassian OAuth callback listener was not started.")
        try:
            return await asyncio.wait_for(callback_future, timeout=300)
        finally:
            if callback_server is not None:
                callback_server.close()
                await callback_server.wait_closed()

    return redirect_handler, callback_handler


class AtlassianAuth:
    """Select and construct OAuth, Bearer, or Basic authentication for MCP."""

    def connection_options(self, config: AppConfig) -> dict[str, object]:
        """Return MCP client options for the selected authentication mode."""
        if config.mcp_auth_type == "oauth2":
            return {"auth": self.oauth_provider(config)}
        return {"headers": {"Authorization": self.authorization_header(config)}}

    def oauth_provider(self, config: AppConfig) -> OAuthClientProvider:
        redirect, callback = oauth_redirect_handlers(config.mcp_oauth_redirect_uri)
        return OAuthClientProvider(
            server_url=config.mcp_url or "",
            client_metadata=OAuthClientMetadata(
                client_name="TCS Confluent Chatbot",
                redirect_uris=[AnyUrl(config.mcp_oauth_redirect_uri)],
            ),
            storage=make_oauth_token_storage(config.mcp_oauth_storage_key),
            redirect_handler=redirect,
            callback_handler=callback,
        )

    @staticmethod
    def authorization_header(config: AppConfig) -> str:
        """Delegate static header generation to the selected auth config class."""
        if config.mcp_auth_type == "bearer":
            return BearerTokenAuthConfig(config.mcp_bearer_token).authorization_header()
        if config.mcp_auth_type == "oauth2":
            raise ConfigError("OAuth 2.0 credentials are provided through the MCP OAuth flow, not a static header.")
        return BasicAuthConfig(config.mcp_basic_email, config.mcp_personal_api_token).authorization_header()
