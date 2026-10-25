"""Backward-compatible imports; implementation lives in atlassian_chatbot.auth."""

from atlassian_chatbot.auth import (
    AtlassianAuth,
    AuthenticationError,
    DPAPIOAuthTokenStorage,
    KeyringOAuthTokenStorage,
    oauth_redirect_handlers,
)

_oauth_redirect_handlers = oauth_redirect_handlers
