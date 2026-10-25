"""Atlassian authentication providers and local OAuth storage."""

from .oauth import AtlassianAuth, oauth_redirect_handlers
from .storage import AuthenticationError, DPAPIOAuthTokenStorage, FileOAuthTokenStorage, KeyringOAuthTokenStorage, make_oauth_token_storage
from .basic import AuthConfigurationError, BasicAuthConfig
from .bearer import BearerTokenAuthConfig

__all__ = [
    "AtlassianAuth",
    "AuthConfigurationError",
    "AuthenticationError",
    "BasicAuthConfig",
    "BearerTokenAuthConfig",
    "DPAPIOAuthTokenStorage",
    "FileOAuthTokenStorage",
    "KeyringOAuthTokenStorage",
    "make_oauth_token_storage",
    "oauth_redirect_handlers",
]
