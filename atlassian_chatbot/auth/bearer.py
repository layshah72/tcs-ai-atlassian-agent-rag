from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .basic import AuthConfigurationError


@dataclass(frozen=True)
class BearerTokenAuthConfig:
    """Configuration and header generation for Atlassian Bearer token auth."""

    token: str | None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "BearerTokenAuthConfig":
        """Load a Bearer token from ATLASSIAN_MCP_BEARER_TOKEN."""
        source = os.environ if environ is None else environ
        return cls(token=source.get("ATLASSIAN_MCP_BEARER_TOKEN", "").strip() or None)

    def authorization_header(self) -> str:
        """Return the HTTP Bearer Authorization header."""
        if not self.token:
            raise AuthConfigurationError("ATLASSIAN_MCP_BEARER_TOKEN is required for Bearer authentication.")
        return f"Bearer {self.token}"
