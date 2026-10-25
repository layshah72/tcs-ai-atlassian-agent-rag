from __future__ import annotations

import os
from base64 import b64encode
from dataclasses import dataclass
from typing import Mapping


class AuthConfigurationError(ValueError):
    """Raised when a selected authentication mode lacks valid credentials."""


@dataclass(frozen=True)
class BasicAuthConfig:
    """Configuration and header generation for Atlassian Basic authentication."""

    email: str | None
    personal_api_token: str | None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "BasicAuthConfig":
        """Load Atlassian Basic credentials from environment variables."""
        source = os.environ if environ is None else environ
        return cls(
            email=source.get("ATLASSIAN_MCP_EMAIL", "").strip() or None,
            personal_api_token=source.get("ATLASSIAN_MCP_PERSONAL_API_TOKEN", "").strip() or None,
        )

    def authorization_header(self) -> str:
        """Return the HTTP Basic Authorization header."""
        if not self.email or not self.personal_api_token:
            raise AuthConfigurationError(
                "ATLASSIAN_MCP_EMAIL and ATLASSIAN_MCP_PERSONAL_API_TOKEN are required for Basic authentication."
            )
        credentials = f"{self.email}:{self.personal_api_token}".encode("utf-8")
        return f"Basic {b64encode(credentials).decode('ascii')}"
