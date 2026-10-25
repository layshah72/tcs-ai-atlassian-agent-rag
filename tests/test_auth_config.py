from __future__ import annotations

import base64

import pytest

from atlassian_chatbot.auth import (
    AuthConfigurationError,
    BasicAuthConfig,
    BearerTokenAuthConfig,
)


def test_basic_auth_config_loads_environment_and_builds_header():
    config = BasicAuthConfig.from_environment(
        {
            "ATLASSIAN_MCP_EMAIL": "user@example.com",
            "ATLASSIAN_MCP_PERSONAL_API_TOKEN": "api-token",
        }
    )

    expected = base64.b64encode(b"user@example.com:api-token").decode("ascii")
    assert config.authorization_header() == f"Basic {expected}"


def test_basic_auth_config_requires_both_values():
    with pytest.raises(AuthConfigurationError, match="ATLASSIAN_MCP_EMAIL"):
        BasicAuthConfig(email="user@example.com", personal_api_token=None).authorization_header()


def test_bearer_auth_config_loads_environment_and_builds_header():
    config = BearerTokenAuthConfig.from_environment({"ATLASSIAN_MCP_BEARER_TOKEN": "access-token"})

    assert config.authorization_header() == "Bearer access-token"


def test_bearer_auth_config_requires_token():
    with pytest.raises(AuthConfigurationError, match="ATLASSIAN_MCP_BEARER_TOKEN"):
        BearerTokenAuthConfig.from_environment({}).authorization_header()
