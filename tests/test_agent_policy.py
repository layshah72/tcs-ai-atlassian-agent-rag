from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from langchain_core.tools import StructuredTool
from mcp.client.auth import OAuthClientProvider

import atlassian_agent
from atlassian_agent import (
    ALLOWED_TOOLS,
    APPROVAL_REQUIRED_TOOLS,
    AgentError,
    AtlassianAgent,
    _canonical_tool_name,
    _find_mcp_tool,
    KeyringOAuthTokenStorage,
)
from atlassian_chatbot.agent import build_system_prompt
from config import AppConfig, ConfigError


def make_config(**overrides):
    values = {
        "groq_api_key": "test-key",
        "groq_model": "llama-3.3-70b-versatile",
        "mcp_transport": "streamable-http",
        "mcp_url": "https://example.invalid/mcp",
        "atlassian_site_url": "https://example.atlassian.net",
        "atlassian_cloud_id": "test-cloud-id",
        "mcp_auth_type": "bearer",
        "mcp_bearer_token": "test-bearer-token",
        "mcp_basic_email": None,
        "mcp_personal_api_token": None,
        "mcp_command": None,
        "mcp_args": (),
        "mcp_env": {},
        "mcp_timeout_seconds": 30,
        "max_tool_rounds": 6,
    }
    values.update(overrides)
    return AppConfig(**values)


def test_only_expected_atlassian_tools_are_allowed():
    assert ALLOWED_TOOLS == {
        "confluence_search",
        "confluence_get_page",
        "confluence_get_page_children",
        "jira_search",
        "jira_get_issue",
        "jira_get_project_issues",
        "jira_create_issue",
        "jira_update_issue",
    }


def test_all_jira_write_tools_require_approval():
    assert APPROVAL_REQUIRED_TOOLS == {"jira_create_issue", "jira_update_issue"}


@pytest.mark.parametrize(
    ("remote_name", "app_name"),
    [
        ("searchConfluence", "confluence_search"),
        ("getConfluenceContent", "confluence_get_page"),
        ("getJiraIssue", "jira_get_issue"),
        ("searchJiraIssuesUsingJql", "jira_search"),
        ("createJiraIssue", "jira_create_issue"),
        ("editJiraIssue", "jira_update_issue"),
    ],
)
def test_atlassian_published_tool_names_are_mapped(remote_name, app_name):
    assert _canonical_tool_name(remote_name) == app_name


def test_unrecognized_mcp_tool_is_not_allowed():
    assert _canonical_tool_name("executeDestructive") is None


def test_write_tool_can_be_found_by_app_alias():
    def create_jira_issue(project_key: str, summary: str) -> str:
        return f"{project_key}: {summary}"

    remote_tool = StructuredTool.from_function(
        func=create_jira_issue,
        name="createJiraIssue",
        description="Create a Jira issue",
    )
    assert _find_mcp_tool([remote_tool], "jira_create_issue") is remote_tool


def test_alias_tool_invokes_original_remote_tool():
    calls = []

    async def remote_jira_search(jql: str) -> str:
        calls.append(("searchJiraIssuesUsingJql", jql))
        return "found one issue"

    remote_tool = StructuredTool.from_function(
        coroutine=remote_jira_search,
        name="searchJiraIssuesUsingJql",
        description="Search Jira issues",
    )
    alias = AtlassianAgent._alias_tool(remote_tool, "jira_search")

    result = asyncio.run(alias.ainvoke({"jql": "assignee = currentUser()"}))

    assert result == "found one issue"
    assert calls == [("searchJiraIssuesUsingJql", "assignee = currentUser()")]


def test_approved_action_requires_write_tool():
    agent = AtlassianAgent(make_config())
    with pytest.raises(AgentError, match="approval-gated"):
        agent.execute_approved_action("jira_search", {})


def test_write_tool_wrapper_only_stages_draft():
    staged = []

    def update_issue(issue_key: str, summary: str) -> str:
        return f"This must never execute before approval: {issue_key} {summary}"

    source_tool = StructuredTool.from_function(
        func=update_issue,
        name="jira_update_issue",
        description="Update a Jira issue",
    )
    draft_tool = AtlassianAgent._draft_tool(source_tool, staged)

    result = draft_tool.invoke({"issue_key": "TEST-1", "summary": "Updated title"})

    assert staged == [
        {
            "name": "jira_update_issue",
            "arguments": {"issue_key": "TEST-1", "summary": "Updated title"},
        }
    ]
    assert "NOT executed" in result


def test_config_requires_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        AppConfig.from_environment()


def test_config_rejects_groq_example_placeholder(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "replace-with-your-groq-api-key")
    with pytest.raises(ConfigError, match="still a placeholder"):
        AppConfig.from_environment()


def test_config_requires_endpoint_for_http(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "test-cloud-id")
    monkeypatch.delenv("ATLASSIAN_MCP_URL", raising=False)
    with pytest.raises(ConfigError, match="ATLASSIAN_MCP_URL"):
        AppConfig.from_environment()


def test_http_transport_requires_bearer_token(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "test-cloud-id")
    monkeypatch.setenv("ATLASSIAN_MCP_AUTH_TYPE", "bearer")
    monkeypatch.delenv("ATLASSIAN_MCP_BEARER_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="ATLASSIAN_MCP_BEARER_TOKEN"):
        AppConfig.from_environment()


def test_bearer_token_is_sent_in_authorization_header():
    config = make_config()
    client = atlassian_agent._make_mcp_client(config)
    connection = client.connections["atlassian"]
    assert connection["headers"]["Authorization"] == "Bearer test-bearer-token"
    assert connection.get("auth") is None


def test_personal_api_token_uses_basic_auth_header():
    config = make_config(
        mcp_auth_type="basic",
        mcp_basic_email="agent@example.com",
        mcp_personal_api_token="personal-api-token",
    )
    client = atlassian_agent._make_mcp_client(config)
    connection = client.connections["atlassian"]
    assert connection["headers"]["Authorization"] == "Basic YWdlbnRAZXhhbXBsZS5jb206cGVyc29uYWwtYXBpLXRva2Vu"


def test_basic_auth_can_be_configured_at_runtime(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "test-cloud-id")
    monkeypatch.setenv("ATLASSIAN_MCP_AUTH_TYPE", "basic")
    monkeypatch.delenv("ATLASSIAN_MCP_EMAIL", raising=False)
    monkeypatch.delenv("ATLASSIAN_MCP_PERSONAL_API_TOKEN", raising=False)
    config = AppConfig.from_environment()

    assert config.mcp_auth_type == "basic"
    assert config.mcp_basic_email is None
    assert config.mcp_personal_api_token is None


def test_oauth2_is_default_and_does_not_require_static_token(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "test-cloud-id")
    monkeypatch.delenv("ATLASSIAN_MCP_AUTH_TYPE", raising=False)
    monkeypatch.delenv("ATLASSIAN_MCP_BEARER_TOKEN", raising=False)

    config = AppConfig.from_environment()

    assert config.mcp_auth_type == "oauth2"
    assert config.mcp_oauth_redirect_uri == "http://127.0.0.1:8765/oauth/callback"


def test_oauth2_rejects_non_local_redirect_uri(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "test-cloud-id")
    monkeypatch.setenv("ATLASSIAN_OAUTH_REDIRECT_URI", "https://evil.example/callback")

    with pytest.raises(ConfigError, match="localhost HTTP callback URL"):
        AppConfig.from_environment()


def test_oauth2_mcp_connection_uses_oauth_provider_without_static_header():
    config = make_config(mcp_auth_type="oauth2")

    client = atlassian_agent._make_mcp_client(config)
    connection = client.connections["atlassian"]

    assert isinstance(connection["auth"], OAuthClientProvider)
    assert connection["headers"] == {}


def test_httpx_factory_preserves_auth_provider():
    auth = httpx.BasicAuth("unused", "test")

    client = atlassian_agent._httpx_client_factory(auth=auth)

    assert client._auth is auth


def test_dpapi_oauth_storage_round_trips_large_tokens(monkeypatch, tmp_path):
    def fake_crypt(data, *, protect, entropy):
        if protect:
            return b"encrypted:" + entropy + b":" + data
        prefix = b"encrypted:" + entropy + b":"
        assert data.startswith(prefix)
        return data[len(prefix) :]

    monkeypatch.setattr(
        atlassian_agent.DPAPIOAuthTokenStorage,
        "_crypt",
        staticmethod(fake_crypt),
    )
    storage = atlassian_agent.DPAPIOAuthTokenStorage("test-session")
    storage.directory = tmp_path
    token = atlassian_agent.OAuthToken(
        access_token="a" * 6000, refresh_token="refresh"
    )

    asyncio.run(storage.set_tokens(token))

    assert asyncio.run(storage.get_tokens()) == token
    assert storage._path("tokens").read_bytes() != token.model_dump_json().encode()


def test_site_url_resolves_cloud_id_from_tenant_info(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_MCP_BEARER_TOKEN", "test-token")
    monkeypatch.setenv("ATLASSIAN_SITE_URL", "https://acme.atlassian.net/")
    monkeypatch.delenv("ATLASSIAN_CLOUD_ID", raising=False)

    class FakeResponse:
        def raise_for_status(self):
            return None

        @staticmethod
        def json():
            return {"cloudId": "resolved-cloud-id"}

    requested = []

    def fake_get(url, **kwargs):
        requested.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr("config.httpx.get", fake_get)
    config = AppConfig.from_environment()

    assert config.atlassian_site_url == "https://acme.atlassian.net"
    assert config.atlassian_cloud_id == "resolved-cloud-id"
    assert requested[0][0] == "https://acme.atlassian.net/_edge/tenant_info"


def test_explicit_cloud_id_environment_value_takes_precedence(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATLASSIAN_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.setenv("ATLASSIAN_SITE_URL", "https://acme.atlassian.net")
    monkeypatch.setenv("ATLASSIAN_CLOUD_ID", "configured-cloud-id")
    monkeypatch.setenv("ATLASSIAN_MCP_AUTH_TYPE", "basic")

    def should_not_resolve(*args, **kwargs):
        raise AssertionError("Cloud ID should not be looked up when explicitly configured")

    monkeypatch.setattr("config.httpx.get", should_not_resolve)
    config = AppConfig.from_environment()

    assert config.atlassian_cloud_id == "configured-cloud-id"
    assert "configured-cloud-id" in caplog.text
    assert "ATLASSIAN_CLOUD_ID environment variable" in caplog.text


def test_tool_alias_makes_site_context_optional_and_injects_it():
    async def remote_search(jql: str, cloudId: str, siteUrl: str) -> str:
        return f"{jql}|{cloudId}|{siteUrl}"

    remote_tool = StructuredTool.from_function(
        coroutine=remote_search,
        name="searchJiraIssuesUsingJql",
        description="Search Jira issues",
    )
    config = make_config()
    alias = AtlassianAgent._alias_tool(remote_tool, "jira_search", config)

    assert "cloudId" not in alias.args_schema.model_json_schema().get("required", [])
    assert "siteUrl" not in alias.args_schema.model_json_schema().get("required", [])
    result = asyncio.run(
        alias.ainvoke(
            {
                "jql": "assignee = currentUser()",
                "cloudId": "wrong-cloud-id",
                "siteUrl": "https://wrong-site.atlassian.net",
            }
        )
    )
    assert result == "assignee = currentUser()|test-cloud-id|https://example.atlassian.net"


def test_agent_system_prompt_does_not_suggest_cloud_id_as_tool_argument():
    prompt = build_system_prompt(make_config(atlassian_cloud_id="prompt-cloud-id"))

    assert "prompt-cloud-id" not in prompt
    assert "cloudId" not in prompt
    assert "atlassian_retrieve" in prompt
