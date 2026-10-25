from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.agent.policy import inject_site_context
from atlassian_chatbot.rag import AtlassianFaissStore, AtlassianIngestionPipeline, IngestionError


class FakeEmbeddingModel:
    def embed(self, documents):
        import numpy as np

        return (np.ones(64, dtype="float32") for _ in documents)

    def query_embed(self, query):
        import numpy as np

        return iter([np.ones(64, dtype="float32")])


class FakeMCPClient:
    def __init__(self, tools):
        self.tools = tools

    async def get_tools(self, server_name):
        assert server_name == "atlassian"
        return self.tools


def make_config():
    return AppConfig(
        groq_api_key="test-key",
        groq_model="test-model",
        mcp_transport="streamable-http",
        mcp_url="https://example.invalid/mcp",
        atlassian_site_url="https://example.atlassian.net",
        atlassian_cloud_id="configured-cloud-id",
        mcp_auth_type="bearer",
        mcp_bearer_token="token",
        mcp_basic_email=None,
        mcp_personal_api_token=None,
        mcp_command=None,
        mcp_args=(),
        mcp_env={},
        mcp_timeout_seconds=30,
        max_tool_rounds=6,
    )


def make_store():
    return AtlassianFaissStore(embedding_model=FakeEmbeddingModel())


def test_refresh_indexes_jira_and_confluence_with_schema_supported_arguments():
    calls = []

    async def jira_search(jql: str, cloudId: str, maxResults: int = 50) -> str:
        calls.append(("jira", jql, maxResults, cloudId))
        return "PAY-1 Payment timeout on checkout"

    async def confluence_search(query: str, limit: int = 10) -> str:
        calls.append(("confluence", query, limit))
        return "Checkout runbook: retry a payment after timeout"

    tools = [
        StructuredTool.from_function(
            coroutine=jira_search,
            name="searchJiraIssuesUsingJql",
            description="Search Jira",
        ),
        StructuredTool.from_function(
            coroutine=confluence_search,
            name="searchConfluence",
            description="Search Confluence",
        ),
    ]
    client = FakeMCPClient(tools)
    pipeline = AtlassianIngestionPipeline(
        make_config(),
        client_factory=lambda config: client,
        store_factory=make_store,
    )

    result = pipeline.refresh(
        "assignee = currentUser() AND updated >= -1d ORDER BY updated DESC",
        "type = page AND lastModified >= startOfDay()",
    )

    assert calls == [
        ("jira", "assignee = currentUser() AND updated >= -1d ORDER BY updated DESC", 100, "configured-cloud-id"),
        ("confluence", "type = page AND lastModified >= startOfDay()", 100),
    ]
    assert result.jira_chunks == 1
    assert result.confluence_chunks == 1
    assert result.store.chunk_count == 2
    assert result.store.search("payment checkout timeout")


def test_refresh_keeps_tool_schema_and_query_requirements_explicit():
    async def jira_search(jql: str, unsupported_required: str) -> str:
        return "PAY-1"

    async def confluence_search(query: str) -> str:
        return "Page content"

    tools = [
        StructuredTool.from_function(
            coroutine=jira_search, name="jira_search", description="Search Jira"
        ),
        StructuredTool.from_function(
            coroutine=confluence_search, name="confluence_search", description="Search Confluence"
        ),
    ]
    pipeline = AtlassianIngestionPipeline(
        make_config(),
        client_factory=lambda config: FakeMCPClient(tools),
        store_factory=make_store,
    )

    with pytest.raises(IngestionError, match="unsupported parameter"):
        pipeline.refresh("project = PAY", "type = page")


def test_refresh_requires_both_search_sources():
    async def jira_search(jql: str) -> str:
        return "PAY-1"

    tools = [
        StructuredTool.from_function(
            coroutine=jira_search, name="jira_search", description="Search Jira"
        )
    ]
    pipeline = AtlassianIngestionPipeline(
        make_config(),
        client_factory=lambda config: FakeMCPClient(tools),
        store_factory=make_store,
    )

    with pytest.raises(IngestionError, match="Confluence search tool"):
        pipeline.refresh("project = PAY", "type = page")


def test_raw_json_schema_cloud_id_is_injected_before_call():
    class JsonSchemaTool:
        name = "jira_search"
        args_schema = {
            "type": "object",
            "properties": {
                "jql": {"type": "string"},
                "cloudId": {"type": "string"},
            },
            "required": ["jql", "cloudId"],
        }

    tool = JsonSchemaTool()
    arguments = inject_site_context({"jql": "project = PAY"}, tool, make_config())

    assert arguments == {"jql": "project = PAY", "cloudId": "configured-cloud-id"}


def test_refresh_does_not_index_mcp_validation_error_payload():
    async def jira_search(jql: str, cloudId: str) -> dict:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation error: cloudId field required"}],
        }

    async def confluence_search(query: str) -> str:
        return "Confluence page data"

    tools = [
        StructuredTool.from_function(
            coroutine=jira_search,
            name="searchJiraIssuesUsingJql",
            description="Search Jira",
        ),
        StructuredTool.from_function(
            coroutine=confluence_search,
            name="searchConfluence",
            description="Search Confluence",
        ),
    ]
    pipeline = AtlassianIngestionPipeline(
        make_config(),
        client_factory=lambda config: FakeMCPClient(tools),
        store_factory=make_store,
    )

    with pytest.raises(IngestionError, match="MCP error"):
        pipeline.refresh("project = PAY", "type = page")


def test_vector_store_refuses_error_payloads():
    store = make_store()

    with pytest.raises(RuntimeError, match="MCP error"):
        store.add_result(
            "jira_search",
            {"isError": True, "content": [{"text": "cloudId field required"}]},
        )

    assert store.chunk_count == 0
