from __future__ import annotations

import asyncio
import hashlib

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from atlassian_chatbot.agent.service import AtlassianAgent
from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.rag import AtlassianFaissStore


class FakeEmbeddingModel:
    def embed(self, texts):
        return (self._vector(text) for text in texts)

    def query_embed(self, query):
        return iter([self._vector(query)])

    @staticmethod
    def _vector(text):
        vector = np.zeros(64, dtype="float32")
        for token in text.lower().split():
            index = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % 64
            vector[index] += 1
        return vector


def make_config() -> AppConfig:
    return AppConfig(
        groq_api_key="test-key",
        groq_model="test-model",
        mcp_transport="streamable-http",
        mcp_url="https://example.invalid/mcp",
        atlassian_site_url="https://example.atlassian.net",
        atlassian_cloud_id="test-cloud-id",
        mcp_auth_type="bearer",
        mcp_bearer_token="test-token",
        mcp_basic_email=None,
        mcp_personal_api_token=None,
        mcp_command=None,
        mcp_args=(),
        mcp_env={},
        mcp_timeout_seconds=30,
        max_tool_rounds=6,
    )


def make_populated_store() -> AtlassianFaissStore:
    store = AtlassianFaissStore(embedding_model=FakeEmbeddingModel())
    store.add_result("jira_search", "PAY-1 checkout payment issue is open")
    return store


def install_agent_spy(monkeypatch):
    observed = {}

    class FakeAgent:
        async def ainvoke(self, payload, config):
            observed["messages"] = payload["messages"]
            return {"messages": [AIMessage(content="Answer from test fixture") ]}

    def fake_create_agent(llm, tools, *, system_prompt):
        observed["tool_names"] = [tool.name for tool in tools]
        observed["system_prompt"] = system_prompt
        return FakeAgent()

    monkeypatch.setattr("atlassian_chatbot.agent.service.create_agent", fake_create_agent)
    return observed


def test_indexed_read_prompt_skips_mcp_discovery_and_search(monkeypatch):
    observed = install_agent_spy(monkeypatch)

    def unexpected_mcp_call(config):
        raise AssertionError("MCP should not be connected for an indexed read-only prompt")

    monkeypatch.setattr("atlassian_chatbot.agent.service.create_mcp_client", unexpected_mcp_call)
    agent = AtlassianAgent.__new__(AtlassianAgent)
    agent.config = make_config()
    agent.llm = object()

    result = asyncio.run(
        agent._answer(
            "What is the status of PAY-1?",
            [],
            document_store=make_populated_store(),
        )
    )

    assert result.text == "Answer from test fixture"
    assert observed["tool_names"] == ["atlassian_retrieve"]
    assert "already been indexed" in observed["system_prompt"]


def test_indexed_write_prompt_keeps_approval_tools_but_not_mcp_reads(monkeypatch):
    observed = install_agent_spy(monkeypatch)

    async def search_jira(jql: str) -> str:
        return "PAY-1"

    async def create_issue(project_key: str, summary: str) -> str:
        return "must not execute before approval"

    tools = [
        StructuredTool.from_function(
            coroutine=search_jira,
            name="searchJiraIssuesUsingJql",
            description="Search Jira issues",
        ),
        StructuredTool.from_function(
            coroutine=create_issue,
            name="createJiraIssue",
            description="Create Jira issue",
        ),
    ]

    class FakeMCPClient:
        async def get_tools(self, server_name):
            return tools

    monkeypatch.setattr(
        "atlassian_chatbot.agent.service.create_mcp_client",
        lambda config: FakeMCPClient(),
    )
    agent = AtlassianAgent.__new__(AtlassianAgent)
    agent.config = make_config()
    agent.llm = object()

    asyncio.run(
        agent._answer(
            "Create an issue in PAY with summary Fix checkout",
            [],
            document_store=make_populated_store(),
        )
    )

    assert set(observed["tool_names"]) == {"jira_create_issue", "atlassian_retrieve"}
    assert "jira_search" not in observed["tool_names"]
