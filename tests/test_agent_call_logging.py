from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import StructuredTool

from atlassian_chatbot.agent.call_logging import (
    reset_current_prompt,
    reset_current_system_prompt,
    set_current_prompt,
    set_current_system_prompt,
)
from atlassian_chatbot.agent.tools import make_draft_tool, make_read_tool
from atlassian_chatbot.config import AppConfig


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


def test_read_tool_logs_prompt_and_parameters_and_redacts_secrets(caplog):
    caplog.set_level(logging.INFO, logger="atlassian_chatbot.call_logging")

    async def remote_search(jql: str, access_token: str) -> str:
        return "ok"

    source_tool = StructuredTool.from_function(
        coroutine=remote_search,
        name="jira_search",
        description="Search Jira",
    )
    wrapped = make_read_tool(source_tool, "jira_search")
    prompt_token = set_current_prompt("Find my assigned tickets using token ATATTsecretValue")
    system_prompt_token = set_current_system_prompt("Use configured cloudId: cloud-id-123")
    try:
        asyncio.run(
            wrapped.ainvoke({"jql": "assignee = currentUser()", "access_token": "secret-value"})
        )
    finally:
        reset_current_prompt(prompt_token)
        reset_current_system_prompt(system_prompt_token)

    assert "Find my assigned tickets" in caplog.text
    assert "Use configured cloudId: cloud-id-123" in caplog.text
    assert "system_prompt=" in caplog.text
    assert 'accepted_parameters=["jql", "access_token"]' in caplog.text
    assert 'passed_parameters={"jql": "assignee = currentUser()", "access_token": "[REDACTED]"}' in caplog.text
    assert "assignee = currentUser()" in caplog.text
    assert "ATATTsecretValue" not in caplog.text
    assert "secret-value" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_draft_tool_retains_originating_prompt_for_approval_logs():
    def remote_update(issue_key: str, summary: str) -> str:
        raise AssertionError("Draft tool must not invoke the source tool")

    source_tool = StructuredTool.from_function(
        func=remote_update,
        name="jira_update_issue",
        description="Update Jira issue",
    )
    pending_actions = []
    wrapped = make_draft_tool(source_tool, pending_actions)
    prompt_token = set_current_prompt("Update TEST-1 title")
    system_prompt_token = set_current_system_prompt("Use configured cloudId: cloud-id-123")
    try:
        wrapped.invoke({"issue_key": "TEST-1", "summary": "New title"})
    finally:
        reset_current_prompt(prompt_token)
        reset_current_system_prompt(system_prompt_token)

    assert pending_actions == [
        {
            "name": "jira_update_issue",
            "arguments": {"issue_key": "TEST-1", "summary": "New title"},
            "prompt": "Update TEST-1 title",
        }
    ]
