from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from langchain_core.tools import BaseTool
from pydantic import Field, create_model

from atlassian_chatbot.config import AppConfig

ALLOWED_TOOLS = {
    # Confluence read
    "confluence_search", "confluence_get_page", "confluence_get_page_children",
    # Confluence write
    "confluence_create_page", "confluence_update_page",
    # Jira read
    "jira_search", "jira_get_issue", "jira_get_project_issues",
    # Jira write
    "jira_create_issue", "jira_update_issue",
}
MCP_TOOL_ALIASES = {
    "searchConfluence": "confluence_search",
    "getConfluenceContent": "confluence_get_page",
    "getConfluencePageChildren": "confluence_get_page_children",
    "createConfluencePage": "confluence_create_page",
    "updateConfluencePage": "confluence_update_page",
    "getJiraIssue": "jira_get_issue",
    "searchJiraIssuesUsingJql": "jira_search",
    "createJiraIssue": "jira_create_issue",
    "editJiraIssue": "jira_update_issue",
}
APPROVAL_REQUIRED_TOOLS = {
    "jira_create_issue", "jira_update_issue",
    "confluence_create_page", "confluence_update_page",
}
READ_ONLY_TOOLS = ALLOWED_TOOLS - APPROVAL_REQUIRED_TOOLS


def canonical_tool_name(mcp_name: str) -> str | None:
    """Map published Atlassian tool names to the app's stable names."""
    return mcp_name if mcp_name in ALLOWED_TOOLS else MCP_TOOL_ALIASES.get(mcp_name)


def find_mcp_tool(tools: list[BaseTool], canonical_name: str) -> BaseTool | None:
    """Find a server tool by app-facing name, including known Atlassian aliases."""
    remote_names = [remote for remote, app in MCP_TOOL_ALIASES.items() if app == canonical_name]
    remote_names.append(canonical_name)
    return next((tool for tool in tools if tool.name in remote_names), None)


def site_context_value(field_name: str, config: AppConfig) -> str | None:
    normalized = field_name.replace("_", "").lower()
    if normalized in {"cloudid", "atlassiancloudid"}:
        return config.atlassian_cloud_id
    if normalized in {"siteurl", "atlassiansiteurl", "baseurl"}:
        return config.atlassian_site_url
    return None


def tool_schema_fields(mcp_tool: BaseTool) -> Mapping[str, Any]:
    """Read field definitions from Pydantic models or JSON-schema tool inputs."""
    schema = getattr(mcp_tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, Mapping):
        return fields
    if isinstance(schema, Mapping):
        properties = schema.get("properties", {})
        return properties if isinstance(properties, Mapping) else {}
    schema_method = getattr(schema, "model_json_schema", None)
    if callable(schema_method):
        properties = schema_method().get("properties", {})
        return properties if isinstance(properties, Mapping) else {}
    return {}


def site_aware_args_schema(mcp_tool: BaseTool, config: AppConfig) -> type:
    """Copy an MCP schema, making configured site identifiers optional defaults."""
    schema = mcp_tool.args_schema
    if schema is None or not hasattr(schema, "model_fields"):
        return schema
    fields: dict[str, Any] = {}
    for name, model_field in schema.model_fields.items():
        value = site_context_value(name, config)
        fields[name] = (
            model_field.annotation,
            Field(default=value, description=model_field.description, title=model_field.title)
            if value is not None else copy.copy(model_field),
        )
    return create_model(f"{schema.__name__}WithSiteConfig", **fields)


def inject_site_context(arguments: Mapping[str, Any], mcp_tool: BaseTool, config: AppConfig) -> dict[str, Any]:
    """Supply configured cloud/site identifiers when accepted by the remote tool."""
    completed = dict(arguments)
    fields = tool_schema_fields(mcp_tool)
    for field_name in fields:
        value = site_context_value(field_name, config)
        # The configured site is authoritative; never let an LLM-provided
        # cloud/site identifier silently route a request to another tenant.
        if value is not None:
            completed[field_name] = value
    return completed


def tool_result_text(result: Any) -> str:
    """Normalize common MCP tool response formats into readable text."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        texts = [item.get("text", "") for item in result if isinstance(item, dict)]
        if texts:
            return "\n".join(texts)
    return json.dumps(result, ensure_ascii=False, default=str)


def exception_details(error: BaseException) -> str:
    """Flatten nested task-group errors into actionable details."""
    nested = getattr(error, "exceptions", None)
    if nested:
        return "; ".join(dict.fromkeys(exception_details(item) for item in nested))
    return f"{type(error).__name__}: {error}"
