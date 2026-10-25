from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping

from langchain_core.tools import BaseTool
from pydantic import Field, create_model

from atlassian_chatbot.config import AppConfig

_WRITE_TOOL_PATTERN = re.compile(
    r"create|update|edit|delete|add|remove|transition|assign|move|close|reopen",
    re.IGNORECASE,
)


def is_write_tool(tool_name: str) -> bool:
    """Return True if the tool name suggests a write/mutating operation."""
    return bool(_WRITE_TOOL_PATTERN.search(tool_name))


def find_mcp_tool(tools: list[BaseTool], name: str) -> BaseTool | None:
    """Find a tool by exact name."""
    return next((tool for tool in tools if tool.name == name), None)


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
