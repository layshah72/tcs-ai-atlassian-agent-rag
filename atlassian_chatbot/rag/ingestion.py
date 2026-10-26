from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool

from atlassian_chatbot.call_logging import accepted_tool_parameters, log_tool_call
from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.mcp.connection import create_mcp_client
from atlassian_chatbot.mcp.results import MCPToolResultError, ensure_successful_tool_result
from .embedder import get_embedding_model, DEFAULT_EMBED_MODEL
from .store import AtlassianFaissStore


class IngestionError(RuntimeError):
    """Raised when Jira/Confluence data cannot be refreshed into the vector index."""


@dataclass(frozen=True)
class IngestionResult:
    store: AtlassianFaissStore
    jira_chunks: int
    confluence_chunks: int


class AtlassianIngestionPipeline:
    """Fetch configured Jira and Confluence searches into a fresh FAISS index."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client_factory: Callable[[AppConfig], Any] = create_mcp_client,
        store_factory: Callable[[], AtlassianFaissStore] | None = None,
    ) -> None:
        self.config = config
        self.client_factory = client_factory
        self.store_factory = store_factory or (
            lambda: AtlassianFaissStore(
                model_name=os.getenv("FASTEMBED_MODEL") or DEFAULT_EMBED_MODEL,
                embedding_model=get_embedding_model(),
            )
        )

    def refresh(self, jira_jql: str, confluence_query: str) -> IngestionResult:
        """Synchronously refresh both sources; the caller's old store stays intact on failure."""
        return asyncio.run(self._refresh(jira_jql.strip(), confluence_query.strip()))

    async def _refresh(self, jira_jql: str, confluence_query: str) -> IngestionResult:
        from atlassian_chatbot.agent.policy import inject_site_context

        if not jira_jql:
            raise IngestionError("Enter a Jira JQL query before refreshing the index.")
        if not confluence_query:
            raise IngestionError("Enter a Confluence query before refreshing the index.")

        try:
            client = self.client_factory(self.config)
            tools = await client.get_tools(server_name="atlassian")
        except Exception as exc:
            raise IngestionError(f"Could not load Atlassian MCP tools: {exc}") from exc

        jira_tool = _find_search_tool(tools, "jira")
        confluence_tool = _find_search_tool(tools, "confluence")
        if jira_tool is None:
            raise IngestionError("The MCP server does not expose the supported Jira search tool.")
        if confluence_tool is None:
            raise IngestionError("The MCP server does not expose the supported Confluence search tool.")

        try:
            replacement_store = self.store_factory()
            jira_arguments = _build_search_arguments(jira_tool, jira_jql, source="jira")
            jira_arguments = inject_site_context(jira_arguments, jira_tool, self.config)
            _validate_required_arguments(jira_tool, jira_arguments)
            log_tool_call(
                "jira_search",
                jira_arguments,
                phase="scheduled ingestion",
                prompt=f"Refresh Jira index using JQL: {jira_jql}",
                accepted_parameters=accepted_tool_parameters(jira_tool),
            )
            jira_result = await jira_tool.ainvoke(jira_arguments)
            _ensure_ingestion_result("Jira search", jira_result)
            jira_chunks = replacement_store.add_result("jira_search", jira_result)

            confluence_arguments = _build_search_arguments(
                confluence_tool, confluence_query, source="confluence"
            )
            confluence_arguments = inject_site_context(
                confluence_arguments, confluence_tool, self.config
            )
            _validate_required_arguments(confluence_tool, confluence_arguments)
            log_tool_call(
                "confluence_search",
                confluence_arguments,
                phase="scheduled ingestion",
                prompt=f"Refresh Confluence index using query: {confluence_query}",
                accepted_parameters=accepted_tool_parameters(confluence_tool),
            )
            confluence_result = await confluence_tool.ainvoke(confluence_arguments)
            _ensure_ingestion_result("Confluence search", confluence_result)
            confluence_chunks = replacement_store.add_result(
                "confluence_search", confluence_result
            )
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                "Index refresh failed before completion; the existing session index was not replaced. "
                f"Details: {exc}"
            ) from exc

        if jira_chunks + confluence_chunks == 0:
            raise IngestionError(
                "The Jira and Confluence searches returned no indexable content; the existing index was kept."
            )
        return IngestionResult(replacement_store, jira_chunks, confluence_chunks)


def _find_search_tool(tools: list[BaseTool], source: str) -> BaseTool | None:
    name_lower = source.lower()
    return next(
        (t for t in tools if name_lower in t.name.lower() and "search" in t.name.lower()),
        None,
    )


def _build_search_arguments(tool: BaseTool, query: str, *, source: str) -> dict[str, Any]:
    """Map a Jira JQL or Confluence query to the server's discovered argument schema."""
    from atlassian_chatbot.agent.policy import tool_schema_fields

    schema = tool.args_schema
    fields = tool_schema_fields(tool)

    normalized_names = {name.replace("_", "").lower(): name for name in fields}
    candidates = (
        ("jql", "query", "searchquery", "search")
        if source == "jira"
        else ("cql", "query", "searchquery", "searchterm", "search")
    )
    query_field = next(
        (normalized_names[name] for name in candidates if name in normalized_names),
        None,
    )
    if query_field is None:
        raise IngestionError(
            f"The {tool.name} input schema has no supported query field. "
            f"Accepted parameters: {', '.join(fields) or '(none)'}."
        )

    arguments: dict[str, Any] = {query_field: query}
    for normalized, field_name in normalized_names.items():
        if field_name == query_field:
            continue
        if normalized in {"maxresults", "maxresult", "limit", "pagesize", "count"}:
            arguments[field_name] = 100
        elif normalized in {"startat", "offset", "start", "skip"}:
            arguments[field_name] = 0
        elif normalized == "page":
            arguments[field_name] = 1

    model_fields = getattr(schema, "model_fields", {}) if schema is not None else {}
    for field_name, model_field in model_fields.items():
        if field_name in arguments or field_name in {query_field}:
            continue
        if not getattr(model_field, "is_required", lambda: False)():
            continue
        if field_name.replace("_", "").lower() in {
            "cloudid", "atlassiancloudid", "siteurl", "atlassiansiteurl", "baseurl",
            "maxresults", "maxresult", "limit", "pagesize", "count", "startat",
            "offset", "start", "skip", "page",
        }:
            continue
        raise IngestionError(
            f"The {tool.name} tool requires unsupported parameter '{field_name}'."
        )
    return arguments


def _validate_required_arguments(tool: BaseTool, arguments: dict[str, Any]) -> None:
    """Fail before sending a call if discovered schema still has missing required fields."""
    from atlassian_chatbot.agent.policy import tool_schema_fields

    schema = tool.args_schema
    fields = tool_schema_fields(tool)
    required_names = set(getattr(schema, "model_fields", {}))
    if isinstance(schema, dict):
        required_names = set(schema.get("required", []))
    else:
        schema_method = getattr(schema, "model_json_schema", None)
        if callable(schema_method):
            required_names = set(schema_method().get("required", []))

    missing = []
    for name, definition in fields.items():
        is_required = name in required_names
        if hasattr(definition, "is_required"):
            is_required = definition.is_required()
        elif isinstance(definition, dict):
            is_required = is_required or definition.get("required") is True
        if is_required and name not in arguments:
            missing.append(name)
    if missing:
        raise IngestionError(
            f"The {tool.name} MCP call is missing required parameters: {', '.join(missing)}. "
            "Check that the configured ATLASSIAN_CLOUD_ID and search inputs match the tool schema."
        )


def _ensure_ingestion_result(operation: str, result: Any) -> None:
    """Translate MCP error payloads to ingestion errors before indexing them."""
    try:
        ensure_successful_tool_result(result, tool_name=operation)
    except MCPToolResultError as exc:
        raise IngestionError(str(exc)) from exc
