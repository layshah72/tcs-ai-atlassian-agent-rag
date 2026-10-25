from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.call_logging import accepted_tool_parameters, get_current_prompt, log_tool_call
from .policy import inject_site_context, site_aware_args_schema
from atlassian_chatbot.rag.vector_store import AtlassianFaissStore
from atlassian_chatbot.mcp.results import MCPToolResultError, ensure_successful_tool_result

logger = logging.getLogger(__name__)


def make_read_tool(
    mcp_tool: BaseTool,
    public_name: str,
    config: AppConfig | None = None,
    *,
    document_store: AtlassianFaissStore | None = None,
) -> BaseTool:
    """Expose an app-facing alias and inject configured site context on invocation."""
    async def invoke_remote_tool(**arguments: Any) -> Any:
        if config is not None:
            arguments = inject_site_context(arguments, mcp_tool, config)
        log_tool_call(
            public_name,
            arguments,
            accepted_parameters=accepted_tool_parameters(mcp_tool),
        )
        result = await mcp_tool.ainvoke(arguments)
        if document_store is not None:
            try:
                ensure_successful_tool_result(result, tool_name=public_name)
                chunk_count = document_store.add_result(public_name, result)
            except MCPToolResultError as exc:
                logger.warning("Not indexing MCP error result from %s: %s", public_name, exc)
                return f"{public_name} failed: {exc}. No result was added to the vector index."
            except Exception:
                logger.exception("Failed to index results returned by MCP tool %s", public_name)
                return (
                    f"The {public_name} call returned data, but indexing failed. "
                    "The raw result was omitted to avoid exceeding the model context window."
                )
            return (
                f"Fetched {public_name} results and indexed them into the current-turn FAISS store "
                f"({chunk_count} chunks; {document_store.chunk_count} total). "
                "Use atlassian_retrieve with the user's question to retrieve relevant context before answering."
            )
        return result

    return StructuredTool.from_function(
        coroutine=invoke_remote_tool,
        name=public_name,
        description=mcp_tool.description or f"Atlassian MCP tool: {public_name}",
        args_schema=site_aware_args_schema(mcp_tool, config) if config is not None else mcp_tool.args_schema,
    )


def make_draft_tool(
    mcp_tool: BaseTool,
    pending_actions: list[dict[str, Any]],
    *,
    public_name: str | None = None,
    config: AppConfig | None = None,
) -> BaseTool:
    """Create a tool wrapper that records write arguments but never executes them."""
    name = public_name or mcp_tool.name

    def stage(**arguments: Any) -> str:
        if config is not None:
            arguments = inject_site_context(arguments, mcp_tool, config)
        prompt = get_current_prompt()
        log_tool_call(
            name,
            arguments,
            phase="draft staged",
            prompt=prompt,
            accepted_parameters=accepted_tool_parameters(mcp_tool),
        )
        action = {"name": name, "arguments": arguments}
        if prompt is not None:
            action["prompt"] = prompt
        pending_actions.append(action)
        return (
            "Draft staged but NOT executed. Tell the user to review the displayed arguments "
            "and approve or reject in the UI."
        )

    description = mcp_tool.description or f"Draft a {name} request; it will not execute until approved."
    schema = site_aware_args_schema(mcp_tool, config) if config is not None else mcp_tool.args_schema
    return StructuredTool.from_function(
        func=stage,
        name=name,
        description=description + " This tool only stages a draft and does not execute the Jira change.",
        args_schema=schema,
    )
