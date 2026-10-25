"""Agent service, tool policy, and guarded MCP tool wrappers."""

from .policy import (
    ALLOWED_TOOLS,
    APPROVAL_REQUIRED_TOOLS,
    MCP_TOOL_ALIASES,
    READ_ONLY_TOOLS,
    canonical_tool_name,
    exception_details,
    find_mcp_tool,
    inject_site_context,
    site_aware_args_schema,
    site_context_value,
    tool_result_text,
)
from .service import AgentError, AgentResult, AtlassianAgent, SYSTEM_PROMPT, build_system_prompt
from .tools import make_draft_tool, make_read_tool

# Private names retained for tests/integrations that imported the original module.
_canonical_tool_name = canonical_tool_name
_exception_details = exception_details
_find_mcp_tool = find_mcp_tool
_inject_site_context = inject_site_context
_site_aware_args_schema = site_aware_args_schema
_site_context_value = site_context_value
_tool_result_text = tool_result_text
