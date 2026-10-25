from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from atlassian_chatbot.call_logging import (
    accepted_tool_parameters,
    log_tool_call,
    reset_current_prompt,
    reset_current_system_prompt,
    set_current_prompt,
    set_current_system_prompt,
)
from atlassian_chatbot.config import AppConfig
from atlassian_chatbot.mcp.connection import create_mcp_client
from .policy import (
    APPROVAL_REQUIRED_TOOLS,
    READ_ONLY_TOOLS,
    canonical_tool_name,
    exception_details,
    find_mcp_tool,
    inject_site_context,
    tool_result_text,
)
from .tools import make_draft_tool, make_read_tool
from atlassian_chatbot.prompts import INDEXED_READ_SYSTEM_PROMPT, SYSTEM_PROMPT, build_system_prompt
from atlassian_chatbot.rag import AtlassianFaissStore, get_embedding_model, make_retrieval_tool

_WRITE_REQUEST_PATTERN = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|i want you to\s+|help me\s+)?"
    r"(?:to\s+)?(?:create|update|edit|change|modify|assign|move|transition|close|reopen)\b",
    re.IGNORECASE,
)


class AgentError(RuntimeError):
    """Raised for LangChain, MCP, or tool workflow errors."""


@dataclass
class AgentResult:
    text: str
    pending_actions: list[dict[str, Any]] = field(default_factory=list)


class AtlassianAgent:
    """LangChain agent backed by Groq and allowlisted Atlassian MCP tools."""

    def __init__(self, config: AppConfig):
        self.config = config
        from langchain_groq import ChatGroq

        self.llm = ChatGroq(api_key=config.groq_api_key, model=config.groq_model)

    def answer(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        *,
        document_store: AtlassianFaissStore | None = None,
    ) -> AgentResult:
        return asyncio.run(self._answer(prompt, history or [], document_store=document_store))

    async def _answer(
        self,
        prompt: str,
        history: list[dict[str, str]],
        *,
        document_store: AtlassianFaissStore | None = None,
    ) -> AgentResult:
        pending_actions: list[dict[str, Any]] = []
        if document_store is None:
            document_store = AtlassianFaissStore(embedding_model=get_embedding_model())
        has_indexed_data = document_store.chunk_count > 0
        write_request = bool(_WRITE_REQUEST_PATTERN.search(prompt.strip()))
        tools: list[BaseTool] = []
        system_prompt = build_system_prompt(self.config)

        # A populated session index is the source of truth for read-only prompts.
        # Write requests always discover MCP tools so approval-gated tools are available.
        if has_indexed_data and not write_request:
            tools.append(make_retrieval_tool(document_store))
            system_prompt = f"{INDEXED_READ_SYSTEM_PROMPT}\n{system_prompt}"
        else:
            try:
                client = create_mcp_client(self.config)
                mcp_tools = await client.get_tools(server_name="atlassian")
            except Exception as exc:
                raise AgentError(f"Could not load Atlassian MCP tools: {exception_details(exc)}") from exc

            discovered_names = [tool.name for tool in mcp_tools]
            for mcp_tool in mcp_tools:
                name = canonical_tool_name(mcp_tool.name)
                if name is None:
                    continue
                if name in APPROVAL_REQUIRED_TOOLS:
                    tools.append(
                        make_draft_tool(
                            mcp_tool,
                            pending_actions,
                            public_name=name,
                            config=self.config,
                        )
                    )
                elif name in READ_ONLY_TOOLS:
                    tools.append(
                        make_read_tool(
                            mcp_tool,
                            name,
                            self.config,
                            document_store=document_store,
                        )
                    )

            if not tools:
                raise AgentError(
                    "The MCP server connected, but none of its tools match the supported Jira/Confluence "
                    f"tools. Discovered: {', '.join(discovered_names) or '(none)'}."
                )
            tools.append(make_retrieval_tool(document_store))
        agent = create_agent(self.llm, tools, system_prompt=system_prompt)
        messages: list[BaseMessage] = []
        for entry in history[-12:]:
            if entry.get("role") == "user":
                messages.append(HumanMessage(content=entry.get("content", "")))
            elif entry.get("role") == "assistant":
                messages.append(AIMessage(content=entry.get("content", "")))
        messages.append(HumanMessage(content=prompt))

        prompt_token = set_current_prompt(prompt)
        system_prompt_token = set_current_system_prompt(system_prompt)
        try:
            result = await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": max(12, self.config.max_tool_rounds * 3)},
            )
        except Exception as exc:
            detail = str(exc)
            if "invalid_api_key" in detail.lower() or "invalid api key" in detail.lower():
                detail = "Groq rejected GROQ_API_KEY. Put a current, valid Groq API key in .env, then restart Streamlit."
            raise AgentError(f"LangChain agent run failed: {detail}") from exc
        finally:
            reset_current_prompt(prompt_token)
            reset_current_system_prompt(system_prompt_token)

        final_text = "I could not produce a response."
        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                final_text = tool_result_text(message.content)
                break
        return AgentResult(text=final_text, pending_actions=pending_actions)

    def execute_approved_action(
        self, name: str, arguments: Mapping[str, Any], *, prompt: str | None = None
    ) -> str:
        if name not in APPROVAL_REQUIRED_TOOLS:
            raise AgentError(f"Tool '{name}' is not an approval-gated Jira write action.")
        return asyncio.run(self._execute_tool(name, dict(arguments), prompt=prompt))

    @staticmethod
    def _alias_tool(
        mcp_tool: BaseTool, public_name: str, config: AppConfig | None = None
    ) -> BaseTool:
        """Retain the former helper API while delegating to the tool factory."""
        return make_read_tool(mcp_tool, public_name, config)

    @staticmethod
    def _draft_tool(
        mcp_tool: BaseTool,
        pending_actions: list[dict[str, Any]],
        *,
        public_name: str | None = None,
        config: AppConfig | None = None,
    ) -> BaseTool:
        """Retain the former helper API while delegating to the draft tool factory."""
        return make_draft_tool(
            mcp_tool, pending_actions, public_name=public_name, config=config
        )

    async def _execute_tool(
        self, name: str, arguments: dict[str, Any], *, prompt: str | None = None
    ) -> str:
        try:
            client = create_mcp_client(self.config)
            tools = await client.get_tools(server_name="atlassian")
            tool = find_mcp_tool(tools, name)
            if tool is None:
                raise AgentError(f"The connected MCP server does not expose '{name}'.")
            final_arguments = inject_site_context(arguments, tool, self.config)
            log_tool_call(
                name,
                final_arguments,
                phase="approved invoke",
                prompt=prompt,
                system_prompt=build_system_prompt(self.config),
                accepted_parameters=accepted_tool_parameters(tool),
            )
            result = await tool.ainvoke(final_arguments)
            return tool_result_text(result)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Could not execute the approved Jira action: {exc}") from exc
