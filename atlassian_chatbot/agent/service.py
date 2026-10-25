from __future__ import annotations

import asyncio
import logging
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
    exception_details,
    find_mcp_tool,
    inject_site_context,
    is_write_tool,
    tool_result_text,
)
from .tools import make_draft_tool, make_read_tool
from atlassian_chatbot.prompts import INDEXED_READ_SYSTEM_PROMPT, build_system_prompt
from atlassian_chatbot.rag import AtlassianFaissStore, get_embedding_model, make_retrieval_tool

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised for LangChain, MCP, or tool workflow errors."""


@dataclass
class AgentResult:
    text: str
    pending_actions: list[dict[str, Any]] = field(default_factory=list)


class AtlassianAgent:
    """LangChain agent backed by Amazon Bedrock and all available Atlassian MCP tools."""

    def __init__(self, config: AppConfig):
        self.config = config
        import boto3
        from langchain_aws import ChatBedrock
        bedrock_client = boto3.client("bedrock-runtime", region_name=config.bedrock_region)
        self.llm = ChatBedrock(model_id=config.bedrock_model_id, client=bedrock_client)

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
        system_prompt = build_system_prompt(self.config)

        try:
            client = create_mcp_client(self.config)
            mcp_tools = await client.get_tools(server_name="atlassian")
        except Exception as exc:
            raise AgentError(f"Could not load Atlassian MCP tools: {exception_details(exc)}") from exc

        logger.debug("MCP tools discovered: %s", [t.name for t in mcp_tools])

        tools: list[BaseTool] = []
        for mcp_tool in mcp_tools:
            if is_write_tool(mcp_tool.name):
                tools.append(
                    make_draft_tool(
                        mcp_tool,
                        pending_actions,
                        public_name=mcp_tool.name,
                        config=self.config,
                    )
                )
            else:
                tools.append(
                    make_read_tool(
                        mcp_tool,
                        mcp_tool.name,
                        self.config,
                        document_store=document_store,
                    )
                )

        if not tools:
            raise AgentError("The MCP server returned no tools.")

        tools.append(make_retrieval_tool(document_store))
        if has_indexed_data:
            system_prompt = f"{INDEXED_READ_SYSTEM_PROMPT}\n{system_prompt}"

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
            if "credentials" in detail.lower() or "access denied" in detail.lower():
                detail = "AWS Bedrock authentication failed. Ensure AWS credentials are configured and have bedrock:InvokeModel permission."
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
        return asyncio.run(self._execute_tool(name, dict(arguments), prompt=prompt))

    @staticmethod
    def _alias_tool(
        mcp_tool: BaseTool, public_name: str, config: AppConfig | None = None
    ) -> BaseTool:
        return make_read_tool(mcp_tool, public_name, config)

    @staticmethod
    def _draft_tool(
        mcp_tool: BaseTool,
        pending_actions: list[dict[str, Any]],
        *,
        public_name: str | None = None,
        config: AppConfig | None = None,
    ) -> BaseTool:
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
            raise AgentError(f"Could not execute the approved action: {exc}") from exc
