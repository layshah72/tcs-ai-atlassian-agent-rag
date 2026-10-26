from __future__ import annotations

from langchain_core.tools import StructuredTool

from atlassian_chatbot.call_logging import log_tool_call
from .vector_store import AtlassianFaissStore


RETRIEVAL_TOOL_NAME = "atlassian_retrieve"


def make_retrieval_tool(store: AtlassianFaissStore) -> StructuredTool:
    """Expose indexed Jira/Confluence snippets to the agent within a size limit."""
    def retrieve(query: str) -> str:
        log_tool_call(
            RETRIEVAL_TOOL_NAME,
            {"query": query},
            phase="RAG retrieval",
            accepted_parameters=["query"],
        )
        chunks = store.search(query, top_k=5, max_chars=7000)
        if not chunks:
            return "No indexed Jira or Confluence results matched this query. Search the relevant source first."
        sections = [
            f"[{index}] source={chunk.source}, similarity={chunk.score:.3f}\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        ]
        return "Retrieved context (cite sources by their bracketed number):\n\n" + "\n\n".join(sections)

    tool = StructuredTool.from_function(
        func=retrieve,
        name=RETRIEVAL_TOOL_NAME,
        description=(
            "Search the current turn's FAISS index for relevant Jira and Confluence content. "
            "Use this after a Jira or Confluence read/search tool reports that its results were indexed, "
            "and before composing the final answer. Pass the user's question as query."
        ),
    )
    return tool

