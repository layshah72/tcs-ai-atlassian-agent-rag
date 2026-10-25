from __future__ import annotations

import asyncio
import hashlib

import numpy as np

from langchain_core.tools import StructuredTool

from atlassian_chatbot.agent.tools import make_read_tool
from atlassian_chatbot.rag import AtlassianFaissStore, make_retrieval_tool


class FakeEmbeddingModel:
    """Deterministic test double for FastEmbed's document/query methods."""

    dimensions = 384

    def embed(self, documents):
        return iter(self._vector(text) for text in documents)

    def query_embed(self, query):
        return iter([self._vector(query)])

    def _vector(self, text):
        vector = np.zeros(self.dimensions, dtype="float32")
        for token in text.lower().split():
            bucket = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % self.dimensions
            vector[bucket] += 1
        return vector


class SemanticEmbeddingModel(FakeEmbeddingModel):
    """Tiny meaning-level fixture demonstrating neural semantic retrieval."""

    SYNONYMS = {
        "duplicate": "charged",
        "payment": "purchase",
        "purchase": "purchase",
        "charged": "charged",
        "bill": "charged",
        "reimbursement": "refund",
    }

    def _vector(self, text):
        normalized = " ".join(self.SYNONYMS.get(token.lower(), token.lower()) for token in text.split())
        return super()._vector(normalized)


def make_store(**kwargs):
    return AtlassianFaissStore(embedding_model=FakeEmbeddingModel(), **kwargs)


def test_faiss_store_chunks_and_retrieves_relevant_live_mcp_results():
    store = AtlassianFaissStore(
        chunk_words=12, overlap_words=2, embedding_model=SemanticEmbeddingModel()
    )
    jira_result = {
        "issues": [
            {
                "key": "PAY-42",
                "summary": "Payment service returns timeout for checkout",
                "status": "Open",
                "assignee": "Asha",
            },
            {
                "key": "DOC-9",
                "summary": "Update internal deployment documentation",
                "status": "Done",
            },
        ]
    }

    added = store.add_result("jira_search", jira_result)
    matches = store.search("checkout payment service timeout", top_k=2)

    assert added >= 1
    assert matches
    assert matches[0].source == "jira_search"
    assert "PAY-42" in matches[0].text
    assert "checkout" in matches[0].text


def test_read_tool_indexes_result_and_returns_only_a_small_receipt():
    store = make_store()
    large_payload = "JIRA-123 payment checkout timeout " * 1500

    async def fetch(jql: str) -> str:
        return large_payload

    source_tool = StructuredTool.from_function(
        coroutine=fetch,
        name="jira_search",
        description="Search Jira issues",
    )
    wrapped = make_read_tool(source_tool, "jira_search", document_store=store)

    receipt = asyncio.run(wrapped.ainvoke({"jql": "project = JIRA"}))
    retrieved = store.search("payment checkout timeout", top_k=1, max_chars=500)

    assert "indexed them into the current-turn FAISS store" in receipt
    assert len(receipt) < 500
    assert large_payload not in receipt
    assert store.chunk_count > 1
    assert retrieved
    assert "JIRA-123" in retrieved[0].text


def test_retrieval_tool_returns_bounded_cited_context():
    store = make_store(chunk_words=40, overlap_words=5)
    store.add_result(
        "confluence_search",
        "Payments handbook: checkout retries use exponential backoff. " * 30,
    )
    tool = make_retrieval_tool(store)

    result = tool.invoke({"query": "checkout retry backoff"})

    assert "Retrieved context" in result
    assert "[1] source=confluence_search" in result
    assert "exponential backoff" in result
    assert len(result) <= 7300


def test_fastembed_embeddings_support_semantically_related_phrasing():
    store = AtlassianFaissStore(
        embedding_model=SemanticEmbeddingModel(),
        chunk_words=40,
        overlap_words=5,
    )
    store.add_result(
        "jira_search",
        "PAY-77 The customer was charged twice for a purchase after checkout.",
    )

    results = store.search("duplicate payment for one purchase", top_k=1)

    assert results
    assert results[0].source == "jira_search"
    assert "PAY-77" in results[0].text


def test_real_fastembed_neural_model_generates_embeddings():
    from atlassian_chatbot.rag import get_embedding_model

    model = get_embedding_model()
    document_vector = next(model.embed(["A customer was charged twice for a purchase."]))
    query_vector = next(model.query_embed("duplicate payment for one purchase"))

    assert len(document_vector) == 384
    assert len(query_vector) == 384
