"""
engine.py — LlamaIndex query engine with retrieval quality instrumentation.

Returns both the answer AND structured source nodes so the UI can render
per-chunk citations with scores, source names, and page numbers.
"""

from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Settings as LISettings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq
from loguru import logger

from app.config import get_settings
from app.retrieval.reranker import build_postprocessors


@dataclass
class SourceNode:
    """A single retrieved chunk with quality metadata."""
    text: str
    source_name: str
    source_type: str        # "pdf" | "url"
    source_path: str
    page_label: str
    score: float
    node_id: str


@dataclass
class QueryResult:
    """Full result returned to the UI layer."""
    answer: str
    sources: list[SourceNode] = field(default_factory=list)
    query: str = ""
    retrieval_count: int = 0        # nodes retrieved before reranking
    final_count: int = 0            # nodes after reranking / filtering
    tokens_in_context: int = 0
    found: bool = True              # False if low-confidence answer


class QueryEngine:
    """
    Wraps LlamaIndex RetrieverQueryEngine with:
    - Configurable top-k retrieval
    - Optional Cohere reranking
    - Similarity score filtering
    - Structured source attribution
    """

    def __init__(self, index: VectorStoreIndex):
        self.cfg = get_settings()
        self._index = index
        self._engine = self._build_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> QueryResult:
        """Run a query and return answer + structured sources."""
        logger.info(f"Query: {question!r}")
        response = self._engine.query(question)

        source_nodes = self._extract_sources(response)
        tokens = self._estimate_tokens(response.source_nodes)
        found = self._is_confident(source_nodes)

        logger.info(
            f"  Retrieved {len(response.source_nodes)} → "
            f"{len(source_nodes)} sources used | tokens≈{tokens}"
        )

        return QueryResult(
            answer=str(response),
            sources=source_nodes,
            query=question,
            retrieval_count=len(response.source_nodes),
            final_count=len(source_nodes),
            tokens_in_context=tokens,
            found=found,
        )

    def rebuild(self, index: VectorStoreIndex):
        """Hot-swap the index (called after new docs are ingested)."""
        self._index = index
        self._engine = self._build_engine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_engine(self) -> RetrieverQueryEngine:
        llm = Groq(
    model=self.cfg.groq_chat_model,
    api_key=self.cfg.groq_api_key,
    )
        LISettings.llm = llm

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self.cfg.top_k,
        )

        postprocessors = build_postprocessors()

        synthesizer = get_response_synthesizer(
            response_mode="compact",        # merges chunks before sending to LLM
            llm=llm,
        )

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
        )

    def _extract_sources(self, response) -> list[SourceNode]:
        """Convert LlamaIndex NodeWithScore objects to our SourceNode dataclass."""
        seen_ids: set[str] = set()
        sources: list[SourceNode] = []

        for node_with_score in response.source_nodes:
            node = node_with_score.node
            score = node_with_score.score or 0.0

            # Filter low-confidence chunks
            if score < self.cfg.similarity_threshold:
                continue

            # Deduplicate by node ID
            if node.node_id in seen_ids:
                continue
            seen_ids.add(node.node_id)

            meta = node.metadata or {}
            sources.append(
                SourceNode(
                    text=node.get_content(),
                    source_name=meta.get("source_name", "Unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    source_path=meta.get("source_path", ""),
                    page_label=meta.get("page_label", "—"),
                    score=round(score, 4),
                    node_id=node.node_id,
                )
            )

        # Sort by score descending
        sources.sort(key=lambda s: s.score, reverse=True)
        return sources

    def _estimate_tokens(self, nodes) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(n.node.get_content()) for n in nodes)
        return total_chars // 4

    def _is_confident(self, sources: list[SourceNode]) -> bool:
        """Return False if the top source score is borderline."""
        if not sources:
            return False
        return sources[0].score >= self.cfg.similarity_threshold + 0.1
