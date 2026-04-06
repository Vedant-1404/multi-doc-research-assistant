"""
test_retrieval.py — Tests for QueryEngine and source extraction logic.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from app.retrieval.engine import QueryEngine, QueryResult, SourceNode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fakes
# ─────────────────────────────────────────────────────────────────────────────

def _make_node(
    text: str,
    source_name: str = "test.pdf",
    source_type: str = "pdf",
    page_label: str = "1",
    score: float = 0.8,
    node_id: str = "node-1",
):
    """Build a fake LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.node_id = node_id
    node.get_content.return_value = text
    node.metadata = {
        "source_name": source_name,
        "source_type": source_type,
        "source_path": f"/{source_name}",
        "page_label": page_label,
    }

    node_with_score = MagicMock()
    node_with_score.node = node
    node_with_score.score = score
    return node_with_score


def _make_response(nodes, answer="Test answer"):
    response = MagicMock()
    response.source_nodes = nodes
    response.__str__ = MagicMock(return_value=answer)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Source extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceExtraction:
    """Test _extract_sources independently via a patched engine."""

    def _make_engine_stub(self):
        """Create a QueryEngine with mocked internals."""
        with patch("app.retrieval.engine.OpenAI"), \
             patch("app.retrieval.engine.VectorIndexRetriever"), \
             patch("app.retrieval.engine.build_postprocessors", return_value=[]), \
             patch("app.retrieval.engine.get_response_synthesizer"), \
             patch("app.retrieval.engine.RetrieverQueryEngine"):
            engine = QueryEngine.__new__(QueryEngine)
            engine.cfg = MagicMock()
            engine.cfg.similarity_threshold = 0.3
            engine.cfg.top_k = 6
            return engine

    def test_extracts_sources_above_threshold(self):
        engine = self._make_engine_stub()
        nodes = [
            _make_node("High relevance text", score=0.85, node_id="n1"),
            _make_node("Medium relevance text", score=0.5, node_id="n2"),
        ]
        response = _make_response(nodes)
        sources = engine._extract_sources(response)
        assert len(sources) == 2

    def test_filters_sources_below_threshold(self):
        engine = self._make_engine_stub()
        nodes = [
            _make_node("Relevant", score=0.8, node_id="n1"),
            _make_node("Not relevant", score=0.1, node_id="n2"),   # below 0.3
        ]
        response = _make_response(nodes)
        sources = engine._extract_sources(response)
        assert len(sources) == 1
        assert sources[0].text == "Relevant"

    def test_deduplicates_by_node_id(self):
        engine = self._make_engine_stub()
        nodes = [
            _make_node("Text A", score=0.9, node_id="same-id"),
            _make_node("Text A duplicate", score=0.85, node_id="same-id"),
        ]
        response = _make_response(nodes)
        sources = engine._extract_sources(response)
        assert len(sources) == 1

    def test_sorts_by_score_descending(self):
        engine = self._make_engine_stub()
        nodes = [
            _make_node("Low", score=0.4, node_id="n1"),
            _make_node("High", score=0.9, node_id="n2"),
            _make_node("Mid", score=0.6, node_id="n3"),
        ]
        response = _make_response(nodes)
        sources = engine._extract_sources(response)
        scores = [s.score for s in sources]
        assert scores == sorted(scores, reverse=True)

    def test_source_metadata_mapped_correctly(self):
        engine = self._make_engine_stub()
        nodes = [
            _make_node(
                "Content",
                source_name="paper.pdf",
                source_type="pdf",
                page_label="42",
                score=0.75,
                node_id="n1",
            )
        ]
        response = _make_response(nodes)
        sources = engine._extract_sources(response)
        src = sources[0]
        assert src.source_name == "paper.pdf"
        assert src.source_type == "pdf"
        assert src.page_label == "42"
        assert src.score == 0.75

    def test_empty_source_nodes(self):
        engine = self._make_engine_stub()
        response = _make_response([])
        sources = engine._extract_sources(response)
        assert sources == []


# ─────────────────────────────────────────────────────────────────────────────
# Confidence detection
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidence:
    def _engine(self):
        with patch("app.retrieval.engine.OpenAI"), \
             patch("app.retrieval.engine.VectorIndexRetriever"), \
             patch("app.retrieval.engine.build_postprocessors", return_value=[]), \
             patch("app.retrieval.engine.get_response_synthesizer"), \
             patch("app.retrieval.engine.RetrieverQueryEngine"):
            engine = QueryEngine.__new__(QueryEngine)
            engine.cfg = MagicMock()
            engine.cfg.similarity_threshold = 0.3
            return engine

    def test_is_confident_high_score(self):
        engine = self._engine()
        sources = [SourceNode(
            text="x", source_name="a", source_type="pdf",
            source_path="a.pdf", page_label="1", score=0.85, node_id="n1"
        )]
        assert engine._is_confident(sources) is True

    def test_not_confident_borderline_score(self):
        engine = self._engine()
        sources = [SourceNode(
            text="x", source_name="a", source_type="pdf",
            source_path="a.pdf", page_label="1", score=0.35, node_id="n1"
        )]
        # 0.35 < 0.3 + 0.1 = 0.4 → not confident
        assert engine._is_confident(sources) is False

    def test_not_confident_empty(self):
        engine = self._engine()
        assert engine._is_confident([]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenEstimation:
    def _engine(self):
        with patch("app.retrieval.engine.OpenAI"), \
             patch("app.retrieval.engine.VectorIndexRetriever"), \
             patch("app.retrieval.engine.build_postprocessors", return_value=[]), \
             patch("app.retrieval.engine.get_response_synthesizer"), \
             patch("app.retrieval.engine.RetrieverQueryEngine"):
            engine = QueryEngine.__new__(QueryEngine)
            engine.cfg = MagicMock()
            return engine

    def test_token_estimate_four_chars_per_token(self):
        engine = self._engine()
        nodes = [_make_node("a" * 400)]  # 400 chars → ~100 tokens
        count = engine._estimate_tokens(nodes)
        assert count == 100

    def test_empty_nodes(self):
        engine = self._engine()
        assert engine._estimate_tokens([]) == 0
