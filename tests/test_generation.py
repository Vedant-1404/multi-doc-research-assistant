"""
test_generation.py — Tests for ResponseSynthesizer formatting and streaming logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.retrieval.engine import QueryResult, SourceNode
from app.generation.synthesizer import ResponseSynthesizer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_source(
    text="Sample content about transformers.",
    source_name="paper.pdf",
    source_type="pdf",
    source_path="paper.pdf",
    page_label="5",
    score=0.82,
    node_id="n1",
) -> SourceNode:
    return SourceNode(
        text=text,
        source_name=source_name,
        source_type=source_type,
        source_path=source_path,
        page_label=page_label,
        score=score,
        node_id=node_id,
    )


def _make_result(sources=None, found=True, query="What is attention?") -> QueryResult:
    return QueryResult(
        answer="Attention allows the model to focus on relevant tokens.",
        sources=sources if sources is not None else [_make_source()],
        query=query,
        retrieval_count=6,
        final_count=len(sources) if sources else 1,
        tokens_in_context=512,
        found=found,
    )


@pytest.fixture
def synthesizer():
    with patch("app.generation.synthesizer.OpenAI"):
        s = ResponseSynthesizer()
        s._client = MagicMock()
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Stream answer
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamAnswer:
    def test_empty_sources_yields_warning(self, synthesizer):
        result = _make_result(sources=[])
        chunks = list(synthesizer.stream_answer(result))
        full = "".join(chunks)
        assert "couldn't find" in full.lower() or "⚠️" in full

    def test_streams_chunks_from_openai(self, synthesizer):
        """Verify chunks are yielded token by token."""
        result = _make_result()

        # Mock OpenAI streaming response
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Attention "

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "is important."

        chunk_end = MagicMock()
        chunk_end.choices = [MagicMock()]
        chunk_end.choices[0].delta.content = None

        synthesizer._client.chat.completions.create.return_value = [
            chunk1, chunk2, chunk_end
        ]

        chunks = list(synthesizer.stream_answer(result))
        full = "".join(chunks)
        assert "Attention" in full
        assert "important" in full

    def test_low_confidence_adds_hedge(self, synthesizer):
        """Low-confidence result should append a warning note."""
        result = _make_result(found=False)

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Some answer."

        synthesizer._client.chat.completions.create.return_value = [chunk]

        chunks = list(synthesizer.stream_answer(result))
        full = "".join(chunks)
        assert "low relevance" in full.lower() or "⚠️" in full

    def test_api_error_yields_error_message(self, synthesizer):
        result = _make_result()
        synthesizer._client.chat.completions.create.side_effect = Exception("API error")

        chunks = list(synthesizer.stream_answer(result))
        full = "".join(chunks)
        assert "❌" in full or "error" in full.lower()

    def test_null_delta_chunks_are_skipped(self, synthesizer):
        """None delta content should not appear in output."""
        result = _make_result()

        chunks_mock = []
        for content in [None, "Real ", None, "answer.", None]:
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = content
            chunks_mock.append(c)

        synthesizer._client.chat.completions.create.return_value = chunks_mock

        chunks = list(synthesizer.stream_answer(result))
        full = "".join(chunks)
        assert full == "Real answer."


# ─────────────────────────────────────────────────────────────────────────────
# Context building
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBuilding:
    def test_context_includes_source_name(self, synthesizer):
        sources = [_make_source(source_name="my_paper.pdf", text="Important finding.")]
        context = synthesizer._build_context(sources)
        assert "my_paper.pdf" in context

    def test_context_includes_page_label_for_pdfs(self, synthesizer):
        sources = [_make_source(source_type="pdf", page_label="23")]
        context = synthesizer._build_context(sources)
        assert "p.23" in context

    def test_context_numbered_correctly(self, synthesizer):
        sources = [
            _make_source(text="First source.", node_id="n1"),
            _make_source(text="Second source.", node_id="n2"),
        ]
        context = synthesizer._build_context(sources)
        assert "[1]" in context
        assert "[2]" in context

    def test_context_includes_score(self, synthesizer):
        sources = [_make_source(score=0.923)]
        context = synthesizer._build_context(sources)
        assert "0.923" in context

    def test_url_source_uses_section_label(self, synthesizer):
        sources = [_make_source(
            source_type="url",
            source_name="arxiv.org",
            page_label="Introduction",
        )]
        context = synthesizer._build_context(sources)
        # URL sources should NOT use "p." prefix
        assert "Introduction" in context
        assert "p.Introduction" not in context


# ─────────────────────────────────────────────────────────────────────────────
# Source markdown formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSourcesMarkdown:
    def test_returns_empty_string_for_no_sources(self, synthesizer):
        assert synthesizer.format_sources_markdown([]) == ""

    def test_renders_pdf_icon(self, synthesizer):
        sources = [_make_source(source_type="pdf")]
        md = synthesizer.format_sources_markdown(sources)
        assert "📄" in md

    def test_renders_url_icon(self, synthesizer):
        sources = [_make_source(source_type="url")]
        md = synthesizer.format_sources_markdown(sources)
        assert "🌐" in md

    def test_truncates_long_excerpts(self, synthesizer):
        long_text = "word " * 200
        sources = [_make_source(text=long_text)]
        md = synthesizer.format_sources_markdown(sources)
        assert "..." in md

    def test_short_text_not_truncated(self, synthesizer):
        short_text = "Brief excerpt."
        sources = [_make_source(text=short_text)]
        md = synthesizer.format_sources_markdown(sources)
        assert "..." not in md

    def test_score_bar_length(self, synthesizer):
        """Score bar should always be exactly 10 chars."""
        for score in [0.0, 0.3, 0.5, 0.8, 1.0]:
            bar = synthesizer._score_bar(score)
            assert len(bar) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Confidence hedge
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceHedge:
    def test_no_hedge_when_confident(self, synthesizer):
        result = _make_result(found=True)
        hedge = synthesizer._confidence_hedge(result)
        assert hedge == ""

    def test_hedge_when_not_confident(self, synthesizer):
        result = _make_result(found=False)
        hedge = synthesizer._confidence_hedge(result)
        assert len(hedge) > 0
        assert "low relevance" in hedge.lower() or "incomplete" in hedge.lower()
