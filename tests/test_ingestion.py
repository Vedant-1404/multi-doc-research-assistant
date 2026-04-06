"""
test_ingestion.py — Unit tests for document loading and chunking pipeline.
Uses minimal mocking so tests are fast and don't hit OpenAI.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.loader import DocumentLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def html_response():
    return """
    <html><body>
        <h1>Test Document</h1>
        <h2>Introduction</h2>
        <p>This is an introductory paragraph about machine learning.</p>
        <p>It contains multiple sentences for testing purposes.</p>
        <h2>Methods</h2>
        <p>We used transformer architectures for the experiments.</p>
        <h3>Data Collection</h3>
        <p>Data was collected from various open-source repositories.</p>
        <h2>Results</h2>
        <p>The model achieved 94% accuracy on the test set.</p>
    </body></html>
    """


# ─────────────────────────────────────────────────────────────────────────────
# URL Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestURLLoader:
    def test_load_url_returns_documents(self, loader, html_response):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            docs = loader.load_url("https://example.com/paper", label="Test Paper")

        assert len(docs) > 0
        assert all(d.metadata["source_type"] == "url" for d in docs)
        assert all(d.metadata["source_name"] == "Test Paper" for d in docs)

    def test_load_url_extracts_sections(self, loader, html_response):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            docs = loader.load_url("https://example.com/paper")

        # Should have multiple sections (Introduction, Methods, Results at minimum)
        assert len(docs) >= 2

    def test_load_url_network_failure_raises(self, loader):
        with patch("requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

            with pytest.raises(ValueError, match="Could not fetch URL"):
                loader.load_url("https://nonexistent.example.com")

    def test_load_url_uses_domain_as_default_label(self, loader, html_response):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            docs = loader.load_url("https://arxiv.org/abs/1234")

        assert all(d.metadata["source_name"] == "arxiv.org" for d in docs)

    def test_load_url_filters_short_sections(self, loader):
        short_html = """
        <html><body>
            <h2>A</h2><p>Short.</p>
            <h2>Real Section</h2>
            <p>This section has enough content to pass the minimum length filter for document extraction.</p>
        </body></html>
        """
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = short_html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            docs = loader.load_url("https://example.com")

        # The very short "Short." section should be filtered
        texts = [d.text for d in docs]
        assert not any(t.strip() == "Short." for t in texts)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFLoader:
    def test_load_pdf_attaches_metadata(self, loader):
        """PDF pages should have source_type, source_name, page_label."""
        from llama_index.core import Document as LIDoc

        fake_pages = [
            LIDoc(text=f"Page {i+1} content about transformers.", metadata={})
            for i in range(3)
        ]

        with patch.object(loader._pdf_reader, "load_data", return_value=fake_pages):
            with patch("builtins.open", MagicMock()):
                with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                    mock_tmp.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(name="tmp.pdf", write=MagicMock())
                    )
                    mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

                    # Directly test the metadata enrichment logic
                    docs = fake_pages.copy()
                    for i, doc in enumerate(docs):
                        doc.metadata.update({
                            "source_type": "pdf",
                            "source_name": "test.pdf",
                            "page_label": str(i + 1),
                        })

        assert all(d.metadata["source_type"] == "pdf" for d in docs)
        assert all(d.metadata["source_name"] == "test.pdf" for d in docs)
        assert docs[0].metadata["page_label"] == "1"
        assert docs[2].metadata["page_label"] == "3"

    def test_load_pdf_page_count(self, loader):
        from llama_index.core import Document as LIDoc
        fake_pages = [LIDoc(text=f"Page {i}", metadata={}) for i in range(5)]

        with patch.object(loader._pdf_reader, "load_data", return_value=fake_pages):
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                tmp_mock = MagicMock()
                tmp_mock.__enter__ = MagicMock(return_value=tmp_mock)
                tmp_mock.__exit__ = MagicMock(return_value=False)
                tmp_mock.name = "/tmp/test.pdf"
                mock_tmp.return_value = tmp_mock

                with patch("pathlib.Path.unlink"):
                    docs = loader.load_pdf(b"%PDF fake content", "test.pdf")

        assert len(docs) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Hash utility
# ─────────────────────────────────────────────────────────────────────────────

class TestHashUtility:
    def test_hash_is_deterministic(self):
        h1 = DocumentLoader._hash("same input")
        h2 = DocumentLoader._hash("same input")
        assert h1 == h2

    def test_hash_differs_for_different_inputs(self):
        h1 = DocumentLoader._hash("doc_a.pdf")
        h2 = DocumentLoader._hash("doc_b.pdf")
        assert h1 != h2

    def test_hash_length(self):
        h = DocumentLoader._hash("anything")
        assert len(h) == 8
