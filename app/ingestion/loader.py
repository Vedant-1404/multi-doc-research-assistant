"""
loader.py — Loads documents from uploaded PDF files or web URLs.

Returns a list of LlamaIndex Document objects with rich metadata:
  - source_type: "pdf" | "url"
  - source_name: filename or domain
  - source_path: full path or URL
  - page_label: page number (PDFs) or section (URLs)
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from loguru import logger


class DocumentLoader:
    """Unified loader for PDFs (file upload) and web URLs."""

    def __init__(self):
        self._pdf_reader = PDFReader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pdf(self, file_bytes: bytes, filename: str) -> list[Document]:
        """
        Load a PDF from raw bytes (e.g. Streamlit file_uploader).
        Returns one Document per page with page metadata.
        """
        logger.info(f"Loading PDF: {filename}")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            docs = self._pdf_reader.load_data(file=tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        doc_id = self._hash(filename)
        enriched = []
        for i, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "source_type": "pdf",
                    "source_name": filename,
                    "source_path": filename,
                    "page_label": str(i + 1),
                    "doc_id": doc_id,
                }
            )
            enriched.append(doc)

        logger.success(f"Loaded {len(enriched)} pages from {filename}")
        return enriched

    def load_url(self, url: str, label: Optional[str] = None) -> list[Document]:
        """
        Fetch a web page and convert it to LlamaIndex Documents.
        Splits on <h2>/<h3> boundaries for logical sections.
        """
        logger.info(f"Fetching URL: {url}")
        domain = urlparse(url).netloc
        source_name = label or domain

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "ResearchBot/1.0"})
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise ValueError(f"Could not fetch URL: {url}\n{e}")

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        sections = self._split_into_sections(soup, url, source_name)
        logger.success(f"Extracted {len(sections)} sections from {url}")
        return sections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_sections(
        self, soup: BeautifulSoup, url: str, source_name: str
    ) -> list[Document]:
        """Split a page into logical sections at heading boundaries."""
        doc_id = self._hash(url)
        sections: list[Document] = []
        current_heading = "Introduction"
        current_text: list[str] = []

        def flush(heading: str, texts: list[str], idx: int):
            text = " ".join(texts).strip()
            if len(text) < 80:          # skip near-empty sections
                return
            sections.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "url",
                        "source_name": source_name,
                        "source_path": url,
                        "page_label": heading,
                        "section_index": str(idx),
                        "doc_id": doc_id,
                    },
                )
            )

        section_idx = 0
        body = soup.find("body") or soup
        for el in body.find_all(True):
            if el.name in ("h1", "h2", "h3"):
                flush(current_heading, current_text, section_idx)
                current_heading = el.get_text(strip=True) or current_heading
                current_text = []
                section_idx += 1
            elif el.name in ("p", "li", "td", "blockquote"):
                text = el.get_text(separator=" ", strip=True)
                if text:
                    current_text.append(text)

        flush(current_heading, current_text, section_idx)

        # Fallback: whole page as single doc if no sections parsed
        if not sections:
            full_text = body.get_text(separator="\n", strip=True)
            sections.append(
                Document(
                    text=full_text[:8000],
                    metadata={
                        "source_type": "url",
                        "source_name": source_name,
                        "source_path": url,
                        "page_label": "Full Page",
                        "doc_id": doc_id,
                    },
                )
            )

        return sections

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.md5(value.encode()).hexdigest()[:8]
