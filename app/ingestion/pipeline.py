"""
pipeline.py — Ingestion pipeline: chunk → embed → store in FAISS.

Key design choices:
- SentenceSplitter preserves sentence boundaries (better than naive char split).
- Each session gets its own FAISS index path so multiple tabs don't collide.
- Index is persisted to disk so the app can reload without re-embedding.
"""

import json
import uuid
from pathlib import Path
from typing import Optional

import faiss
from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from loguru import logger

from app.config import get_settings


class IngestionPipeline:
    """
    Manages the full ingest → index lifecycle.

    Usage:
        pipeline = IngestionPipeline(session_id="abc123")
        pipeline.add_documents(docs)
        index = pipeline.get_index()
    """

    def __init__(self, session_id: Optional[str] = None):
        self.cfg = get_settings()
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.index_dir = self.cfg.faiss_index_dir / self.session_id
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Configure LlamaIndex globals
        LISettings.embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )
        LISettings.chunk_size = self.cfg.chunk_size
        LISettings.chunk_overlap = self.cfg.chunk_overlap

        self._splitter = SentenceSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        self._index: Optional[VectorStoreIndex] = None
        self._ingested_sources: list[dict] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> int:
        """
        Chunk, embed, and upsert documents into the FAISS index.
        Returns the number of nodes added.
        """
        if not documents:
            return 0

        logger.info(f"Ingesting {len(documents)} documents into session {self.session_id}")
        nodes = self._splitter.get_nodes_from_documents(documents)
        logger.info(f"  → {len(nodes)} chunks after splitting")

        if self._index is None:
            self._index = self._build_index(nodes)
        else:
            self._index.insert_nodes(nodes)
            self._persist()

        # Track source metadata for UI display
        source_names = {d.metadata.get("source_name", "unknown") for d in documents}
        for name in source_names:
            if not any(s["name"] == name for s in self._ingested_sources):
                meta = next(
                    (d.metadata for d in documents if d.metadata.get("source_name") == name),
                    {},
                )
                self._ingested_sources.append(
                    {
                        "name": name,
                        "type": meta.get("source_type", "unknown"),
                        "path": meta.get("source_path", ""),
                        "chunks": sum(
                            1
                            for n in nodes
                            if n.metadata.get("source_name") == name
                        ),
                    }
                )
        self._save_manifest()
        logger.success(f"  → Index updated. Total sources: {len(self._ingested_sources)}")
        return len(nodes)

    def get_index(self) -> Optional[VectorStoreIndex]:
        """Return the current index, loading from disk if needed."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def get_ingested_sources(self) -> list[dict]:
        return self._ingested_sources

    def is_ready(self) -> bool:
        return self.get_index() is not None

    def clear(self):
        """Wipe the index and start fresh."""
        import shutil
        shutil.rmtree(self.index_dir, ignore_errors=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._ingested_sources = []
        logger.info(f"Cleared index for session {self.session_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, nodes) -> VectorStoreIndex:
        """Create a new FAISS index from nodes."""
        embed_dim = 384  # BAAI/bge-small-en-v1.5 dimensions

        faiss_index = faiss.IndexFlatIP(embed_dim)   # Inner product = cosine on normalized vecs
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        self._persist(index=index, faiss_index=faiss_index)
        return index

    def _persist(self, index=None, faiss_index=None):
        """Persist LlamaIndex docstore + FAISS binary to disk."""
        idx = index or self._index
        if idx is None:
            return
        idx.storage_context.persist(persist_dir=str(self.index_dir))
        logger.debug(f"Index persisted to {self.index_dir}")

    def _load_index(self) -> Optional[VectorStoreIndex]:
        """Reload index from disk if it exists."""
        faiss_path = self.index_dir / "vector_store.json"
        if not faiss_path.exists():
            return None
        try:
            from llama_index.core import load_index_from_storage
            vector_store = FaissVectorStore.from_persist_dir(str(self.index_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.index_dir),
            )
            index = load_index_from_storage(storage_context)
            logger.info(f"Loaded existing index from {self.index_dir}")
            return index
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            return None

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _load_manifest(self) -> list[dict]:
        p = self._manifest_path()
        if p.exists():
            return json.loads(p.read_text())
        return []

    def _save_manifest(self):
        self._manifest_path().write_text(json.dumps(self._ingested_sources, indent=2))
