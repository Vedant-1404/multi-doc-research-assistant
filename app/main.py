"""
main.py — Multi-Doc Research Assistant
Streamlit entry point.

Run with: streamlit run app/main.py
"""

import uuid
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui.styles import CUSTOM_CSS
from app.ui.components import (
    render_source_cards,
    render_retrieval_metrics,
    render_document_library,
    render_empty_state,
    render_welcome_message,
)
from app.ingestion import DocumentLoader, IngestionPipeline
from app.retrieval.engine import QueryEngine
from app.generation.synthesizer import ResponseSynthesizer

# ── Inject CSS ───────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session state bootstrap ──────────────────────────────────────────
def _init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = IngestionPipeline(
            session_id=st.session_state.session_id
        )
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "synthesizer" not in st.session_state:
        st.session_state.synthesizer = ResponseSynthesizer()
    if "loader" not in st.session_state:
        st.session_state.loader = DocumentLoader()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


_init_session()

pipeline: IngestionPipeline = st.session_state.pipeline
loader: DocumentLoader = st.session_state.loader
synthesizer: ResponseSynthesizer = st.session_state.synthesizer


def get_engine():
    """Return or create the query engine once an index exists."""
    if st.session_state.query_engine is None and pipeline.is_ready():
        st.session_state.query_engine = QueryEngine(pipeline.get_index())
    return st.session_state.query_engine


def rebuild_engine():
    """Called after new docs are ingested to hot-swap the index."""
    index = pipeline.get_index()
    if index is not None:
        if st.session_state.query_engine is None:
            st.session_state.query_engine = QueryEngine(index)
        else:
            st.session_state.query_engine.rebuild(index)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown(
        "<p style='color:#8892a4;font-size:0.82rem;margin-top:-0.5rem'>"
        "LlamaIndex · FAISS · GPT-4o-mini</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── PDF Upload ───────────────────────────────────────────────────
    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest PDFs", disabled=not uploaded_files, use_container_width=True):
        new_docs = []
        errors = []
        for f in uploaded_files:
            try:
                docs = loader.load_pdf(f.read(), f.name)
                new_docs.extend(docs)
            except Exception as e:
                errors.append(f"{f.name}: {e}")

        if new_docs:
            with st.spinner(f"Embedding {len(new_docs)} pages…"):
                n = pipeline.add_documents(new_docs)
                rebuild_engine()
            st.success(f"✓ Added {n} chunks from {len(uploaded_files)} PDF(s)")

        for err in errors:
            st.error(f"❌ {err}")

    st.divider()

    # ── URL Ingestion ────────────────────────────────────────────────
    st.markdown("### 🌐 Add Web URLs")
    url_input = st.text_input(
        "Paste a URL",
        placeholder="https://arxiv.org/abs/...",
        label_visibility="collapsed",
    )
    url_label = st.text_input(
        "Label (optional)",
        placeholder="e.g. Attention Is All You Need",
        label_visibility="collapsed",
    )

    if st.button("Fetch & Ingest URL", disabled=not url_input, use_container_width=True):
        with st.spinner("Fetching and parsing page…"):
            try:
                docs = loader.load_url(url_input, label=url_label or None)
                n = pipeline.add_documents(docs)
                rebuild_engine()
                st.success(f"✓ Added {n} chunks from URL")
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()

    # ── Loaded documents ─────────────────────────────────────────────
    st.markdown("### 📚 Loaded Documents")
    render_document_library(pipeline.get_ingested_sources())

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        from app.config import get_settings
        cfg = get_settings()
        st.markdown(
            f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                        color:#8892a4;line-height:1.8">
                embed model: <span style="color:#f5a623">BAAI/bge-small-en-v1.5</span><br>
                chat model: <span style="color:#f5a623">{cfg.groq_chat_model}</span><br>
                chunk size: <span style="color:#f5a623">{cfg.chunk_size}</span><br>
                overlap: <span style="color:#f5a623">{cfg.chunk_overlap}</span><br>
                top-k: <span style="color:#f5a623">{cfg.top_k}</span><br>
                threshold: <span style="color:#f5a623">{cfg.similarity_threshold}</span><br>
                reranker: <span style="color:#f5a623">{'on (Cohere)' if cfg.use_reranker else 'off'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Clear session ────────────────────────────────────────────────
    if st.button("🗑️ Clear Everything", use_container_width=True):
        pipeline.clear()
        st.session_state.query_engine = None
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════

st.markdown("# Research Assistant")
st.markdown(
    "<p style='color:#8892a4;margin-top:-0.5rem;font-size:0.9rem'>"
    "Ask questions across multiple documents. Every answer shows its sources and retrieval confidence.</p>",
    unsafe_allow_html=True,
)

# ── Gate: no documents loaded yet ───────────────────────────────────
if not pipeline.is_ready():
    render_empty_state()
    st.stop()

# ── Welcome message ──────────────────────────────────────────────────
if not st.session_state.messages:
    render_welcome_message()

# ── Chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Re-render source cards for past assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Retrieval Details", expanded=False):
                render_retrieval_metrics(msg["result"])
                render_source_cards(msg["sources"])

# ── Chat input ────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about your documents…")

if query:
    engine = get_engine()
    if engine is None:
        st.error("Index not ready — please ingest documents first.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Run retrieval ────────────────────────────────────────────────
    with st.spinner("Searching documents…"):
        result = engine.query(query)

    # ── Stream answer ────────────────────────────────────────────────
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""

        # Stream tokens
        for chunk in synthesizer.stream_answer(result):
            full_answer += chunk
            answer_placeholder.markdown(full_answer + "▌")

        answer_placeholder.markdown(full_answer)

        # ── Retrieval metrics + sources ──────────────────────────────
        if result.sources:
            with st.expander("Retrieval Details", expanded=True):
                render_retrieval_metrics(result)
                st.divider()
                render_source_cards(result.sources)

    # ── Persist to history ───────────────────────────────────────────
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": result.sources,
            "result": result,
        }
    )
    st.session_state.last_result = result
