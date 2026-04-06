"""
components.py — Reusable Streamlit UI building blocks.
"""

import streamlit as st
from app.retrieval.engine import QueryResult, SourceNode


# ------------------------------------------------------------------
# Source card rendering
# ------------------------------------------------------------------

def render_source_cards(sources: list[SourceNode]):
    """Render retrieved source chunks as styled cards."""
    if not sources:
        return

    st.markdown("#### 📚 Retrieved Sources")
    for i, src in enumerate(sources, 1):
        icon = "📄" if src.source_type == "pdf" else "🌐"
        score_pct = int(src.score * 100)
        score_color = (
            "#4ade80" if src.score >= 0.7
            else "#f5a623" if src.score >= 0.5
            else "#f87171"
        )
        page_display = (
            f"p. {src.page_label}" if src.source_type == "pdf"
            else src.page_label
        )

        with st.container():
            st.markdown(
                f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                            color:{score_color};margin-bottom:0.5rem;">
                    ▸ Relevance: {_score_bar(src.score)} {src.score:.4f}
                    &nbsp;&nbsp;|&nbsp;&nbsp; 
                    Source: {src.source_path if src.source_type == 'url' else src.source_name}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"> {src.text[:500].strip()}{'...' if len(src.text) > 500 else ''}"
            )


def render_retrieval_metrics(result: QueryResult):
    """Render compact retrieval quality stats as metric columns."""
    cols = st.columns(4)

    with cols[0]:
        st.metric("Sources Used", result.final_count)
    with cols[1]:
        st.metric("Chunks Retrieved", result.retrieval_count)
    with cols[2]:
        st.metric("Context Tokens", f"~{result.tokens_in_context:,}")
    with cols[3]:
        top_score = result.sources[0].score if result.sources else 0
        confidence = (
            "🟢 High" if top_score >= 0.7
            else "🟡 Medium" if top_score >= 0.5
            else "🔴 Low"
        )
        st.metric("Top Match", confidence)


# ------------------------------------------------------------------
# Document library sidebar
# ------------------------------------------------------------------

def render_document_library(sources: list[dict]):
    """Show ingested documents in the sidebar."""
    if not sources:
        st.info("No documents loaded yet.")
        return

    st.markdown("**Loaded documents:**")
    for src in sources:
        icon = "📄" if src["type"] == "pdf" else "🌐"
        st.markdown(
            f"""
            <div class="doc-badge">
                {icon} {src['name']}
                <span style="margin-left:auto;color:#f5a623">{src['chunks']} chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Empty / loading states
# ------------------------------------------------------------------

def render_empty_state():
    """Shown when no documents are loaded."""
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#8892a4;">
            <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
            <p style="font-size:1.1rem;font-weight:600;color:#e8eaf0">
                No documents loaded
            </p>
            <p style="font-size:0.88rem">
                Upload PDFs or add URLs in the sidebar to get started.
                <br>You can mix and match multiple sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_message():
    """First message shown in the chat area."""
    st.markdown(
        """
        <div style="background:#1a1f2e;border:1px solid #2d3452;border-left:3px solid #f5a623;
                    border-radius:8px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
            <p style="font-family:'Libre Baskerville',serif;font-size:1rem;
                      color:#e8eaf0;margin:0 0 0.5rem 0;font-weight:700;">
                Ready to research
            </p>
            <p style="color:#8892a4;font-size:0.85rem;margin:0;line-height:1.6">
                Load your documents on the left, then ask anything.<br>
                Every answer includes <strong style="color:#f5a623">source citations</strong> 
                and <strong style="color:#f5a623">retrieval scores</strong> 
                so you can judge the evidence quality yourself.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _score_bar(score: float) -> str:
    filled = round(score * 10)
    filled = max(0, min(10, filled))
    return "█" * filled + "░" * (10 - filled)
