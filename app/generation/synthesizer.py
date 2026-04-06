"""
synthesizer.py — Formats the final answer with source attribution.

Separate from the LlamaIndex synthesizer because we want to:
1. Stream the response token-by-token into Streamlit.
2. Attach a structured citation block after the answer.
3. Add a confidence hedge when retrieval quality is low.
"""

from typing import Generator

from groq import Groq
from loguru import logger

from app.config import get_settings
from app.retrieval.engine import QueryResult, SourceNode


SYSTEM_PROMPT = """You are a precise research assistant. 
Your job is to answer the user's question using ONLY the provided source excerpts.

Rules:
- Cite sources inline using [Source: <name>, p.<page>] notation.
- If the excerpts don't contain enough information, say so clearly.
- Be concise but thorough. Use bullet points for lists.
- Never fabricate information not present in the sources.
- When multiple sources agree, mention that explicitly.
- When sources conflict, flag the discrepancy.
"""


class ResponseSynthesizer:
    """
    Handles final answer formatting and streaming.
    Used as a thin layer over the raw QueryResult from the engine.
    """

    def __init__(self):
        self.cfg = get_settings()
        import os
        os.environ["GROQ_API_KEY"] = self.cfg.groq_api_key
        self._client = Groq()

    def stream_answer(self, result: QueryResult) -> Generator[str, None, None]:
        """
        Re-synthesize the answer with inline citations using streaming.
        Yields text chunks for Streamlit's st.write_stream().

        Why re-synthesize instead of using engine answer directly?
        LlamaIndex's compact synthesizer is great for accuracy but doesn't
        do inline [Source: X] citations. We do a cheap second pass here.
        """
        if not result.sources:
            yield (
                "⚠️ I couldn't find relevant information in the uploaded documents "
                "to answer this question. Try uploading more relevant sources or "
                "rephrasing the question."
            )
            return

        context_block = self._build_context(result.sources)
        hedge = self._confidence_hedge(result)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {result.query}\n\n"
                    f"Source excerpts:\n{context_block}\n\n"
                    f"Answer the question with inline citations."
                ),
            },
        ]

        try:
            stream = self._client.chat.completions.create(
                model=self.cfg.groq_chat_model,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=1024,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

            if hedge:
                yield f"\n\n{hedge}"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\n❌ Error generating response: {e}"

    def format_sources_markdown(self, sources: list[SourceNode]) -> str:
        """Render the source cards shown below the answer."""
        if not sources:
            return ""

        lines = ["### 📚 Sources Used\n"]
        for i, src in enumerate(sources, 1):
            icon = "📄" if src.source_type == "pdf" else "🌐"
            score_bar = self._score_bar(src.score)
            lines.append(
                f"**{i}. {icon} {src.source_name}** — "
                f"{'p.' + src.page_label if src.source_type == 'pdf' else src.page_label}  \n"
                f"Relevance: {score_bar} `{src.score:.3f}`  \n"
                f"> {src.text[:300].strip()}{'...' if len(src.text) > 300 else ''}\n"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, sources: list[SourceNode]) -> str:
        """Format source nodes into a numbered context block for the LLM."""
        blocks = []
        for i, src in enumerate(sources, 1):
            label = (
                f"p.{src.page_label}" if src.source_type == "pdf"
                else src.page_label
            )
            blocks.append(
                f"[{i}] Source: {src.source_name} | {label} | Score: {src.score:.3f}\n"
                f"{src.text.strip()}"
            )
        return "\n\n---\n\n".join(blocks)

    def _confidence_hedge(self, result: QueryResult) -> str:
        """Add a note when retrieval confidence is borderline."""
        if not result.found:
            return (
                "_⚠️ Note: The retrieved excerpts had low relevance scores. "
                "This answer may be incomplete — consider uploading additional sources._"
            )
        return ""

    @staticmethod
    def _score_bar(score: float) -> str:
        """Visual relevance bar using Unicode blocks."""
        filled = round(score * 10)
        filled = max(0, min(10, filled))
        return "█" * filled + "░" * (10 - filled)
