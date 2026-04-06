"""
reranker.py — Post-retrieval quality pipeline.

Two postprocessors:
1. SimilarityPostprocessor  — hard-filters nodes below score threshold
2. CohereRerank (optional)  — re-scores top-k with a cross-encoder model

If COHERE_API_KEY is absent, only the similarity filter is applied.
The two-stage design means: retrieve broadly → filter → rerank → synthesize.
"""

from llama_index.core.postprocessor import SimilarityPostprocessor
from loguru import logger

from app.config import get_settings


def build_postprocessors() -> list:
    """
    Build the list of node postprocessors to attach to the query engine.
    Order matters: similarity filter first, then reranker.
    """
    cfg = get_settings()
    postprocessors = []

    # Stage 1: similarity threshold filter (always active)
    postprocessors.append(
        SimilarityPostprocessor(similarity_cutoff=cfg.similarity_threshold)
    )

    # Stage 2: Cohere cross-encoder reranker (optional)
    if cfg.use_reranker and cfg.cohere_api_key:
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            reranker = CohereRerank(
                api_key=cfg.cohere_api_key,
                top_n=cfg.rerank_top_n,
                model="rerank-english-v3.0",
            )
            postprocessors.append(reranker)
            logger.info("Cohere reranker enabled")
        except ImportError:
            logger.warning(
                "llama-index-postprocessor-cohere-rerank not installed — "
                "skipping reranker. Run: pip install llama-index-postprocessor-cohere-rerank"
            )
    else:
        logger.info("Reranker disabled (no COHERE_API_KEY). Using similarity filter only.")

    return postprocessors
