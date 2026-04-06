from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Groq ---
    groq_api_key: str
    groq_chat_model: str = "llama-3.1-8b-instant"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    top_k: int = 6
    similarity_threshold: float = 0.3

    # --- Reranking (disabled by default) ---
    cohere_api_key: str = ""
    use_reranker: bool = False
    rerank_top_n: int = 3

    # --- Storage ---
    faiss_index_dir: Path = Path("./storage/faiss_indexes")

    def model_post_init(self, __context):
        if self.cohere_api_key:
            object.__setattr__(self, "use_reranker", True)
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()