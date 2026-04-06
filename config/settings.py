"""
Application settings - loaded from environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    claude_model: str = "claude-opus-4-6"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store
    vector_store_dir: str = "data/vector_store"
    collection_name: str = "rag_collection"

    # Retrieval
    top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
