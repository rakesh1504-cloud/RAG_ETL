"""
Embedder - Converts text chunks to vector embeddings.
Uses sentence-transformers by default; easily swappable.
"""

from src.processing.chunker import Chunk


class EmbeddedChunk:
    def __init__(self, chunk: Chunk, embedding: list[float]):
        self.content = chunk.content
        self.metadata = chunk.metadata
        self.embedding = embedding


class Embedder:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        return self._model

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return [
            EmbeddedChunk(chunk=chunk, embedding=emb.tolist())
            for chunk, emb in zip(chunks, embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()
