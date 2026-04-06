"""
Vector Store - Stores and retrieves embedded chunks via similarity search.
Uses ChromaDB by default; can be swapped for FAISS, Pinecone, etc.
"""

import json
from src.embedding.embedder import EmbeddedChunk


class VectorStore:
    """Chroma-backed vector store for RAG retrieval."""

    def __init__(self, collection_name: str = "rag_collection", persist_dir: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = client.get_or_create_collection(self.collection_name)
            except ImportError:
                raise ImportError("Install chromadb: pip install chromadb")
        return self._collection

    def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        ids = [f"{ec.metadata.get('source', 'doc')}_{ec.metadata.get('chunk_index', i)}"
               for i, ec in enumerate(embedded_chunks)]
        documents = [ec.content for ec in embedded_chunks]
        embeddings = [ec.embedding for ec in embedded_chunks]
        metadatas = [{k: str(v) for k, v in ec.metadata.items()} for ec in embedded_chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"content": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self.collection.count()
