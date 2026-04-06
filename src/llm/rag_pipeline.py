"""
RAG Pipeline - Orchestrates the full ETL + Retrieval + Generation flow.
"""

from src.ingestion.document_loader import DocumentLoader
from src.processing.chunker import TextChunker
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.llm.claude_client import ClaudeRAGClient


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Ingest documents (ETL)
      2. Chunk text
      3. Embed chunks
      4. Store in vector DB
      5. Retrieve relevant chunks for a query
      6. Generate answer via Claude
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_dir: str = "data/vector_store",
        top_k: int = 5,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(persist_dir=vector_store_dir)
        self.llm = ClaudeRAGClient()
        self.top_k = top_k

    # --- ETL Phase ---

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file. Returns number of chunks stored."""
        doc = self.loader.load_file(file_path)
        chunks = self.chunker.chunk_document(doc)
        embedded = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(embedded)
        print(f"Ingested '{file_path}': {len(chunks)} chunks stored.")
        return len(chunks)

    def ingest_directory(self, dir_path: str, extensions: list[str] | None = None) -> int:
        """Ingest all supported files from a directory."""
        docs = list(self.loader.load_directory(dir_path, extensions))
        chunks = self.chunker.chunk_documents(docs)
        embedded = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(embedded)
        print(f"Ingested {len(docs)} docs from '{dir_path}': {len(chunks)} chunks stored.")
        return len(chunks)

    # --- Query Phase ---

    def query(self, question: str) -> str:
        """Retrieve relevant context and generate an answer using Claude."""
        query_embedding = self.embedder.embed_query(question)
        retrieved = self.vector_store.search(query_embedding, top_k=self.top_k)
        print(f"\nRetrieved {len(retrieved)} relevant chunks.")
        answer = self.llm.answer(question, retrieved)
        return answer

    def stats(self) -> dict:
        return {"total_chunks": self.vector_store.count()}
