"""
Text Chunker - ETL Processing Layer
Splits documents into overlapping chunks for embedding.
"""

from src.ingestion.document_loader import Document


class Chunk:
    def __init__(self, content: str, metadata: dict, chunk_index: int):
        self.content = content
        self.metadata = {**metadata, "chunk_index": chunk_index}

    def __repr__(self):
        return f"Chunk(index={self.metadata['chunk_index']}, length={len(self.content)})"


class TextChunker:
    """Split documents into fixed-size overlapping chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=document.metadata.copy(),
                    chunk_index=len(chunks),
                ))
            start += self.chunk_size - self.overlap

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
