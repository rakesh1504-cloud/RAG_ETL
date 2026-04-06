"""Unit tests for TextChunker."""

import pytest
from src.ingestion.document_loader import Document
from src.processing.chunker import TextChunker


def make_doc(text: str) -> Document:
    return Document(content=text, metadata={"source": "test"})


def test_chunk_short_document():
    chunker = TextChunker(chunk_size=100, overlap=10)
    doc = make_doc("Hello world. This is a short document.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world. This is a short document."


def test_chunk_long_document():
    chunker = TextChunker(chunk_size=10, overlap=2)
    doc = make_doc("A" * 25)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    chunker = TextChunker(chunk_size=50, overlap=5)
    doc = make_doc("Some content " * 10)
    chunks = chunker.chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["chunk_index"] == i


def test_chunk_index_increments():
    chunker = TextChunker(chunk_size=20, overlap=5)
    doc = make_doc("word " * 30)
    chunks = chunker.chunk_document(doc)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
