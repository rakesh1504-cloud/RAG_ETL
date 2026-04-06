"""Unit tests for DocumentLoader."""

import os
import tempfile
import pytest
from src.ingestion.document_loader import DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader()


def test_load_txt_file(loader, tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello from a text file.", encoding="utf-8")
    doc = loader.load_file(str(txt_file))
    assert doc.content == "Hello from a text file."
    assert doc.metadata["type"] == "text"
    assert doc.metadata["source"] == str(txt_file)


def test_load_unsupported_extension(loader, tmp_path):
    unknown = tmp_path / "file.xyz"
    unknown.write_text("data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_file(str(unknown))


def test_load_directory(loader, tmp_path):
    (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Document B", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skip me", encoding="utf-8")
    docs = list(loader.load_directory(str(tmp_path), extensions=[".txt"]))
    assert len(docs) == 2
    contents = {d.content for d in docs}
    assert "Document A" in contents
    assert "Document B" in contents
