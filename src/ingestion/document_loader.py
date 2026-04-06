"""
Document Loader - ETL Ingestion Layer
Supports PDF, TXT, DOCX, HTML, and web URLs.
"""

import os
from pathlib import Path
from typing import Iterator


class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(source={self.metadata.get('source')}, length={len(self.content)})"


class DocumentLoader:
    """Load documents from various sources into a unified format."""

    def load_file(self, file_path: str) -> Document:
        path = Path(file_path)
        suffix = path.suffix.lower()

        loaders = {
            ".txt": self._load_text,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".html": self._load_html,
            ".md": self._load_text,
        }

        loader = loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader(file_path)

    def load_directory(self, dir_path: str, extensions: list[str] | None = None) -> Iterator[Document]:
        extensions = extensions or [".txt", ".pdf", ".docx", ".md"]
        for root, _, files in os.walk(dir_path):
            for filename in files:
                path = os.path.join(root, filename)
                if any(filename.endswith(ext) for ext in extensions):
                    try:
                        yield self.load_file(path)
                    except Exception as e:
                        print(f"Warning: Could not load {path}: {e}")

    def _load_text(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Document(content=content, metadata={"source": file_path, "type": "text"})

    def _load_pdf(self, file_path: str) -> Document:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")
        return Document(content=content, metadata={"source": file_path, "type": "pdf"})

    def _load_docx(self, file_path: str) -> Document:
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            content = "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
        return Document(content=content, metadata={"source": file_path, "type": "docx"})

    def _load_html(self, file_path: str) -> Document:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            content = soup.get_text(separator="\n", strip=True)
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")
        return Document(content=content, metadata={"source": file_path, "type": "html"})
