"""
Codebase Loader
---------------
Walks a directory tree, applies the correct code chunker per file extension,
embeds all chunks, and persists them in a dedicated ChromaDB collection.

Designed for the ETL codebase (Python + SAS) but works with any directory
containing .py / .sas files.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.processing.code_chunker import CodeChunk, chunk_file, get_chunker
from src.embedding.embedder import Embedder, EmbeddedChunk
from src.retrieval.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Codebase Scanner
# ---------------------------------------------------------------------------

class CodebaseLoader:
    """
    Scan a directory, chunk every supported source file, embed the chunks,
    and store them in a named ChromaDB collection.

    Parameters
    ----------
    root_dir : str
        Root directory to walk (e.g. "etl/").
    collection_name : str
        ChromaDB collection to write to (kept separate from document RAG).
    vector_store_dir : str
        Path to the ChromaDB persist directory.
    embedding_model : str
        sentence-transformers model name.
    extensions : list[str]
        File extensions to index (default: .py and .sas).
    exclude_dirs : list[str]
        Directory names to skip (e.g. __pycache__, .git).
    """

    SUPPORTED_EXTENSIONS = {".py", ".sas"}

    def __init__(
        self,
        root_dir: str,
        collection_name: str = "etl_codebase",
        vector_store_dir: str = "data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        extensions: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.collection_name = collection_name
        self.embedder = Embedder(model_name=embedding_model)
        self.store = VectorStore(
            collection_name=collection_name,
            persist_dir=vector_store_dir,
        )
        self.extensions = {e.lower() for e in (extensions or [".py", ".sas"])}
        self.exclude_dirs = set(exclude_dirs or ["__pycache__", ".git", ".venv", "venv"])

    # ── File discovery ──────────────────────────────────────────────────────

    def discover_files(self) -> list[Path]:
        """Return all indexable source files under *root_dir*."""
        files: list[Path] = []
        for root, dirs, filenames in os.walk(self.root_dir):
            # Prune excluded directories in-place so os.walk won't descend
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for fname in filenames:
                path = Path(root) / fname
                if path.suffix.lower() in self.extensions:
                    files.append(path)
        return sorted(files)

    # ── Chunking ────────────────────────────────────────────────────────────

    def chunk_files(self, files: list[Path]) -> list[CodeChunk]:
        """Chunk every file in *files* using the appropriate code chunker."""
        all_chunks: list[CodeChunk] = []
        for path in files:
            try:
                chunks = chunk_file(str(path))
                print(f"  {str(path.relative_to(self.root_dir.parent)):50s}  →  {len(chunks):3d} chunks")
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"  WARNING: could not chunk {path}: {exc}")
        return all_chunks

    # ── Embedding + storage ─────────────────────────────────────────────────

    def embed_and_store(self, chunks: list[CodeChunk]) -> int:
        """
        Embed *chunks* and add them to the vector store.

        Uses the Embedder's bulk embed API so it works with both the neural
        (sentence-transformers) backend and the offline TF-IDF fallback.

        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        texts     = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        print(f"\n  Embedding {len(chunks)} chunks  [backend: {self.embedder.backend}] …")
        records = self.embedder.embed_texts_bulk(texts, metadatas)

        # Build lightweight EmbeddedChunk-compatible objects the vector store accepts
        embedded: list[EmbeddedChunk] = []
        for rec in records:
            ec = EmbeddedChunk.__new__(EmbeddedChunk)
            ec.content   = rec["content"]
            ec.metadata  = rec["metadata"]
            ec.embedding = rec["embedding"]
            embedded.append(ec)

        self.store.add_chunks(embedded)
        return len(embedded)

    # ── Top-level run ────────────────────────────────────────────────────────

    def index(self) -> dict:
        """
        Full pipeline: discover → chunk → embed → store.

        Returns a summary dict with file counts, chunk counts, and per-file
        breakdown.
        """
        print(f"\n{'='*60}")
        print(f"  Codebase Indexer")
        print(f"  Root       : {self.root_dir}")
        print(f"  Collection : {self.collection_name}")
        print(f"{'='*60}")

        # 1. Discover files
        files = self.discover_files()
        if not files:
            print("  No indexable files found.")
            return {"files": 0, "chunks": 0, "by_language": {}}

        print(f"\n  Found {len(files)} source file(s):\n")

        # 2. Chunk
        all_chunks = self.chunk_files(files)

        # 3. Build stats before embedding
        stats = self._compute_stats(all_chunks, files)
        self._print_stats(stats)

        # 4. Embed + store
        stored = self.embed_and_store(all_chunks)

        stats["stored"] = stored
        stats["collection"] = self.collection_name
        stats["vector_store_total"] = self.store.count()
        print(f"\n  ✓  {stored} chunks stored  |  collection total: {stats['vector_store_total']}")
        print(f"{'='*60}\n")
        return stats

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stats(chunks: list[CodeChunk], files: list[Path]) -> dict:
        by_lang: dict[str, dict] = {}
        for c in chunks:
            lang = c.language
            if lang not in by_lang:
                by_lang[lang] = {"chunks": 0, "by_type": {}}
            by_lang[lang]["chunks"] += 1
            by_lang[lang]["by_type"][c.chunk_type] = (
                by_lang[lang]["by_type"].get(c.chunk_type, 0) + 1
            )
        return {
            "files":       len(files),
            "chunks":      len(chunks),
            "by_language": by_lang,
        }

    @staticmethod
    def _print_stats(stats: dict) -> None:
        print(f"\n  ── Chunk breakdown ──")
        print(f"  Total files  : {stats['files']}")
        print(f"  Total chunks : {stats['chunks']}\n")
        for lang, info in stats["by_language"].items():
            print(f"  {lang.upper()} ({info['chunks']} chunks)")
            for ctype, n in sorted(info["by_type"].items()):
                print(f"    {ctype:<15s}: {n}")
