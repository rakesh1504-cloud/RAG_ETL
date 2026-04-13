"""
Azure Knowledge Base Loader
---------------------------
Indexes the ETL codebase into the Azure knowledge base:

  1. Discover files (.py, .sas, .md, .yaml)
  2. Chunk using code-aware / doc-aware chunkers
  3. Upload raw files to Azure Blob Storage  (optional but recommended)
  4. Embed chunks with Azure OpenAI text-embedding-3-small
  5. Index into Azure AI Search (hybrid vector+BM25+semantic index)

This produces a fully Azure-hosted knowledge base that scales to millions
of documents without any local infrastructure.

Environment variables
  AZURE_OPENAI_ENDPOINT         Azure OpenAI resource URL
  AZURE_OPENAI_API_KEY          optional (managed identity if unset)
  AZURE_OPENAI_EMBED_DEPLOY     embedding model deployment name
  AZURE_OPENAI_API_VERSION      API version string
  AZURE_SEARCH_ENDPOINT         Azure AI Search endpoint
  AZURE_SEARCH_API_KEY          optional (managed identity if unset)
  AZURE_SEARCH_INDEX            index name (default: etl-codebase)
  AZURE_STORAGE_CONNECTION_STR  Blob Storage connection string (optional)
  AZURE_STORAGE_CONTAINER       container name (default: etl-codebase-docs)
"""

from __future__ import annotations

import os
from pathlib import Path

from src.processing.code_chunker import chunk_file
from src.processing.doc_chunker  import chunk_doc_file, is_doc_file
from src.embedding.azure_openai_embedder import AzureOpenAIEmbedder
from src.retrieval.azure_search_store    import AzureSearchStore


# ---------------------------------------------------------------------------
# Azure Knowledge Base Loader
# ---------------------------------------------------------------------------

class AzureKnowledgeBaseLoader:
    """
    Walk a directory, chunk all supported files, embed with Azure OpenAI,
    and index into Azure AI Search.  Optionally backs up raw files to
    Azure Blob Storage for audit / re-indexing purposes.

    Parameters
    ----------
    root_dirs : list[str]
        Root directories to index.  Pass multiple to index code + docs together.
        e.g. ["etl/", "docs/", "data/schemas/"]
    index_name : str
        Azure AI Search index name.
    aoai_endpoint : str | None
        Azure OpenAI endpoint.  Defaults to AZURE_OPENAI_ENDPOINT.
    aoai_api_key : str | None
        Azure OpenAI key.  Leave None for managed identity.
    embed_deployment : str
        Embedding deployment name.  Default: "text-embedding-3-small".
    search_endpoint : str | None
        Azure AI Search endpoint.  Defaults to AZURE_SEARCH_ENDPOINT.
    search_api_key : str | None
        Azure AI Search key.  Leave None for managed identity.
    storage_connection_str : str | None
        Azure Blob Storage connection string.  Pass None to skip Blob upload.
    storage_container : str
        Blob container name.  Default: "etl-codebase-docs".
    extensions : list[str]
        File extensions to index.  Default: .py, .sas, .md, .yaml, .yml
    exclude_dirs : list[str]
        Directory names to skip.
    """

    SUPPORTED_EXTENSIONS = {".py", ".sas", ".md", ".yaml", ".yml"}

    def __init__(
        self,
        root_dirs: list[str],
        index_name: str                  = "etl-codebase",
        aoai_endpoint: str | None        = None,
        aoai_api_key: str | None         = None,
        embed_deployment: str            = "",
        search_endpoint: str | None      = None,
        search_api_key: str | None       = None,
        storage_connection_str: str | None = None,
        storage_container: str           = "etl-codebase-docs",
        extensions: list[str] | None     = None,
        exclude_dirs: list[str] | None   = None,
    ):
        self.root_dirs   = [Path(d) for d in root_dirs]
        self.extensions  = {e.lower() for e in (extensions or list(self.SUPPORTED_EXTENSIONS))}
        self.exclude_dirs = set(exclude_dirs or ["__pycache__", ".git", ".venv", "venv",
                                                  "vector_store", ".pytest_cache"])

        self.embedder = AzureOpenAIEmbedder(
            endpoint   = aoai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key    = aoai_api_key  or os.environ.get("AZURE_OPENAI_API_KEY"),
            deployment = embed_deployment or os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
                                                            "text-embedding-3-small"),
        )

        self.store = AzureSearchStore(
            endpoint   = search_endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", ""),
            api_key    = search_api_key  or os.environ.get("AZURE_SEARCH_API_KEY"),
            index_name = index_name,
        )

        self._storage_connection_str = (storage_connection_str or
                                        os.environ.get("AZURE_STORAGE_CONNECTION_STR"))
        self._storage_container      = storage_container

    # ── File discovery ────────────────────────────────────────────────────

    def discover_files(self) -> list[Path]:
        files: list[Path] = []
        for root_dir in self.root_dirs:
            for root, dirs, filenames in os.walk(root_dir):
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
                for fname in filenames:
                    path = Path(root) / fname
                    if path.suffix.lower() in self.extensions:
                        files.append(path)
        return sorted(files)

    # ── Blob Storage upload ───────────────────────────────────────────────

    def upload_to_blob(self, files: list[Path]) -> int:
        """
        Upload raw files to Azure Blob Storage.
        Returns the number of files uploaded.  Skips if no connection string.
        """
        if not self._storage_connection_str:
            print("  [Blob] Skipping blob upload (AZURE_STORAGE_CONNECTION_STR not set).")
            return 0

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            print("  [Blob] azure-storage-blob not installed. Skipping upload.")
            return 0

        client    = BlobServiceClient.from_connection_string(self._storage_connection_str)
        container = client.get_container_client(self._storage_container)

        # Create container if it doesn't exist
        try:
            container.create_container()
            print(f"  [Blob] Created container '{self._storage_container}'.")
        except Exception:
            pass   # container already exists

        uploaded = 0
        for path in files:
            blob_name = str(path).replace("\\", "/")
            blob_client = container.get_blob_client(blob_name)
            with open(path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            uploaded += 1

        print(f"  [Blob] Uploaded {uploaded} files to '{self._storage_container}'.")
        return uploaded

    # ── Chunking ──────────────────────────────────────────────────────────

    def chunk_files(self, files: list[Path]) -> list:
        all_chunks = []
        for path in files:
            try:
                chunks = chunk_doc_file(str(path)) if is_doc_file(path) else chunk_file(str(path))
                print(f"  {str(path):55s}  →  {len(chunks):3d} chunks")
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"  WARNING: {path}: {exc}")
        return all_chunks

    # ── Embed + index ─────────────────────────────────────────────────────

    def embed_and_index(self, chunks: list) -> int:
        """Embed chunks with Azure OpenAI and upload to Azure AI Search."""
        if not chunks:
            return 0

        texts     = [c.content  for c in chunks]
        metadatas = [c.metadata for c in chunks]

        print(f"\n  Embedding {len(chunks)} chunks with Azure OpenAI "
              f"({self.embedder.deployment}) …")
        records = self.embedder.embed_texts_bulk(texts, metadatas)

        # Build EmbeddedChunk-like objects
        class _EC:
            pass

        embedded = []
        for i, rec in enumerate(records):
            ec           = _EC()
            ec.content   = rec["content"]
            ec.metadata  = {**rec["metadata"], "chunk_index": i}
            ec.embedding = rec["embedding"]
            embedded.append(ec)

        print(f"  Uploading {len(embedded)} documents to Azure AI Search …")
        self.store.add_chunks(embedded)
        return len(embedded)

    # ── Top-level run ─────────────────────────────────────────────────────

    def index(self, upload_blobs: bool = False, drop_existing: bool = False) -> dict:
        """
        Full pipeline: discover → (upload blobs) → chunk → embed → index.

        Parameters
        ----------
        upload_blobs : bool
            Upload raw files to Azure Blob Storage before indexing.
        drop_existing : bool
            Drop and recreate the Azure AI Search index before indexing.
            Use this for a full rebuild.

        Returns
        -------
        dict  Summary with file/chunk/stored counts.
        """
        print(f"\n{'='*65}")
        print(f"  Azure Knowledge Base Loader")
        print(f"  Roots      : {', '.join(str(d) for d in self.root_dirs)}")
        print(f"  Index      : {self.store.index_name}")
        print(f"  Embed model: {self.embedder.deployment}")
        print(f"{'='*65}\n")

        # 1. Discover
        files = self.discover_files()
        if not files:
            print("  No indexable files found.")
            return {"files": 0, "chunks": 0, "stored": 0}
        print(f"  Found {len(files)} file(s):\n")

        # 2. Optional blob upload
        if upload_blobs:
            self.upload_to_blob(files)

        # 3. Optional index reset
        if drop_existing:
            try:
                self.store.drop_index()
            except Exception:
                pass

        # 4. Chunk
        chunks = self.chunk_files(files)

        # 5. Stats
        by_lang: dict[str, int] = {}
        for c in chunks:
            by_lang[c.language] = by_lang.get(c.language, 0) + 1
        print(f"\n  Total chunks: {len(chunks)}")
        for lang, n in sorted(by_lang.items()):
            print(f"    {lang:<12}: {n}")

        # 6. Embed + index
        stored = self.embed_and_index(chunks)

        print(f"\n  ✓ {stored} chunks indexed into '{self.store.index_name}'")
        print(f"{'='*65}\n")
        return {"files": len(files), "chunks": len(chunks), "stored": stored}
