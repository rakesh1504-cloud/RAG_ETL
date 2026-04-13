"""
Azure AI Search Store
---------------------
Wraps the Azure AI Search SDK to provide the same interface as VectorStore
(ChromaDB) so the rest of the pipeline works without changes.

Azure AI Search advantages over ChromaDB
  • Hybrid search: vector (HNSW) + BM25 keyword in a single API call
  • Semantic reranking: cross-encoder reranker for top-K results (no extra code)
  • Metadata filtering: OData $filter expressions — rich, production-grade
  • Managed service: no self-hosting, scales automatically, SLA-backed
  • Integrated with Azure OpenAI, Blob Storage, and Azure RBAC

Index schema
  Each document in the index has:
    id            – unique string key (e.g. etl/python/pipeline.py_3)
    content       – full chunk text (searchable, used for BM25 + semantic)
    embedding     – dense float vector (used for ANN / HNSW vector search)
    language      – filterable: "python" | "sas" | "markdown" | "yaml"
    chunk_type    – filterable: function | class | macro | heading | ...
    name          – the function/macro/section name
    source        – relative file path
    start_line    – integer
    end_line      – integer

Authentication
  Supports both:
    • API key  (AZURE_SEARCH_API_KEY env var)
    • Azure DefaultAzureCredential (managed identity, az login, etc.)

Environment variables
  AZURE_SEARCH_ENDPOINT   e.g. https://my-search.search.windows.net
  AZURE_SEARCH_API_KEY    optional; leave unset to use DefaultAzureCredential
  AZURE_SEARCH_INDEX      index name, default: etl-codebase
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Index field definitions
# ---------------------------------------------------------------------------

VECTOR_FIELD      = "embedding"
CONTENT_FIELD     = "content"
KEY_FIELD         = "id"
FILTERABLE_FIELDS = {"language", "chunk_type", "name", "source"}
VECTOR_DIMENSIONS = 1536      # text-embedding-3-small default


def _build_index_schema(index_name: str, dimensions: int = VECTOR_DIMENSIONS) -> Any:
    """Return a SearchIndex object with vector + keyword + filterable fields."""
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile,
        SemanticConfiguration,
        SemanticSearch,
        SemanticPrioritizedFields,
        SemanticField,
    )

    fields = [
        SimpleField(name="id",         type=SearchFieldDataType.String,  key=True, filterable=True),
        SearchableField(name="content",  type=SearchFieldDataType.String,  analyzer_name="en.lucene"),
        SimpleField(name="language",   type=SearchFieldDataType.String,  filterable=True, facetable=True),
        SimpleField(name="chunk_type", type=SearchFieldDataType.String,  filterable=True, facetable=True),
        SimpleField(name="name",       type=SearchFieldDataType.String,  filterable=True),
        SimpleField(name="source",     type=SearchFieldDataType.String,  filterable=True),
        SimpleField(name="start_line", type=SearchFieldDataType.Int32,   filterable=True),
        SimpleField(name="end_line",   type=SearchFieldDataType.Int32,   filterable=True),
        SearchField(
            name             = VECTOR_FIELD,
            type             = SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable       = True,
            vector_search_dimensions = dimensions,
            vector_search_profile_name = "etl-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="etl-hnsw")],
        profiles=[VectorSearchProfile(
            name              = "etl-vector-profile",
            algorithm_configuration_name = "etl-hnsw",
        )],
    )

    semantic_config = SemanticConfiguration(
        name="etl-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[
                SemanticField(field_name="name"),
                SemanticField(field_name="chunk_type"),
            ],
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    return SearchIndex(
        name           = index_name,
        fields         = fields,
        vector_search  = vector_search,
        semantic_search= semantic_search,
    )


# ---------------------------------------------------------------------------
# Azure Search Store
# ---------------------------------------------------------------------------

class AzureSearchStore:
    """
    Azure AI Search-backed vector + keyword store.

    Drop-in replacement for VectorStore (ChromaDB) with the same
    add_chunks() / search() / count() interface plus Azure-specific features.

    Parameters
    ----------
    endpoint : str | None
        Azure AI Search endpoint URL.  Defaults to AZURE_SEARCH_ENDPOINT env var.
    api_key : str | None
        Admin API key.  Defaults to AZURE_SEARCH_API_KEY env var.
        Leave None to use DefaultAzureCredential (managed identity / az login).
    index_name : str
        Name of the search index.  Created automatically if it does not exist.
    dimensions : int
        Embedding vector size.  Must match the embedder output (default: 1536).
    semantic_reranking : bool
        Enable Azure semantic reranker on top-K results.  Requires the index
        to be in a region that supports semantic search (most regions do).
    """

    def __init__(
        self,
        endpoint: str | None    = None,
        api_key: str | None     = None,
        index_name: str         = "etl-codebase",
        dimensions: int         = VECTOR_DIMENSIONS,
        semantic_reranking: bool = True,
    ):
        self.endpoint           = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")
        self.api_key            = api_key  or os.environ.get("AZURE_SEARCH_API_KEY")
        self.index_name         = index_name
        self.dimensions         = dimensions
        self.semantic_reranking = semantic_reranking

        if not self.endpoint:
            raise ValueError(
                "Azure Search endpoint is required. "
                "Set AZURE_SEARCH_ENDPOINT or pass endpoint= parameter."
            )

        self._search_client  = None
        self._index_client   = None
        self._index_ready    = False

    # ── Client initialisation ─────────────────────────────────────────────

    def _get_credential(self):
        if self.api_key:
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(self.api_key)
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential()

    @property
    def search_client(self):
        if self._search_client is None:
            from azure.search.documents import SearchClient
            self._ensure_index()
            self._search_client = SearchClient(
                endpoint    = self.endpoint,
                index_name  = self.index_name,
                credential  = self._get_credential(),
            )
        return self._search_client

    @property
    def index_client(self):
        if self._index_client is None:
            from azure.search.documents.indexes import SearchIndexClient
            self._index_client = SearchIndexClient(
                endpoint   = self.endpoint,
                credential = self._get_credential(),
            )
        return self._index_client

    # ── Index management ──────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Create the index if it does not already exist."""
        if self._index_ready:
            return
        existing = [idx.name for idx in self.index_client.list_index_names()]
        if self.index_name not in existing:
            schema = _build_index_schema(self.index_name, self.dimensions)
            self.index_client.create_index(schema)
            print(f"  [AzureSearchStore] Created index '{self.index_name}'.")
        else:
            print(f"  [AzureSearchStore] Using existing index '{self.index_name}'.")
        self._index_ready = True

    def drop_index(self) -> None:
        """Delete the index and all its documents. Requires re-indexing."""
        self.index_client.delete_index(self.index_name)
        self._index_ready    = False
        self._search_client  = None
        print(f"  [AzureSearchStore] Deleted index '{self.index_name}'.")

    # ── Write ─────────────────────────────────────────────────────────────

    def add_chunks(self, embedded_chunks) -> None:
        """
        Upload embedded chunks to Azure AI Search.

        Accepts the same EmbeddedChunk-like objects as VectorStore.add_chunks().
        Uploads in batches of 1000 (Azure SDK limit per request).
        """
        documents = []
        for i, ec in enumerate(embedded_chunks):
            meta = ec.metadata if hasattr(ec, "metadata") else {}
            doc_id = f"{meta.get('source', 'doc')}_{meta.get('chunk_index', i)}"
            # Azure Search IDs must be URL-safe: replace slashes and dots
            doc_id = doc_id.replace("/", "_").replace(".", "_")

            documents.append({
                "id":         doc_id,
                "content":    ec.content,
                "embedding":  ec.embedding,
                "language":   str(meta.get("language", "")),
                "chunk_type": str(meta.get("chunk_type", "")),
                "name":       str(meta.get("name", "")),
                "source":     str(meta.get("source", "")),
                "start_line": int(meta.get("start_line", 0) or 0),
                "end_line":   int(meta.get("end_line", 0) or 0),
            })

        # Upload in batches of 1000
        batch_size = 1000
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            result = self.search_client.upload_documents(documents=batch)
            failed = [r for r in result if not r.succeeded]
            if failed:
                print(f"  Warning: {len(failed)} documents failed to upload.")

        print(f"  [AzureSearchStore] Uploaded {len(documents)} documents to '{self.index_name}'.")

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        query_text: str = "*",
        where: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid search: vector ANN + BM25 keyword, with optional semantic reranking.

        Parameters
        ----------
        query_embedding : list[float]
            Dense query vector for ANN search.
        top_k : int
            Number of results to return.
        query_text : str
            Text query for BM25 keyword search.  Use "*" for pure vector search.
        where : dict | None
            OData $filter string or a simple dict of field=value pairs.
            Examples:
              {"language": "sas"}
              {"chunk_type": "macro"}
              "language eq 'sas' and chunk_type eq 'macro'"

        Returns
        -------
        list[dict]
            List of {"content", "metadata", "distance", "score"} dicts,
            compatible with the VectorStore.search() return format.
        """
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector     = query_embedding,
            k_nearest_neighbors = top_k * 2,   # fetch more for reranker
            fields     = VECTOR_FIELD,
        )

        filter_str = _build_odata_filter(where) if where else None

        search_kwargs: dict = dict(
            search_text   = query_text,
            vector_queries= [vector_query],
            filter        = filter_str,
            top           = top_k,
            select        = ["id", "content", "language", "chunk_type",
                             "name", "source", "start_line", "end_line"],
        )

        if self.semantic_reranking:
            search_kwargs["query_type"]            = "semantic"
            search_kwargs["semantic_configuration_name"] = "etl-semantic"
            search_kwargs["query_caption"]         = "extractive"

        results = self.search_client.search(**search_kwargs)

        hits: list[dict] = []
        for r in results:
            # @search.score is BM25+vector combined; @search.reranker_score is semantic
            score    = r.get("@search.reranker_score") or r.get("@search.score") or 0.0
            distance = max(0.0, 1.0 - min(float(score) / 4.0, 1.0))   # normalise to [0,1]

            hits.append({
                "content":  r["content"],
                "metadata": {
                    "language":   r.get("language", ""),
                    "chunk_type": r.get("chunk_type", ""),
                    "name":       r.get("name", ""),
                    "source":     r.get("source", ""),
                    "start_line": r.get("start_line", 0),
                    "end_line":   r.get("end_line", 0),
                },
                "distance": distance,
                "score":    score,
            })
        return hits

    # ── Stats ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of documents in the index."""
        return self.search_client.get_document_count()

    def get_all(self, limit: int = 10_000) -> dict:
        """Return all documents (for BM25 corpus loading or export)."""
        results = self.search_client.search(
            search_text = "*",
            top         = limit,
            select      = ["id", "content", "language", "chunk_type",
                           "name", "source", "start_line", "end_line"],
        )
        ids, docs, metas = [], [], []
        for r in results:
            ids.append(r["id"])
            docs.append(r["content"])
            metas.append({
                "language":   r.get("language", ""),
                "chunk_type": r.get("chunk_type", ""),
                "name":       r.get("name", ""),
                "source":     r.get("source", ""),
                "start_line": r.get("start_line", 0),
                "end_line":   r.get("end_line", 0),
            })
        return {"ids": ids, "documents": docs, "metadatas": metas}


# ---------------------------------------------------------------------------
# OData filter builder
# ---------------------------------------------------------------------------

def _build_odata_filter(where: dict | str) -> str:
    """
    Convert a simple dict or pre-built OData string into an OData $filter.

    Examples:
        {"language": "sas"}
            → "language eq 'sas'"
        {"language": "sas", "chunk_type": "macro"}
            → "language eq 'sas' and chunk_type eq 'macro'"
        "language eq 'python'"
            → "language eq 'python'"    (passed through unchanged)
    """
    if isinstance(where, str):
        return where
    clauses: list[str] = []
    for field, value in where.items():
        if isinstance(value, list):
            # IN  →  (field eq 'a' or field eq 'b')
            inner = " or ".join(f"{field} eq '{v}'" for v in value)
            clauses.append(f"({inner})")
        else:
            clauses.append(f"{field} eq '{value}'")
    return " and ".join(clauses)
