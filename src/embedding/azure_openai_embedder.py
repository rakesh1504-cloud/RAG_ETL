"""
Azure OpenAI Embedder
---------------------
Wraps the Azure OpenAI Embeddings endpoint to produce dense vectors
for use with Azure AI Search or ChromaDB.

Why Azure OpenAI instead of direct OpenAI?
  • Data stays inside your Azure tenant — no data sent to openai.com
  • Uses Azure RBAC + managed identity — no API key in code
  • Same models (text-embedding-3-small, gpt-4o) with Azure SLA
  • Supports Azure Private Endpoints for full network isolation

Environment variables
  AZURE_OPENAI_ENDPOINT       e.g. https://my-aoai.openai.azure.com/
  AZURE_OPENAI_API_KEY        optional; use DefaultAzureCredential if unset
  AZURE_OPENAI_API_VERSION    e.g. 2024-02-01  (default used if unset)
  AZURE_OPENAI_EMBED_DEPLOY   deployment name for the embedding model
                              e.g. text-embedding-3-small
"""

from __future__ import annotations

import os
from typing import Sequence


_DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIEmbedder:
    """
    Embed text using Azure OpenAI text-embedding-3-small (or any Azure-hosted
    embedding deployment).

    Parameters
    ----------
    endpoint : str | None
        Azure OpenAI resource endpoint.
        Defaults to AZURE_OPENAI_ENDPOINT env var.
    api_key : str | None
        Azure OpenAI API key.  Defaults to AZURE_OPENAI_API_KEY env var.
        Leave None to use DefaultAzureCredential (managed identity / az login).
    deployment : str
        Name of the embedding model deployment in Azure OpenAI Studio.
        Defaults to AZURE_OPENAI_EMBED_DEPLOY env var or "text-embedding-3-small".
    api_version : str
        Azure OpenAI API version string.  Default: 2024-02-01.
    dimensions : int | None
        Optional output dimensionality (text-embedding-3-* only).
    batch_size : int
        Texts per API request.  Azure limit: 2048.
    """

    def __init__(
        self,
        endpoint: str | None    = None,
        api_key: str | None     = None,
        deployment: str         = "",
        api_version: str        = _DEFAULT_API_VERSION,
        dimensions: int | None  = None,
        batch_size: int         = 512,
    ):
        self.endpoint    = endpoint    or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_key     = api_key     or os.environ.get("AZURE_OPENAI_API_KEY")
        self.deployment  = deployment  or os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
                                                         "text-embedding-3-small")
        self.api_version = api_version
        self.dimensions  = dimensions
        self.batch_size  = batch_size

        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. "
                "Set AZURE_OPENAI_ENDPOINT or pass endpoint= parameter."
            )

        self._client = None

    # ── Client init ───────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            try:
                import openai as _openai
            except ImportError:
                raise ImportError("pip install openai")

            if self.api_key:
                from azure.core.credentials import AzureKeyCredential
                self._client = _openai.AzureOpenAI(
                    azure_endpoint = self.endpoint,
                    api_key        = self.api_key,
                    api_version    = self.api_version,
                )
            else:
                # Managed identity / az login  →  get token via DefaultAzureCredential
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                self._client = _openai.AzureOpenAI(
                    azure_endpoint    = self.endpoint,
                    azure_ad_token_provider = token_provider,
                    api_version       = self.api_version,
                )
        return self._client

    # ── Dimension info ────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self.dimensions or 1536   # text-embedding-3-small default

    # ── Core encode ───────────────────────────────────────────────────────

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Encode a list of texts into embeddings (batched).
        Returns a list of float vectors in the same order as input.
        """
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict = {
                "input":      batch,
                "model":      self.deployment,  # Azure uses deployment name as model
            }
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)
            batch_embeddings = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.encode([query])[0]

    def embed_texts_bulk(
        self,
        texts: list[str],
        metadata_list: list[dict],
    ) -> list[dict]:
        """
        Embed texts with their metadata.
        Returns list of {"content", "metadata", "embedding"} dicts
        — same shape as Embedder.embed_texts_bulk() for pipeline compatibility.
        """
        embeddings = self.encode(texts)
        return [
            {"content": t, "metadata": m, "embedding": e}
            for t, m, e in zip(texts, metadata_list, embeddings)
        ]
