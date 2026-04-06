"""
Integration test: full RAG pipeline (ingestion → embedding → retrieval → generation).
Requires ANTHROPIC_API_KEY to be set.
"""

import os
import pytest
import tempfile

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture
def pipeline(tmp_path):
    from src.llm.rag_pipeline import RAGPipeline
    return RAGPipeline(
        chunk_size=200,
        overlap=20,
        vector_store_dir=str(tmp_path / "vector_store"),
    )


def test_ingest_and_query(pipeline, tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text(
        "Our refund policy: customers may return products within 30 days for a full refund. "
        "Items must be unused and in original packaging.",
        encoding="utf-8",
    )
    count = pipeline.ingest_file(str(doc))
    assert count >= 1

    answer = pipeline.query("What is the refund window?")
    assert "30" in answer or "day" in answer.lower()
