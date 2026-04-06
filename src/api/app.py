"""
FastAPI Application - REST API for the RAG pipeline.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tempfile
import os

from src.llm.rag_pipeline import RAGPipeline

app = FastAPI(
    title="RAG ETL API",
    description="LLM-powered document Q&A using Claude and vector search",
    version="1.0.0",
)

pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_in_store: int


@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_store": pipeline.stats()["total_chunks"]}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        count = pipeline.ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_stored": count}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = pipeline.query(request.question)
    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_in_store=pipeline.stats()["total_chunks"],
    )


@app.get("/stats")
def stats():
    return pipeline.stats()
