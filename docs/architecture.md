# Architecture Overview

## System Purpose

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline on top of an **ETL** (Extract, Transform, Load) framework. Raw data from multiple sources is ingested, cleaned, embedded into a vector store, and made queryable through Claude — Anthropic's large language model.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                  │
│  CSV / Excel  │  Relational DB  │  REST API  │  JSON / JSONL        │
└───────┬───────┴────────┬────────┴─────┬──────┴──────────┬───────────┘
        │                │              │                  │
        ▼                ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTRACT LAYER                                    │
│   src/ingestion/document_loader.py  │  etl/python/extract.py        │
│   etl/sas/01_extract.sas                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Raw DataFrames / SAS datasets
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFORM LAYER                                   │
│   etl/python/transform.py           │  etl/sas/02_transform.sas     │
│   - DataCleaner                     │  - DATA step cleaning          │
│   - TypeCaster                      │  - PROC SQL aggregation        │
│   - FeatureEngineer                 │  - Validation macros           │
│   - DataValidator                                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Clean DataFrames / SAS datasets
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐
      │  FILE LOAD   │ │   DB LOAD    │ │   RAG LOAD         │
      │  CSV/Parquet │ │ SQLite/RDBMS │ │ Chunk → Embed      │
      └──────────────┘ └──────────────┘ └────────┬──────────┘
                                                  │
                                                  ▼
                                       ┌─────────────────────┐
                                       │   VECTOR STORE       │
                                       │   (ChromaDB)         │
                                       └──────────┬──────────┘
                                                  │ similarity search
                                                  ▼
                                       ┌─────────────────────┐
                                       │   CLAUDE claude-opus-4-6    │
                                       │   (adaptive thinking │
                                       │    + streaming)      │
                                       └──────────┬──────────┘
                                                  │ grounded answer
                                                  ▼
                                       ┌─────────────────────┐
                                       │   REST API           │
                                       │   (FastAPI)          │
                                       └─────────────────────┘
```

---

## Component Descriptions

### ETL – Python (`etl/python/`)

| Module | Responsibility |
|---|---|
| `extract.py` | Pull data from CSV, DB, APIs, JSON |
| `transform.py` | Clean, type-cast, engineer features, validate |
| `load.py` | Write to CSV, Parquet, SQLite, RAG store |
| `pipeline.py` | Orchestrate Extract → Transform → Validate → Load |

### ETL – SAS (`etl/sas/`)

| Program | Responsibility |
|---|---|
| `01_extract.sas` | `PROC IMPORT`, SQL pass-through, row-count checks |
| `02_transform.sas` | DATA step cleaning, PROC SQL aggregation, validation macros |
| `03_load.sas` | PROC EXPORT, permanent library, PDF report, audit log |
| `04_master_etl.sas` | Master controller with error trapping and timing |
| `macros/etl_macros.sas` | Reusable macro utilities (`%log_msg`, `%assert_no_nulls`, etc.) |

### RAG Pipeline (`src/`)

| Module | Responsibility |
|---|---|
| `ingestion/document_loader.py` | Load documents (PDF, TXT, DOCX, HTML, MD) |
| `processing/chunker.py` | Split documents into overlapping chunks |
| `embedding/embedder.py` | Generate embeddings via sentence-transformers |
| `retrieval/vector_store.py` | Store and search embeddings (ChromaDB) |
| `llm/claude_client.py` | Call Claude API with adaptive thinking + streaming |
| `llm/rag_pipeline.py` | End-to-end RAG orchestrator |
| `api/app.py` | FastAPI REST interface |

---

## Data Flow

```
1. Raw source data (CSV/DB/API)
      ↓
2. Python/SAS ETL cleans and enriches data
      ↓
3. Processed records exported to CSV/Parquet/SQLite
      ↓  (optionally)
4. Text columns chunked + embedded → ChromaDB vector store
      ↓
5. User query → embed query → top-k similarity search
      ↓
6. Retrieved chunks + query sent to Claude claude-opus-4-6
      ↓
7. Grounded answer streamed back to user via API
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude claude-opus-4-6 (`anthropic` SDK) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (local persistent) |
| ETL – Python | pandas, SQLAlchemy, requests |
| ETL – SAS | Base SAS, SAS/ACCESS, ODS |
| API | FastAPI + uvicorn |
| Testing | pytest |
| Data Format | CSV, Parquet, SQLite, SAS7BDAT |
