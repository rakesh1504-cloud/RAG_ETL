# RAG ETL Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** system built on a
production-grade **ETL** framework.  Raw data is ingested via Python or SAS,
cleaned, embedded into a vector store, and made queryable through
**Claude claude-opus-4-6**.

---

## Project Structure

```
RAG_ETL/
├── src/                        # RAG pipeline source
│   ├── ingestion/              # Document loaders (PDF, CSV, DOCX, HTML, MD)
│   ├── processing/             # Text chunker
│   ├── embedding/              # sentence-transformers embedder
│   ├── retrieval/              # ChromaDB vector store
│   ├── llm/                    # Claude client + RAG orchestrator
│   ├── api/                    # FastAPI REST endpoints
│   └── utils/                  # Logging utilities
│
├── etl/
│   ├── python/                 # Python ETL (extract / transform / load / pipeline)
│   └── sas/                    # SAS ETL programs + macro library
│       └── macros/
│
├── data/
│   ├── raw/                    # Drop source files here
│   ├── processed/              # ETL output (CSV, Parquet, SQLite, reports)
│   ├── vector_store/           # ChromaDB persisted index
│   └── samples/                # Sample data for local testing
│
├── config/                     # Application settings
├── scripts/                    # CLI: ingest.py, query.py
├── tests/                      # Unit + integration tests
├── notebooks/                  # Exploration notebooks
└── docs/                       # Documentation
    ├── architecture.md
    ├── etl_guide.md
    ├── rag_guide.md
    └── api_reference.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...

# 3. Ingest sample data
python scripts/ingest.py --path data/samples/customers.csv

# 4. Ask a question
python scripts/query.py --question "Who are the active customers?"

# 5. Start the REST API
uvicorn src.api.app:app --reload
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, data flow, tech stack |
| [ETL Guide](docs/etl_guide.md) | Python and SAS ETL usage and config |
| [RAG Guide](docs/rag_guide.md) | RAG pipeline setup and query examples |
| [API Reference](docs/api_reference.md) | REST API + Python/SAS API reference |

---

## Sample Data

| File | Description |
|---|---|
| `data/samples/customers.csv` | 10 customer records |
| `data/samples/transactions.csv` | 15 transaction records |
| `data/samples/product_lookup.txt` | 5-row pipe-delimited product reference |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude claude-opus-4-6 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| ETL – Python | pandas, SQLAlchemy |
| ETL – SAS | Base SAS, SAS/ACCESS, ODS |
| API | FastAPI + uvicorn |
| Testing | pytest |
