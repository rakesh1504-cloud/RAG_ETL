# RAG Pipeline Guide

## What is RAG?

**Retrieval-Augmented Generation** grounds an LLM's answer in real documents
rather than relying solely on training data.  The three steps are:

1. **Index** — chunk documents → embed → store in a vector DB
2. **Retrieve** — embed the query → find the most similar chunks
3. **Generate** — pass the retrieved chunks as context to Claude → get a grounded answer

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY
```

### 2 – Ingest documents

```bash
# Single file
python scripts/ingest.py --path data/samples/customers.csv

# Entire directory
python scripts/ingest.py --path data/raw/ --dir
```

### 3 – Ask questions

```bash
# One-shot
python scripts/query.py --question "Which customers signed up in 2023?"

# Interactive REPL
python scripts/query.py
```

### 4 – Use the REST API

```bash
uvicorn src.api.app:app --reload

# Ingest a file via API
curl -X POST http://localhost:8000/ingest/file \
     -F "file=@data/samples/customers.csv"

# Query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the total spend for Alice Johnson?"}'
```

---

## Python API

### RAGPipeline

```python
from src.llm.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    chunk_size       = 512,    # characters per chunk
    overlap          = 64,     # overlap between adjacent chunks
    embedding_model  = "all-MiniLM-L6-v2",
    vector_store_dir = "data/vector_store",
    top_k            = 5,      # chunks to retrieve per query
)

# Ingest
pipeline.ingest_file("data/samples/customers.csv")
pipeline.ingest_directory("data/raw/")

# Query (streams answer to stdout, returns full string)
answer = pipeline.query("Which customers are inactive?")

# Metadata
print(pipeline.stats())   # {"total_chunks": 42}
```

### ClaudeRAGClient

```python
from src.llm.claude_client import ClaudeRAGClient
from src.retrieval.vector_store import VectorStore
from src.embedding.embedder import Embedder

embedder = Embedder()
store    = VectorStore()
client   = ClaudeRAGClient()

query     = "What products do we offer?"
embedding = embedder.embed_query(query)
chunks    = store.search(embedding, top_k=5)
answer    = client.answer(query, chunks)
```

---

## Chunking Strategy

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 512 | Larger = more context per chunk; fewer chunks |
| `overlap` | 64 | Larger = better continuity across boundaries |

**Guidelines:**
- Short factual documents (FAQs, product specs): `chunk_size=256, overlap=32`
- Long prose (reports, manuals): `chunk_size=512, overlap=64`
- Code files: `chunk_size=1024, overlap=128` (preserve function scope)

---

## Embedding Model

Default: `all-MiniLM-L6-v2` (22M params, 384-dim, fast, good quality)

To swap the model:

```python
from src.embedding.embedder import Embedder

embedder = Embedder(model_name="all-mpnet-base-v2")   # higher quality, slower
# or
embedder = Embedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")  # multilingual
```

---

## Claude Model Configuration

The default model is `claude-opus-4-6` with:
- `thinking: {type: "adaptive"}` — Claude decides when to reason deeply
- Streaming enabled for all responses (prevents HTTP timeouts on long outputs)

To change in `src/llm/claude_client.py`:

```python
client = ClaudeRAGClient(model="claude-sonnet-4-6")   # faster / cheaper
```

---

## Vector Store

ChromaDB is used for local persistent storage.

```python
from src.retrieval.vector_store import VectorStore

store = VectorStore(
    collection_name = "my_collection",
    persist_dir     = "data/vector_store",
)

# Add pre-embedded chunks
store.add_chunks(embedded_chunks)

# Similarity search
hits = store.search(query_embedding, top_k=5)
# hits: [{"content": "...", "metadata": {...}, "distance": 0.12}, ...]

print(store.count())   # total chunks indexed
```

To swap to a different vector store, replace `src/retrieval/vector_store.py`
while keeping the same `add_chunks(embedded_chunks)` / `search(embedding, top_k)` interface.

---

## REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + chunk count |
| `GET` | `/stats` | Vector store stats |
| `POST` | `/ingest/file` | Upload and index a document |
| `POST` | `/query` | Ask a question, get an answer |

### Example: `/query` request & response

```json
POST /query
{
  "question": "What is the refund policy?"
}

200 OK
{
  "question": "What is the refund policy?",
  "answer": "Based on the provided context, customers may return products within 30 days...",
  "chunks_in_store": 47
}
```

---

## Integrating ETL Output with RAG

After running the Python ETL pipeline, text columns are automatically pushed
to the vector store when `load_to_rag=True`:

```python
from etl.python.pipeline import ETLPipeline, ETLConfig

config = ETLConfig(
    source_type          = "csv",
    source_path          = "data/samples/customers.csv",
    load_to_rag          = True,
    rag_text_columns     = ["first_name", "last_name", "email"],
    rag_metadata_columns = ["customer_id", "status", "age_group"],
)
ETLPipeline(config).run()

# Now query the ingested data
from src.llm.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.query("Find all active customers who signed up in 2023")
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `AuthenticationError` | Missing API key | Set `ANTHROPIC_API_KEY` in `.env` |
| Empty answers | No chunks indexed | Run `ingest_file()` first |
| Slow first query | Model downloading | `all-MiniLM-L6-v2` downloads ~90 MB on first run |
| `ImportError: chromadb` | Missing package | `pip install chromadb` |
| Answers not grounded | `top_k` too low | Increase `top_k` (try 8–10) |
