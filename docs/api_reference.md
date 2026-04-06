# API Reference

## REST API (`src/api/app.py`)

Base URL: `http://localhost:8000`

---

### GET /health

Liveness probe.

**Response**
```json
{"status": "ok", "chunks_in_store": 42}
```

---

### GET /stats

Vector store statistics.

**Response**
```json
{"total_chunks": 42}
```

---

### POST /ingest/file

Upload a document and index it in the vector store.

**Request** — multipart/form-data

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Document to ingest (PDF, TXT, DOCX, MD, HTML) |

**Response**
```json
{
  "filename": "policy.pdf",
  "chunks_stored": 18
}
```

**Example**
```bash
curl -X POST http://localhost:8000/ingest/file \
     -F "file=@docs/policy.pdf"
```

---

### POST /query

Ask a natural language question.

**Request** — application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | The question to answer |

**Response**
```json
{
  "question": "What is the return window?",
  "answer": "Customers may return products within 30 days...",
  "chunks_in_store": 42
}
```

**Example**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the return window?"}'
```

---

## Python API

### `RAGPipeline`

```
src/llm/rag_pipeline.py
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `__init__` | `(chunk_size, overlap, embedding_model, vector_store_dir, top_k)` | — | Configure pipeline |
| `ingest_file` | `(file_path: str)` | `int` | Ingest single file; return chunk count |
| `ingest_directory` | `(dir_path, extensions)` | `int` | Ingest directory; return chunk count |
| `query` | `(question: str)` | `str` | Retrieve + generate answer |
| `stats` | `()` | `dict` | `{"total_chunks": N}` |

---

### `ClaudeRAGClient`

```
src/llm/claude_client.py
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `answer` | `(question, retrieved_chunks)` | `str` | Stream answer to stdout; return full string |
| `answer_batch` | `(qa_pairs: list[dict])` | `list[dict]` | Batch answers (non-streaming) |
| `build_context_block` | `(retrieved_chunks)` | `str` | Format chunks into context string |

---

### `ETLPipeline`

```
etl/python/pipeline.py
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `__init__` | `(config: ETLConfig)` | — | Initialise with config |
| `extract` | `()` | `DataFrame` | Pull raw data |
| `transform` | `(df)` | `DataFrame` | Clean and enrich |
| `validate` | `(df)` | `DataFrame` | Quality checks |
| `load` | `(df)` | `None` | Write to all targets |
| `run` | `()` | `dict` | Execute all stages; return metrics |

---

### `ETLConfig` Fields

```
etl/python/pipeline.py
```

| Field | Type | Default | Description |
|---|---|---|---|
| `source_type` | str | `"csv"` | `csv`, `csv_dir`, `db`, `sqlite`, `json`, `jsonl` |
| `source_path` | str | `""` | Path to source file/directory |
| `source_query` | str | `"SELECT * FROM ..."` | SQL for `db`/`sqlite` sources |
| `connection_string` | str | SQLite | SQLAlchemy connection string |
| `date_columns` | list[str] | `[]` | Columns to parse as dates |
| `numeric_columns` | list[str] | `[]` | Columns to coerce to float |
| `boolean_columns` | list[str] | `[]` | Columns to coerce to bool |
| `drop_null_subset` | list[str]\|None | `None` | Drop rows where these cols are null |
| `custom_transforms` | list[Callable] | `[]` | `(df) -> df` transform functions |
| `not_null_columns` | list[str] | `[]` | Validation: flag nulls |
| `unique_columns` | list[str] | `[]` | Validation: flag duplicates |
| `fail_on_violations` | bool | `False` | Raise on validation failure |
| `output_csv_path` | str | `""` | Write cleaned CSV here |
| `output_parquet_path` | str | `""` | Write Parquet here |
| `output_db_path` | str | `""` | SQLite DB path |
| `output_table_name` | str | `"etl_output"` | Target table name |
| `db_if_exists` | str | `"replace"` | `replace`, `append`, `fail` |
| `load_to_rag` | bool | `False` | Push text to vector store |
| `rag_text_columns` | list[str] | `[]` | Columns to embed as RAG documents |
| `rag_metadata_columns` | list[str] | `[]` | Columns to store as metadata |
| `rag_vector_store_dir` | str | `"data/vector_store"` | ChromaDB persist directory |

---

## SAS Macro Reference

```
etl/sas/macros/etl_macros.sas
```

| Macro | Parameters | Description |
|---|---|---|
| `%log_msg` | `msg, level=NOTE` | Timestamped log entry |
| `%ds_exists` | `ds` | Returns 1 if dataset exists, else 0 |
| `%row_count` | `ds, mvar` | Stores row count in macro variable `mvar` |
| `%drop_if_exists` | `ds` | Drops dataset silently if it exists |
| `%assert_no_nulls` | `ds, col` | `%abort cancel` if nulls found |
| `%standardise_dates` | `ds, col_list` | Parse char dates in-place with `anydtdte.` |
| `%trim_all_char` | `ds` | Strip whitespace from all character vars |
| `%export_csv_ts` | `ds, outdir, basename` | Export CSV with `YYYYMMDD` timestamp suffix |
| `%freq_check` | `ds, col_list` | Log frequency table for each column |
