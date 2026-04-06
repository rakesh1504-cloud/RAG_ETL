# ETL Pipeline Guide

## Overview

The ETL pipeline extracts data from raw sources, cleans and enriches it,
validates quality, then loads it to flat files, databases, and/or the RAG
vector store for LLM-powered queries.

---

## Python ETL

### Quick Start

```python
from etl.python.pipeline import run_csv_to_rag

# Ingest a CSV and push text columns straight into the RAG store
run_csv_to_rag(
    input_csv="data/raw/customers.csv",
    text_columns=["first_name", "last_name", "email"],
    output_csv="data/processed/customers_clean.csv",
)
```

### Full Pipeline Configuration

```python
from etl.python.pipeline import ETLPipeline, ETLConfig

config = ETLConfig(
    # ── Source ──────────────────────────────────────────
    source_type  = "csv",
    source_path  = "data/samples/customers.csv",

    # ── Transform ───────────────────────────────────────
    date_columns    = ["signup_date"],
    numeric_columns = ["age"],

    # ── Validate ────────────────────────────────────────
    not_null_columns  = ["customer_id", "email"],
    unique_columns    = ["customer_id"],
    fail_on_violations = False,   # log warnings, don't abort

    # ── Load – File ──────────────────────────────────────
    output_csv     = "data/processed/customers_clean.csv",
    output_parquet = "data/processed/customers_clean.parquet",

    # ── Load – DB ────────────────────────────────────────
    output_db_path    = "data/processed/etl.sqlite",
    output_table_name = "customers",
    db_if_exists      = "replace",

    # ── Load – RAG ───────────────────────────────────────
    load_to_rag          = True,
    rag_text_columns     = ["first_name", "last_name", "email"],
    rag_metadata_columns = ["customer_id", "status"],
)

metrics = ETLPipeline(config).run()
print(metrics)
```

### Component Reference

#### Extract (`etl/python/extract.py`)

| Class | Method | Description |
|---|---|---|
| `CSVExtractor` | `extract_file(path)` | Load single CSV |
| `CSVExtractor` | `extract_directory(dir, pattern)` | Load all CSVs in a folder |
| `CSVExtractor` | `extract_stream(path, chunk_size)` | Chunk-iterate large files |
| `DatabaseExtractor` | `extract(query)` | SQLAlchemy SQL query |
| `DatabaseExtractor` | `extract_sqlite(db, query)` | Pure sqlite3 query |
| `APIExtractor` | `extract_endpoint(ep, params)` | Single GET endpoint |
| `APIExtractor` | `extract_paginated(ep, ...)` | Walk paginated API |
| `JSONExtractor` | `extract_json(path)` | Nested JSON file |
| `JSONExtractor` | `extract_jsonl(path)` | Line-delimited JSON |

#### Transform (`etl/python/transform.py`)

| Class | Description |
|---|---|
| `DataCleaner` | Dedup, null removal, whitespace strip, column name standardisation |
| `TypeCaster` | Parse dates, coerce numerics, cast booleans |
| `FeatureEngineer` | Date parts, text lengths, binning, custom `apply()` functions |
| `DataValidator` | Rule-based quality checks with violation report |

Custom transform functions can be injected via `ETLConfig.custom_transforms`:

```python
def flag_vip(df):
    df["is_vip"] = df["total_spend"] > 500
    return df

config.custom_transforms = [flag_vip]
```

#### Load (`etl/python/load.py`)

| Class | Method | Description |
|---|---|---|
| `FileLoader` | `to_csv` | Write CSV |
| `FileLoader` | `to_parquet` | Write Parquet (snappy by default) |
| `FileLoader` | `to_json` | Write JSON |
| `FileLoader` | `to_excel` | Write Excel |
| `DatabaseLoader` | `to_sqlite` | Write to SQLite |
| `DatabaseLoader` | `to_sqlalchemy` | Write to any SQLAlchemy DB |
| `DatabaseLoader` | `upsert_sqlite` | Insert-or-replace on primary key |
| `RAGLoader` | `load_dataframe` | Chunk + embed + push to ChromaDB |

---

## SAS ETL

### Running the Full Pipeline

```sas
/* From a SAS session or batch job */
sas /home/user/RAG_ETL/etl/sas/04_master_etl.sas
```

Or run stages individually:

```sas
%include "/home/user/RAG_ETL/etl/sas/01_extract.sas";
%include "/home/user/RAG_ETL/etl/sas/02_transform.sas";
%include "/home/user/RAG_ETL/etl/sas/03_load.sas";
```

### Configuring Source Paths

At the top of each program, update the `%let` macro variables:

```sas
%let raw_data  = /your/path/to/data/raw;
%let proc_data = /your/path/to/data/processed;
```

### Macro Utilities (`etl/sas/macros/etl_macros.sas`)

| Macro | Usage |
|---|---|
| `%log_msg(msg)` | Timestamped log line |
| `%ds_exists(ds)` | Returns 1/0 — use in `%if` |
| `%row_count(ds, mvar)` | Store row count in macro var |
| `%drop_if_exists(ds)` | Safe drop without WARNING |
| `%assert_no_nulls(ds, col)` | Abort if nulls found |
| `%standardise_dates(ds, cols)` | Parse char dates in-place |
| `%trim_all_char(ds)` | Strip whitespace from all char cols |
| `%export_csv_ts(ds, dir, base)` | Export with datestamp filename |
| `%freq_check(ds, cols)` | Quick frequency to log |

### Adding a New Source Table

1. Add a `PROC IMPORT` block to `01_extract.sas`
2. Add a `DATA` step or `PROC SQL` block to `02_transform.sas`
3. Add `%export_csv` and `%audit_log` calls to `03_load.sas`

---

## Sample Data

The `data/samples/` directory contains three files for local testing:

| File | Rows | Description |
|---|---|---|
| `customers.csv` | 10 | Customer master with demographics |
| `transactions.csv` | 15 | Transaction ledger with amounts and status |
| `product_lookup.txt` | 5 | Pipe-delimited product reference |

Run the Python pipeline against them:

```bash
python scripts/ingest.py --path data/samples/customers.csv
python scripts/query.py --question "Who are the active customers?"
```

---

## Validation Rules

| Rule | Python | SAS |
|---|---|---|
| No nulls in key columns | `DataValidator.expect_no_nulls` | `%assert_no_nulls` |
| Unique primary key | `DataValidator.expect_unique` | `PROC SORT nodupkey` |
| Allowed status values | `DataValidator.expect_values_in_set` | `SELECT / WHEN` in DATA step |
| Numeric range | `DataValidator.expect_range` | Conditional `DELETE` in DATA step |

---

## Output Files

| Path | Format | Contents |
|---|---|---|
| `data/processed/customers_clean.csv` | CSV | Cleaned customer records |
| `data/processed/transactions_enriched.csv` | CSV | Transactions joined to products |
| `data/processed/customer_summary.csv` | CSV | Aggregated customer KPIs |
| `data/processed/etl_report.pdf` | PDF | SAS ODS summary report |
| `data/processed/audit_log.txt` | TXT | Row-count audit trail |
| `data/vector_store/` | ChromaDB | Embedded chunks for RAG |
