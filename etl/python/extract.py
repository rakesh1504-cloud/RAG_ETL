"""
ETL Extract Layer
-----------------
Pulls raw data from CSV files, relational databases, REST APIs,
and S3-compatible object stores into a unified pandas DataFrame.
"""

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# CSV Extractor
# ---------------------------------------------------------------------------

class CSVExtractor:
    """Extract data from one or more CSV files."""

    def extract_file(self, file_path: str, **read_kwargs) -> pd.DataFrame:
        """Load a single CSV file into a DataFrame."""
        df = pd.read_csv(file_path, **read_kwargs)
        df["_source_file"] = Path(file_path).name
        return df

    def extract_directory(
        self,
        dir_path: str,
        pattern: str = "*.csv",
        **read_kwargs,
    ) -> pd.DataFrame:
        """Load all CSV files matching *pattern* from a directory."""
        frames = []
        for path in Path(dir_path).glob(pattern):
            frames.append(self.extract_file(str(path), **read_kwargs))
        if not frames:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {dir_path}")
        return pd.concat(frames, ignore_index=True)

    def extract_stream(
        self,
        file_path: str,
        chunk_size: int = 10_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield chunks for large CSV files to avoid memory issues."""
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            yield chunk


# ---------------------------------------------------------------------------
# Database Extractor (SQLite / any SQLAlchemy-compatible DB)
# ---------------------------------------------------------------------------

class DatabaseExtractor:
    """Extract data from a relational database via SQL."""

    def __init__(self, connection_string: str):
        """
        connection_string examples:
          - SQLite  : 'sqlite:///data/raw/mydb.sqlite'
          - Postgres: 'postgresql://user:pass@host:5432/dbname'
          - MySQL   : 'mysql+pymysql://user:pass@host/dbname'
        """
        self.connection_string = connection_string

    def extract(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Run *query* and return results as a DataFrame."""
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params or {})
            return df
        except ImportError:
            raise ImportError("Install sqlalchemy: pip install sqlalchemy")

    def extract_table(self, table_name: str, schema: str | None = None) -> pd.DataFrame:
        """Extract an entire table."""
        qualified = f"{schema}.{table_name}" if schema else table_name
        return self.extract(f"SELECT * FROM {qualified}")

    # Convenience: pure sqlite3 (no SQLAlchemy needed)
    def extract_sqlite(self, db_path: str, query: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)


# ---------------------------------------------------------------------------
# REST API Extractor
# ---------------------------------------------------------------------------

class APIExtractor:
    """Extract data from a JSON REST API with optional pagination."""

    def __init__(self, base_url: str, headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    def extract_endpoint(
        self,
        endpoint: str,
        params: dict | None = None,
        data_key: str | None = None,
    ) -> pd.DataFrame:
        """
        GET *endpoint* and normalise the JSON payload into a DataFrame.
        *data_key* is the top-level JSON key that holds the records list.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(url, headers=self.headers, params=params or {})
        resp.raise_for_status()
        payload = resp.json()
        records = payload[data_key] if data_key else payload
        return pd.json_normalize(records)

    def extract_paginated(
        self,
        endpoint: str,
        page_param: str = "page",
        page_size_param: str = "per_page",
        page_size: int = 100,
        data_key: str | None = None,
        max_pages: int = 50,
    ) -> pd.DataFrame:
        """Walk paginated endpoints until an empty page is returned."""
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        all_records: list[dict] = []

        for page in range(1, max_pages + 1):
            resp = requests.get(
                url,
                headers=self.headers,
                params={page_param: page, page_size_param: page_size},
            )
            resp.raise_for_status()
            payload = resp.json()
            records = payload[data_key] if data_key else payload
            if not records:
                break
            all_records.extend(records)

        return pd.json_normalize(all_records)


# ---------------------------------------------------------------------------
# JSON / JSONL Extractor
# ---------------------------------------------------------------------------

class JSONExtractor:
    def extract_json(self, file_path: str, record_path: str | None = None) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if record_path:
            for key in record_path.split("."):
                data = data[key]
        return pd.json_normalize(data)

    def extract_jsonl(self, file_path: str) -> pd.DataFrame:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
