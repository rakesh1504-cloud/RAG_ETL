"""
ETL Load Layer
--------------
Writes transformed DataFrames to CSV, Parquet, SQLite/RDBMS,
JSON, and the RAG vector store.
"""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# File Loaders
# ---------------------------------------------------------------------------

class FileLoader:
    """Write DataFrames to flat files."""

    def to_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        index: bool = False,
        **kwargs,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=index, **kwargs)
        print(f"Saved {len(df)} rows → {output_path}")
        return output_path

    def to_parquet(
        self,
        df: pd.DataFrame,
        output_path: str,
        compression: str = "snappy",
        **kwargs,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression=compression, index=False, **kwargs)
        print(f"Saved {len(df)} rows → {output_path} (parquet/{compression})")
        return output_path

    def to_json(
        self,
        df: pd.DataFrame,
        output_path: str,
        orient: str = "records",
        indent: int = 2,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient=orient, indent=indent, date_format="iso")
        print(f"Saved {len(df)} rows → {output_path} (json/{orient})")
        return output_path

    def to_excel(self, df: pd.DataFrame, output_path: str, sheet_name: str = "Sheet1") -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, sheet_name=sheet_name, index=False)
        print(f"Saved {len(df)} rows → {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Database Loader
# ---------------------------------------------------------------------------

class DatabaseLoader:
    """Load DataFrames into relational databases."""

    def to_sqlite(
        self,
        df: pd.DataFrame,
        db_path: str,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"Loaded {len(df)} rows → SQLite:{db_path}/{table_name}")
        return len(df)

    def to_sqlalchemy(
        self,
        df: pd.DataFrame,
        connection_string: str,
        table_name: str,
        schema: str | None = None,
        if_exists: str = "append",
        chunksize: int = 1_000,
    ) -> int:
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("Install sqlalchemy: pip install sqlalchemy")

        engine = create_engine(connection_string)
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )
        print(f"Loaded {len(df)} rows → {connection_string}/{table_name}")
        return len(df)

    def upsert_sqlite(
        self,
        df: pd.DataFrame,
        db_path: str,
        table_name: str,
        primary_keys: list[str],
    ) -> int:
        """Insert or replace rows based on primary key(s)."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            df.to_sql(f"_staging_{table_name}", conn, if_exists="replace", index=False)
            pk_clause = " AND ".join(
                f"{table_name}.{k} = _staging_{table_name}.{k}" for k in primary_keys
            )
            cols = ", ".join(df.columns)
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} ({cols})
                SELECT {cols} FROM _staging_{table_name}
                """
            )
            conn.execute(f"DROP TABLE IF EXISTS _staging_{table_name}")
        print(f"Upserted {len(df)} rows → {db_path}/{table_name}")
        return len(df)


# ---------------------------------------------------------------------------
# RAG Vector Store Loader
# ---------------------------------------------------------------------------

class RAGLoader:
    """
    Load structured/text data into the RAG vector store so it
    can be queried via Claude.  Each row becomes a Document.
    """

    def __init__(
        self,
        text_columns: list[str],
        metadata_columns: list[str] | None = None,
        chunk_size: int = 512,
        overlap: int = 64,
        vector_store_dir: str = "data/vector_store",
    ):
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns or []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store_dir = vector_store_dir

    def load_dataframe(self, df: pd.DataFrame, source_tag: str = "dataframe") -> int:
        from src.ingestion.document_loader import Document
        from src.processing.chunker import TextChunker
        from src.embedding.embedder import Embedder
        from src.retrieval.vector_store import VectorStore

        chunker = TextChunker(chunk_size=self.chunk_size, overlap=self.overlap)
        embedder = Embedder()
        store = VectorStore(persist_dir=self.vector_store_dir)

        all_chunks = []
        for _, row in df.iterrows():
            text = " | ".join(str(row[col]) for col in self.text_columns if col in row)
            metadata = {"source": source_tag}
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = str(row[col])
            doc = Document(content=text, metadata=metadata)
            all_chunks.extend(chunker.chunk_document(doc))

        if all_chunks:
            embedded = embedder.embed_chunks(all_chunks)
            store.add_chunks(embedded)

        print(f"RAGLoader: {len(df)} rows → {len(all_chunks)} chunks stored")
        return len(all_chunks)
