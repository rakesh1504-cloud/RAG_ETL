"""
ETL Pipeline Orchestrator
--------------------------
Wires Extract → Transform → Load into a single configurable run.
Each stage can be individually enabled/disabled for incremental runs.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from etl.python.extract import CSVExtractor, DatabaseExtractor, JSONExtractor
from etl.python.transform import DataCleaner, TypeCaster, FeatureEngineer, DataValidator
from etl.python.load import FileLoader, DatabaseLoader, RAGLoader


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class ETLConfig:
    # Source
    source_type: str = "csv"          # csv | db | api | json | jsonl
    source_path: str = ""
    source_query: str = "SELECT * FROM source_table"
    connection_string: str = "sqlite:///data/raw/source.sqlite"

    # Transform
    date_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    boolean_columns: list[str] = field(default_factory=list)
    drop_null_subset: list[str] | None = None
    text_columns_to_clean: list[str] | None = None
    custom_transforms: list[Callable[[pd.DataFrame], pd.DataFrame]] = field(default_factory=list)

    # Validate
    not_null_columns: list[str] = field(default_factory=list)
    unique_columns: list[str] = field(default_factory=list)
    fail_on_violations: bool = False

    # Load – File
    output_csv_path: str = ""
    output_parquet_path: str = ""

    # Load – DB
    output_db_path: str = ""
    output_table_name: str = "etl_output"
    db_if_exists: str = "replace"           # replace | append | fail

    # Load – RAG
    load_to_rag: bool = False
    rag_text_columns: list[str] = field(default_factory=list)
    rag_metadata_columns: list[str] = field(default_factory=list)
    rag_vector_store_dir: str = "data/vector_store"


# ---------------------------------------------------------------------------
# ETL Pipeline
# ---------------------------------------------------------------------------

class ETLPipeline:
    """
    Configurable Extract → Transform → Validate → Load pipeline.

    Example
    -------
    >>> config = ETLConfig(
    ...     source_type="csv",
    ...     source_path="data/raw/customers.csv",
    ...     date_columns=["signup_date"],
    ...     numeric_columns=["age", "spend"],
    ...     output_csv_path="data/processed/customers_clean.csv",
    ...     load_to_rag=True,
    ...     rag_text_columns=["name", "email", "notes"],
    ... )
    >>> result = ETLPipeline(config).run()
    """

    def __init__(self, config: ETLConfig):
        self.config = config
        self.metrics: dict = {}

    # --- Stage 1: Extract ---

    def extract(self) -> pd.DataFrame:
        t0 = time.time()
        cfg = self.config
        print(f"\n[EXTRACT] source_type={cfg.source_type}")

        if cfg.source_type == "csv":
            df = CSVExtractor().extract_file(cfg.source_path)
        elif cfg.source_type == "csv_dir":
            df = CSVExtractor().extract_directory(cfg.source_path)
        elif cfg.source_type == "db":
            df = DatabaseExtractor(cfg.connection_string).extract(cfg.source_query)
        elif cfg.source_type == "sqlite":
            db_path, query = cfg.source_path, cfg.source_query
            df = DatabaseExtractor("").extract_sqlite(db_path, query)
        elif cfg.source_type == "json":
            df = JSONExtractor().extract_json(cfg.source_path)
        elif cfg.source_type == "jsonl":
            df = JSONExtractor().extract_jsonl(cfg.source_path)
        else:
            raise ValueError(f"Unknown source_type: {cfg.source_type!r}")

        self.metrics["extract_rows"] = len(df)
        self.metrics["extract_cols"] = len(df.columns)
        self.metrics["extract_time_s"] = round(time.time() - t0, 3)
        print(f"  → {len(df)} rows × {len(df.columns)} cols  ({self.metrics['extract_time_s']}s)")
        return df

    # --- Stage 2: Transform ---

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        cfg = self.config
        print(f"\n[TRANSFORM] {len(df)} rows in")

        cleaner = DataCleaner()
        caster = TypeCaster()
        engineer = FeatureEngineer()

        df = cleaner.standardise_column_names(df)
        df = cleaner.strip_whitespace(df, cfg.text_columns_to_clean)
        df = cleaner.drop_duplicates(df)

        if cfg.drop_null_subset:
            df = cleaner.drop_null_rows(df, cfg.drop_null_subset)

        if cfg.date_columns:
            df = caster.cast_dates(df, cfg.date_columns)

        if cfg.numeric_columns:
            df = caster.cast_numeric(df, cfg.numeric_columns)

        if cfg.boolean_columns:
            df = caster.cast_boolean(df, cfg.boolean_columns)

        # Derive date parts for all date columns
        for col in cfg.date_columns:
            if col in df.columns:
                df = engineer.add_date_parts(df, col)

        # Apply any caller-supplied transform functions
        for fn in cfg.custom_transforms:
            df = fn(df)

        self.metrics["transform_rows"] = len(df)
        self.metrics["transform_time_s"] = round(time.time() - t0, 3)
        print(f"  → {len(df)} rows out  ({self.metrics['transform_time_s']}s)")
        return df

    # --- Stage 3: Validate ---

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        print(f"\n[VALIDATE]")

        validator = (
            DataValidator()
            .expect_no_nulls(df, cfg.not_null_columns)
            .expect_unique(df, cfg.unique_columns)
        )
        report = validator.report()
        self.metrics["validation"] = report

        if report["violation_count"]:
            for v in report["violations"]:
                print(f"  ⚠  {v['rule']} | {v['column']} | {v['detail']}")
            if cfg.fail_on_violations:
                raise ValueError(f"Validation failed: {report['violation_count']} violations")
        else:
            print("  ✓ All validation checks passed")

        return df

    # --- Stage 4: Load ---

    def load(self, df: pd.DataFrame) -> None:
        t0 = time.time()
        cfg = self.config
        print(f"\n[LOAD] {len(df)} rows")

        file_loader = FileLoader()

        if cfg.output_csv_path:
            file_loader.to_csv(df, cfg.output_csv_path)

        if cfg.output_parquet_path:
            file_loader.to_parquet(df, cfg.output_parquet_path)

        if cfg.output_db_path and cfg.output_table_name:
            DatabaseLoader().to_sqlite(
                df, cfg.output_db_path, cfg.output_table_name, cfg.db_if_exists
            )

        if cfg.load_to_rag and cfg.rag_text_columns:
            RAGLoader(
                text_columns=cfg.rag_text_columns,
                metadata_columns=cfg.rag_metadata_columns,
                vector_store_dir=cfg.rag_vector_store_dir,
            ).load_dataframe(df, source_tag=Path(cfg.source_path).stem or "etl_output")

        self.metrics["load_time_s"] = round(time.time() - t0, 3)

    # --- Run All Stages ---

    def run(self) -> dict:
        overall_start = time.time()
        df = self.extract()
        df = self.transform(df)
        df = self.validate(df)
        self.load(df)
        self.metrics["total_time_s"] = round(time.time() - overall_start, 3)
        print(f"\n[DONE] total={self.metrics['total_time_s']}s  metrics={self.metrics}")
        return self.metrics


# ---------------------------------------------------------------------------
# Quick-run helper
# ---------------------------------------------------------------------------

def run_csv_to_rag(
    input_csv: str,
    text_columns: list[str],
    output_csv: str = "",
    vector_store_dir: str = "data/vector_store",
) -> dict:
    """One-liner: read a CSV, clean it, and push text columns into the RAG store."""
    config = ETLConfig(
        source_type="csv",
        source_path=input_csv,
        output_csv_path=output_csv,
        load_to_rag=True,
        rag_text_columns=text_columns,
        rag_vector_store_dir=vector_store_dir,
    )
    return ETLPipeline(config).run()
