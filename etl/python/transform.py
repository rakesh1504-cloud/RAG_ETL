"""
ETL Transform Layer
-------------------
Cleaning, standardisation, enrichment, and validation transformations
applied to raw DataFrames before they are loaded downstream.
"""

import re
from datetime import datetime
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Data Cleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """Basic data quality and cleaning operations."""

    def drop_duplicates(
        self,
        df: pd.DataFrame,
        subset: list[str] | None = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        print(f"drop_duplicates: removed {before - len(df)} rows → {len(df)} remain")
        return df

    def drop_null_rows(self, df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=subset)
        print(f"drop_null_rows: removed {before - len(df)} rows → {len(df)} remain")
        return df

    def fill_nulls(self, df: pd.DataFrame, fill_map: dict) -> pd.DataFrame:
        """Fill nulls per column: {'col_name': fill_value}."""
        return df.fillna(fill_map)

    def strip_whitespace(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or df.select_dtypes(include="object").columns.tolist()
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase + replace spaces/hyphens with underscores."""
        df = df.copy()
        df.columns = [
            re.sub(r"[\s\-]+", "_", col.strip()).lower()
            for col in df.columns
        ]
        return df

    def remove_special_characters(
        self, df: pd.DataFrame, columns: list[str], pattern: str = r"[^a-zA-Z0-9\s\.,\-]"
    ) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: re.sub(pattern, "", x)
                )
        return df


# ---------------------------------------------------------------------------
# Type Caster
# ---------------------------------------------------------------------------

class TypeCaster:
    """Coerce column types with error handling."""

    def cast_dates(
        self,
        df: pd.DataFrame,
        columns: list[str],
        date_format: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
        return df

    def cast_numeric(
        self,
        df: pd.DataFrame,
        columns: list[str],
        downcast: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if downcast:
                    df[col] = pd.to_numeric(df[col], downcast=downcast)
        return df

    def cast_boolean(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        truthy = {"true", "yes", "1", "y", "t"}
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(
                    lambda x: True if x in truthy else (False if x != "nan" else None)
                )
        return df


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Derive new columns from existing data."""

    def add_date_parts(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df = df.copy()
        col = pd.to_datetime(df[date_column], errors="coerce")
        prefix = date_column
        df[f"{prefix}_year"] = col.dt.year
        df[f"{prefix}_month"] = col.dt.month
        df[f"{prefix}_day"] = col.dt.day
        df[f"{prefix}_dayofweek"] = col.dt.day_name()
        df[f"{prefix}_quarter"] = col.dt.quarter
        return df

    def add_text_length(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df = df.copy()
        df[f"{text_column}_length"] = df[text_column].astype(str).str.len()
        df[f"{text_column}_word_count"] = df[text_column].astype(str).str.split().str.len()
        return df

    def bin_numeric(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int | list,
        labels: list | None = None,
        new_column: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        target = new_column or f"{column}_bin"
        df[target] = pd.cut(df[column], bins=bins, labels=labels)
        return df

    def apply_custom(
        self,
        df: pd.DataFrame,
        column: str,
        func: Callable,
        new_column: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        target = new_column or column
        df[target] = df[column].apply(func)
        return df

    def one_hot_encode(self, df: pd.DataFrame, columns: list[str], drop_first: bool = False) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=drop_first)


# ---------------------------------------------------------------------------
# Data Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """Assert data quality rules; collect violations instead of crashing."""

    def __init__(self):
        self.violations: list[dict] = []

    def expect_no_nulls(self, df: pd.DataFrame, columns: list[str]) -> "DataValidator":
        for col in columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                self.violations.append({
                    "rule": "no_nulls",
                    "column": col,
                    "detail": f"{null_count} null values found",
                })
        return self

    def expect_unique(self, df: pd.DataFrame, columns: list[str]) -> "DataValidator":
        for col in columns:
            dup_count = df[col].duplicated().sum()
            if dup_count > 0:
                self.violations.append({
                    "rule": "unique",
                    "column": col,
                    "detail": f"{dup_count} duplicate values found",
                })
        return self

    def expect_values_in_set(
        self, df: pd.DataFrame, column: str, valid_set: set
    ) -> "DataValidator":
        invalid = df[~df[column].isin(valid_set)][column].unique()
        if len(invalid) > 0:
            self.violations.append({
                "rule": "values_in_set",
                "column": column,
                "detail": f"Invalid values: {invalid[:10].tolist()}",
            })
        return self

    def expect_range(
        self, df: pd.DataFrame, column: str, min_val=None, max_val=None
    ) -> "DataValidator":
        if min_val is not None and (df[column] < min_val).any():
            self.violations.append({
                "rule": "range_min",
                "column": column,
                "detail": f"Values below {min_val} found",
            })
        if max_val is not None and (df[column] > max_val).any():
            self.violations.append({
                "rule": "range_max",
                "column": column,
                "detail": f"Values above {max_val} found",
            })
        return self

    def report(self) -> dict:
        return {
            "passed": len(self.violations) == 0,
            "violation_count": len(self.violations),
            "violations": self.violations,
        }
