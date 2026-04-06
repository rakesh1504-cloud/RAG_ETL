"""Unit tests for the code-aware chunker (Python + SAS)."""

import pytest
from src.processing.code_chunker import (
    PythonCodeChunker,
    SASCodeChunker,
    CodeChunk,
    chunk_file,
    get_chunker,
)


# ---------------------------------------------------------------------------
# Python chunker
# ---------------------------------------------------------------------------

PYTHON_SOURCE = '''\
import os
import sys
from pathlib import Path


CONSTANT = 42


def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b


def greet(name: str) -> str:
    return f"Hello, {name}!"


class Processor:
    """A simple processor."""

    def __init__(self, value: int):
        self.value = value

    def process(self) -> int:
        return self.value * 2
'''


def test_python_chunk_types():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source(PYTHON_SOURCE, "sample.py")
    types = {c.chunk_type for c in chunks}
    assert "imports"  in types
    assert "function" in types
    assert "class"    in types


def test_python_function_names():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source(PYTHON_SOURCE, "sample.py")
    fn_names = {c.name for c in chunks if c.chunk_type == "function"}
    assert fn_names == {"add", "greet"}


def test_python_class_name():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source(PYTHON_SOURCE, "sample.py")
    cls_names = {c.name for c in chunks if c.chunk_type == "class"}
    assert "Processor" in cls_names


def test_python_imports_content():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source(PYTHON_SOURCE, "sample.py")
    imp = next(c for c in chunks if c.chunk_type == "imports")
    assert "import os" in imp.content
    assert "from pathlib" in imp.content


def test_python_metadata_fields():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source(PYTHON_SOURCE, "sample.py")
    for c in chunks:
        meta = c.metadata
        assert "source"     in meta
        assert "language"   in meta
        assert "chunk_type" in meta
        assert "name"       in meta
        assert "start_line" in meta
        assert "end_line"   in meta
        assert meta["language"] == "python"


def test_python_large_function_splits():
    # A function body that exceeds max_chars should be sub-divided
    big_body = "    pass\n" * 300
    source = f"def big_func():\n{big_body}"
    chunker = PythonCodeChunker(max_chars=200, overlap_chars=20)
    chunks = chunker.chunk_source(source, "big.py")
    assert len(chunks) > 1
    assert all(c.name.startswith("big_func") for c in chunks)


def test_python_syntax_error_fallback():
    bad_source = "def broken(\n    pass\n"
    chunker = PythonCodeChunker(max_chars=500)
    chunks = chunker.chunk_source(bad_source, "bad.py")
    assert len(chunks) >= 1   # fallback chunks produced, not an exception


def test_python_empty_file():
    chunker = PythonCodeChunker()
    chunks = chunker.chunk_source("", "empty.py")
    assert chunks == []


# ---------------------------------------------------------------------------
# SAS chunker
# ---------------------------------------------------------------------------

SAS_SOURCE = """\
options symbolgen mlogic mprint;
%let root = /data;
libname PROCLIB "/data/processed";

%macro clean_names(ds);
    data &ds.;
        set &ds.;
        name = strip(name);
    run;
%mend clean_names;

data WORK.customers;
    set PROCLIB.raw_customers;
    age_group = "adult";
run;

proc freq data=WORK.customers;
    tables age_group / nocum;
run;

proc sql;
    create table WORK.summary as
    select age_group, count(*) as n
    from WORK.customers
    group by age_group;
quit;
"""


def test_sas_chunk_types():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source(SAS_SOURCE, "sample.sas")
    types = {c.chunk_type for c in chunks}
    assert "macro"     in types
    assert "data_step" in types
    assert "proc_step" in types


def test_sas_macro_name():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source(SAS_SOURCE, "sample.sas")
    macro_names = {c.name for c in chunks if c.chunk_type == "macro"}
    assert "clean_names" in macro_names


def test_sas_data_step_chunk_contains_run():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source(SAS_SOURCE, "sample.sas")
    ds_chunks = [c for c in chunks if c.chunk_type == "data_step"]
    assert len(ds_chunks) >= 1
    assert any("RUN" in c.content.upper() for c in ds_chunks)


def test_sas_proc_chunk():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source(SAS_SOURCE, "sample.sas")
    proc_names = {c.name for c in chunks if c.chunk_type == "proc_step"}
    assert "FREQ" in proc_names or "SQL" in proc_names


def test_sas_metadata_language():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source(SAS_SOURCE, "sample.sas")
    for c in chunks:
        assert c.metadata["language"] == "sas"


def test_sas_empty_file():
    chunker = SASCodeChunker()
    chunks = chunker.chunk_source("", "empty.sas")
    assert chunks == []


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_get_chunker_python():
    assert isinstance(get_chunker("foo.py"), PythonCodeChunker)

def test_get_chunker_sas():
    assert isinstance(get_chunker("bar.sas"), SASCodeChunker)

def test_get_chunker_unsupported():
    assert get_chunker("file.csv") is None


def test_chunk_file_on_real_py(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("def hello():\n    return 'hi'\n")
    chunks = chunk_file(str(f))
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].name == "hello"


def test_chunk_file_on_real_sas(tmp_path):
    f = tmp_path / "test.sas"
    f.write_text("%macro my_mac; %put hello; %mend my_mac;\n")
    chunks = chunk_file(str(f))
    macro_chunks = [c for c in chunks if c.chunk_type == "macro"]
    assert len(macro_chunks) >= 1
    assert macro_chunks[0].name == "my_mac"
