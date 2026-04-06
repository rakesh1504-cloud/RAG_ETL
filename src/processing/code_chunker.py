"""
Code-Aware Chunker
------------------
Splits source files at logical boundaries instead of arbitrary character counts:

  Python  – uses the built-in `ast` module to extract top-level imports,
             functions, classes, and remaining module-level statements.

  SAS     – uses regex to extract %MACRO/%MEND blocks, DATA steps, PROC
             steps, and a file-header block for LIBNAME/OPTIONS/%LET etc.

Each chunk preserves the full text of its logical unit plus rich metadata
(language, chunk_type, name, file, line numbers) so the RAG retriever can
surface not just relevant text but exactly *which* function or procedure it
came from.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    content: str
    language: str        # "python" | "sas"
    chunk_type: str      # function | class | imports | data_step | proc_step | macro | header | misc
    name: str            # function/class/macro name, or filename for header/misc
    file_path: str
    start_line: int
    end_line: int

    @property
    def metadata(self) -> dict:
        return {
            "source":     self.file_path,
            "language":   self.language,
            "chunk_type": self.chunk_type,
            "name":       self.name,
            "start_line": str(self.start_line),
            "end_line":   str(self.end_line),
        }

    def __repr__(self) -> str:
        return (
            f"CodeChunk({self.language}/{self.chunk_type} "
            f"name={self.name!r} lines={self.start_line}-{self.end_line} "
            f"len={len(self.content)})"
        )


# ---------------------------------------------------------------------------
# Python code chunker  (ast-based)
# ---------------------------------------------------------------------------

class PythonCodeChunker:
    """
    Parse a Python source file with the `ast` module and split it into
    logical units:

      - ``imports``   : all import / from-import lines at module level
      - ``function``  : every top-level def / async def (with decorators)
      - ``class``     : every top-level class (with decorators + body)
      - ``misc``      : remaining module-level statements (constants, etc.)

    Large functions/classes that exceed *max_chars* are further split into
    overlapping sub-chunks so nothing exceeds the embedding model's limit.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 150):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk_file(self, file_path: str) -> list[CodeChunk]:
        source = Path(file_path).read_text(encoding="utf-8")
        return self.chunk_source(source, file_path)

    def chunk_source(self, source: str, file_path: str = "<string>") -> list[CodeChunk]:
        lines = source.splitlines(keepends=True)
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as exc:
            # Fall back to plain character chunking if the file won't parse
            return self._fallback_chunks(source, file_path, str(exc))

        chunks: list[CodeChunk] = []

        # Collect the decorators that precede each function/class so we can
        # include them in the chunk (AST nodes store the def line, not the @).
        decorator_starts: dict[int, int] = {}   # node.lineno → decorator start line
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.decorator_list:
                    decorator_starts[node.lineno] = node.decorator_list[0].lineno
                else:
                    decorator_starts[node.lineno] = node.lineno

        # Walk only the top-level statements
        import_lines: list[str] = []
        import_start = import_end = None

        for node in tree.body:
            # ── imports ───────────────────────────────────────────────────
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if import_start is None:
                    import_start = node.lineno
                import_end = node.end_lineno or node.lineno
                import_lines.extend(lines[node.lineno - 1: (node.end_lineno or node.lineno)])
                continue

            # Flush collected imports before the first non-import node
            if import_lines:
                chunks.append(CodeChunk(
                    content    = "".join(import_lines).rstrip(),
                    language   = "python",
                    chunk_type = "imports",
                    name       = Path(file_path).stem + "_imports",
                    file_path  = file_path,
                    start_line = import_start,
                    end_line   = import_end,
                ))
                import_lines = []

            # ── function / async function ──────────────────────────────────
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = decorator_starts.get(node.lineno, node.lineno)
                end   = node.end_lineno or node.lineno
                text  = "".join(lines[start - 1: end]).rstrip()
                chunks.extend(self._split_if_large(
                    text, "python", "function", node.name, file_path, start, end
                ))
                continue

            # ── class ─────────────────────────────────────────────────────
            if isinstance(node, ast.ClassDef):
                start = decorator_starts.get(node.lineno, node.lineno)
                end   = node.end_lineno or node.lineno
                text  = "".join(lines[start - 1: end]).rstrip()
                chunks.extend(self._split_if_large(
                    text, "python", "class", node.name, file_path, start, end
                ))
                continue

            # ── anything else (constants, __all__, module docstring, etc.) ─
            end_ln = node.end_lineno or node.lineno
            text = "".join(lines[node.lineno - 1: end_ln]).rstrip()
            if text.strip():
                chunks.append(CodeChunk(
                    content    = text,
                    language   = "python",
                    chunk_type = "misc",
                    name       = Path(file_path).stem + f"_line{node.lineno}",
                    file_path  = file_path,
                    start_line = node.lineno,
                    end_line   = end_ln,
                ))

        # Flush any trailing imports
        if import_lines:
            chunks.append(CodeChunk(
                content    = "".join(import_lines).rstrip(),
                language   = "python",
                chunk_type = "imports",
                name       = Path(file_path).stem + "_imports",
                file_path  = file_path,
                start_line = import_start,
                end_line   = import_end,
            ))

        return chunks

    # ── Private helpers ─────────────────────────────────────────────────────

    def _split_if_large(
        self,
        text: str,
        language: str,
        chunk_type: str,
        name: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> list[CodeChunk]:
        """Return one chunk for small units, multiple overlapping for large ones."""
        if len(text) <= self.max_chars:
            return [CodeChunk(text, language, chunk_type, name, file_path, start_line, end_line)]

        # Split by character with overlap, preserving line boundaries
        sub_chunks: list[CodeChunk] = []
        text_lines = text.splitlines(keepends=True)
        buf: list[str] = []
        buf_len = 0
        part = 0

        for line in text_lines:
            buf.append(line)
            buf_len += len(line)
            if buf_len >= self.max_chars:
                sub_text = "".join(buf).rstrip()
                sub_chunks.append(CodeChunk(
                    content    = sub_text,
                    language   = language,
                    chunk_type = chunk_type,
                    name       = f"{name}_part{part}",
                    file_path  = file_path,
                    start_line = start_line,
                    end_line   = end_line,
                ))
                part += 1
                # Keep last N chars of overlap
                overlap_text = "".join(buf)[-self.overlap_chars:]
                buf = [overlap_text]
                buf_len = len(overlap_text)

        if buf and "".join(buf).strip():
            sub_chunks.append(CodeChunk(
                content    = "".join(buf).rstrip(),
                language   = language,
                chunk_type = chunk_type,
                name       = f"{name}_part{part}",
                file_path  = file_path,
                start_line = start_line,
                end_line   = end_line,
            ))

        return sub_chunks

    def _fallback_chunks(self, source: str, file_path: str, error: str) -> list[CodeChunk]:
        """Character-based fallback when ast.parse fails."""
        print(f"  Warning: AST parse failed for {file_path}: {error}. Using fallback chunking.")
        chunks = []
        for i in range(0, len(source), self.max_chars - self.overlap_chars):
            text = source[i: i + self.max_chars]
            if text.strip():
                chunks.append(CodeChunk(
                    content    = text,
                    language   = "python",
                    chunk_type = "misc",
                    name       = f"{Path(file_path).stem}_chunk{i}",
                    file_path  = file_path,
                    start_line = 0,
                    end_line   = 0,
                ))
        return chunks


# ---------------------------------------------------------------------------
# SAS code chunker  (regex-based)
# ---------------------------------------------------------------------------

# Patterns compiled once at module level for speed
_SAS_PATTERNS = {
    # %MACRO name(...); ... %MEND name;
    "macro": re.compile(
        r'(%MACRO\s+(\w+)\b.*?%MEND\s*\2?\s*;)',
        re.IGNORECASE | re.DOTALL,
    ),
    # DATA lib.table ...; ... RUN;
    "data_step": re.compile(
        r'(DATA\s+[\w\.\s,]+;.*?RUN\s*;)',
        re.IGNORECASE | re.DOTALL,
    ),
    # PROC name ...; ... RUN; or QUIT;
    "proc_step": re.compile(
        r'(PROC\s+\w+\b.*?(?:RUN|QUIT)\s*;)',
        re.IGNORECASE | re.DOTALL,
    ),
}

# Lines that belong to the "header" (preamble before first DATA/PROC/%MACRO)
_SAS_HEADER_LINE = re.compile(
    r'^\s*(OPTIONS|LIBNAME|FILENAME|%LET|%GLOBAL|%LOCAL|%INCLUDE|/\*)',
    re.IGNORECASE,
)


class SASCodeChunker:
    """
    Regex-based chunker for SAS programs.

    Chunk types produced:
      - ``header``    : LIBNAME / OPTIONS / %LET / %GLOBAL lines before first block
      - ``macro``     : %MACRO ... %MEND block
      - ``data_step`` : DATA ... RUN; step
      - ``proc_step`` : PROC ... RUN; / QUIT; step
      - ``misc``      : anything left over (comments, %include calls, etc.)

    Chunks that still exceed *max_chars* are sub-divided with overlap.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 150):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk_file(self, file_path: str) -> list[CodeChunk]:
        source = Path(file_path).read_text(encoding="utf-8")
        return self.chunk_source(source, file_path)

    def chunk_source(self, source: str, file_path: str = "<string>") -> list[CodeChunk]:
        stem = Path(file_path).stem
        chunks: list[CodeChunk] = []

        # ── 1. Extract the file header (preamble) ──────────────────────────
        header_lines: list[str] = []
        non_header_start = 0
        for i, line in enumerate(source.splitlines(keepends=True)):
            stripped = line.strip()
            if not stripped or stripped.startswith("/*") or _SAS_HEADER_LINE.match(line):
                header_lines.append(line)
                non_header_start = i + 1
            else:
                # Stop collecting header on first substantive non-header line
                # (DATA / PROC / %MACRO all start their own chunks)
                break

        if header_lines:
            hdr_text = "".join(header_lines).strip()
            if hdr_text:
                chunks.append(CodeChunk(
                    content    = hdr_text,
                    language   = "sas",
                    chunk_type = "header",
                    name       = f"{stem}_header",
                    file_path  = file_path,
                    start_line = 1,
                    end_line   = non_header_start,
                ))

        # ── 2. Find all named blocks and their positions ────────────────────
        # Build a list of (start, end, chunk_type, name, text) sorted by start
        found: list[tuple[int, int, str, str, str]] = []

        for ctype, pattern in _SAS_PATTERNS.items():
            for m in pattern.finditer(source):
                block_text = m.group(1)

                # Derive a name from the block text
                if ctype == "macro":
                    name = m.group(2)
                elif ctype == "data_step":
                    # Grab the dataset name(s) after DATA keyword
                    ds_match = re.match(r'DATA\s+([\w\.\,\s]+?)\s*;', block_text, re.IGNORECASE)
                    name = ds_match.group(1).strip().replace(" ", "_") if ds_match else "unknown_dataset"
                else:
                    # PROC: grab procedure name
                    pr_match = re.match(r'PROC\s+(\w+)', block_text, re.IGNORECASE)
                    name = pr_match.group(1).upper() if pr_match else "unknown_proc"

                found.append((m.start(), m.end(), ctype, name, block_text))

        # Sort by start position; resolve overlaps (keep whichever comes first)
        found.sort(key=lambda x: x[0])
        non_overlapping: list[tuple[int, int, str, str, str]] = []
        last_end = 0
        for start, end, ctype, name, text in found:
            if start >= last_end:
                non_overlapping.append((start, end, ctype, name, text))
                last_end = end

        # ── 3. Turn each block into one-or-more CodeChunks ─────────────────
        covered_ranges: list[tuple[int, int]] = []
        for start, end, ctype, name, text in non_overlapping:
            start_line = source[:start].count("\n") + 1
            end_line   = source[:end].count("\n") + 1
            covered_ranges.append((start, end))
            chunks.extend(self._make_chunks(text, "sas", ctype, name, file_path, start_line, end_line))

        # ── 4. Collect uncovered text as "misc" chunks ─────────────────────
        uncovered = self._uncovered_segments(source, covered_ranges)
        for seg_start, seg_end in uncovered:
            text = source[seg_start:seg_end].strip()
            if len(text) < 10:   # skip trivial whitespace / single semicolons
                continue
            start_line = source[:seg_start].count("\n") + 1
            end_line   = source[:seg_end].count("\n") + 1
            chunks.extend(self._make_chunks(text, "sas", "misc", f"{stem}_misc", file_path, start_line, end_line))

        # Sort by start line for readability
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # ── Private helpers ─────────────────────────────────────────────────────

    def _make_chunks(
        self,
        text: str,
        language: str,
        chunk_type: str,
        name: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> list[CodeChunk]:
        """Return one chunk if text fits, else split with overlap."""
        if len(text) <= self.max_chars:
            return [CodeChunk(text, language, chunk_type, name, file_path, start_line, end_line)]

        sub_chunks: list[CodeChunk] = []
        for part, i in enumerate(range(0, len(text), self.max_chars - self.overlap_chars)):
            sub_text = text[i: i + self.max_chars].strip()
            if sub_text:
                sub_chunks.append(CodeChunk(
                    content    = sub_text,
                    language   = language,
                    chunk_type = chunk_type,
                    name       = f"{name}_part{part}",
                    file_path  = file_path,
                    start_line = start_line,
                    end_line   = end_line,
                ))
        return sub_chunks

    @staticmethod
    def _uncovered_segments(
        source: str, covered: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return (start, end) byte ranges in *source* not covered by any block."""
        covered_sorted = sorted(covered)
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in covered_sorted:
            if cursor < start:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(source):
            gaps.append((cursor, len(source)))
        return gaps


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_chunker(file_path: str) -> PythonCodeChunker | SASCodeChunker | None:
    """Return the right chunker for a given file extension, or None if unsupported."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return PythonCodeChunker()
    if suffix == ".sas":
        return SASCodeChunker()
    return None


def chunk_file(file_path: str) -> list[CodeChunk]:
    """Chunk any supported source file, dispatching by extension."""
    chunker = get_chunker(file_path)
    if chunker is None:
        raise ValueError(f"No code chunker registered for extension: {Path(file_path).suffix!r}")
    return chunker.chunk_file(file_path)
