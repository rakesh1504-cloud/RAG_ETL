"""
Document Chunker
----------------
Section-aware chunking for Markdown (.md) and YAML (.yaml / .yml) files.

Markdown chunking
    Splits at heading boundaries (# ## ###).  Each heading + its body text
    becomes one chunk.  A preamble chunk is created for text before the first
    heading.  Very long sections are sub-divided at blank-line boundaries.

YAML chunking
    Top-level keys (e.g. ``columns:``, ``quality_rules:``, ``lineage:``) are
    extracted as individual chunks.  Header metadata (table, description, etc.)
    forms its own chunk.  This keeps schema definitions and quality rules
    retrievable independently.

Both chunkers attach rich metadata so the retrieval engine can filter by
document type, source file, and section name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared chunk dataclass (parallel to CodeChunk)
# ---------------------------------------------------------------------------

@dataclass
class DocChunk:
    content: str
    language: str       # "markdown" | "yaml"
    chunk_type: str     # heading | preamble | yaml_section | yaml_header
    name: str           # heading text or YAML top-level key
    file_path: str
    start_line: int
    end_line: int

    @property
    def metadata(self) -> dict:
        return {
            "language":   self.language,
            "chunk_type": self.chunk_type,
            "name":       self.name,
            "source":     self.file_path,
            "start_line": self.start_line,
            "end_line":   self.end_line,
        }


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

# Matches ATX headings: # Heading, ## Heading, ### Heading
_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MAX_CHUNK_CHARS = 2000          # sub-divide sections longer than this
_OVERLAP_LINES   = 2             # lines of overlap between sub-chunks


class MarkdownChunker:
    """
    Split a Markdown document at heading boundaries.

    Each ``# / ## / ###`` heading becomes the start of a new chunk.
    Text before the first heading is collected as a ``preamble`` chunk.
    Sections longer than *max_chars* are split at blank lines with overlap.
    """

    def __init__(self, max_chars: int = _MAX_CHUNK_CHARS):
        self.max_chars = max_chars

    def chunk(self, source: str, file_path: str) -> list[DocChunk]:
        lines = source.splitlines()
        chunks: list[DocChunk] = []

        # Locate heading positions
        sections: list[tuple[int, str, int]] = []   # (line_index, heading_text, level)
        for i, line in enumerate(lines):
            m = _MD_HEADING.match(line)
            if m:
                level = len(m.group(1))
                sections.append((i, m.group(2).strip(), level))

        # Build (start, end, name, chunk_type) spans
        spans: list[tuple[int, int, str, str]] = []
        if sections:
            # Preamble before first heading
            if sections[0][0] > 0:
                spans.append((0, sections[0][0] - 1, Path(file_path).stem, "preamble"))
            for idx, (line_no, heading, _level) in enumerate(sections):
                end_line = (sections[idx + 1][0] - 1) if idx + 1 < len(sections) else len(lines) - 1
                spans.append((line_no, end_line, heading, "heading"))
        else:
            # No headings – treat the whole file as one chunk
            spans.append((0, len(lines) - 1, Path(file_path).stem, "preamble"))

        for start, end, name, ctype in spans:
            body = "\n".join(lines[start : end + 1]).strip()
            if not body:
                continue
            if len(body) <= self.max_chars:
                chunks.append(DocChunk(
                    content    = body,
                    language   = "markdown",
                    chunk_type = ctype,
                    name       = name,
                    file_path  = file_path,
                    start_line = start + 1,
                    end_line   = end + 1,
                ))
            else:
                chunks.extend(self._split_large(body, name, ctype, file_path, start))

        return chunks

    def _split_large(
        self,
        text: str,
        name: str,
        ctype: str,
        file_path: str,
        base_line: int,
    ) -> list[DocChunk]:
        """Split oversized sections at blank lines."""
        paragraphs = re.split(r"\n{2,}", text)
        result: list[DocChunk] = []
        current: list[str] = []
        current_len = 0
        part = 0

        for para in paragraphs:
            if current_len + len(para) > self.max_chars and current:
                chunk_text = "\n\n".join(current)
                result.append(DocChunk(
                    content    = chunk_text,
                    language   = "markdown",
                    chunk_type = ctype,
                    name       = f"{name} (part {part + 1})",
                    file_path  = file_path,
                    start_line = base_line + 1,
                    end_line   = base_line + chunk_text.count("\n") + 1,
                ))
                part += 1
                # Overlap: keep last paragraph
                current = current[-_OVERLAP_LINES:] if current else []
                current_len = sum(len(p) for p in current)

            current.append(para)
            current_len += len(para)

        if current:
            chunk_text = "\n\n".join(current)
            result.append(DocChunk(
                content    = chunk_text,
                language   = "markdown",
                chunk_type = ctype,
                name       = f"{name} (part {part + 1})" if part > 0 else name,
                file_path  = file_path,
                start_line = base_line + 1,
                end_line   = base_line + chunk_text.count("\n") + 1,
            ))
        return result


# ---------------------------------------------------------------------------
# YAML chunker
# ---------------------------------------------------------------------------

class YAMLChunker:
    """
    Split a YAML schema/contract file at top-level key boundaries.

    Top-level keys (``columns:``, ``quality_rules:``, ``lineage:``, etc.) are
    extracted as individual chunks.  Scalar header fields (``table:``,
    ``description:``, ``database:``, etc.) are grouped into a single
    ``yaml_header`` chunk.
    """

    # Keys whose values are treated as scalars and grouped into the header chunk
    HEADER_KEYS = {
        "table", "database", "schema", "description",
        "update_frequency", "owner", "version", "status",
    }

    def chunk(self, source: str, file_path: str) -> list[DocChunk]:
        try:
            import yaml as _yaml
            data = _yaml.safe_load(source)
        except Exception:
            # Fall back to raw line-based splitting if PyYAML is unavailable
            return self._raw_chunk(source, file_path)

        if not isinstance(data, dict):
            return self._raw_chunk(source, file_path)

        lines = source.splitlines()
        chunks: list[DocChunk] = []

        # Find line numbers for each top-level key
        key_lines: dict[str, int] = {}
        for i, line in enumerate(lines):
            m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):", line)
            if m and m.group(1) in data:
                key_lines[m.group(1)] = i

        sorted_keys = sorted(key_lines, key=lambda k: key_lines[k])
        key_end_lines: dict[str, int] = {}
        for idx, key in enumerate(sorted_keys):
            if idx + 1 < len(sorted_keys):
                key_end_lines[key] = key_lines[sorted_keys[idx + 1]] - 1
            else:
                key_end_lines[key] = len(lines) - 1

        # Group header scalar keys
        header_lines: list[str] = []
        header_start = None
        header_end   = 0

        for key in sorted_keys:
            if key in self.HEADER_KEYS:
                sl = key_lines[key]
                el = key_end_lines[key]
                if header_start is None:
                    header_start = sl
                header_end = el
                header_lines.extend(lines[sl : el + 1])
            else:
                sl   = key_lines[key]
                el   = key_end_lines[key]
                body = "\n".join(lines[sl : el + 1]).strip()
                if body:
                    chunks.append(DocChunk(
                        content    = body,
                        language   = "yaml",
                        chunk_type = "yaml_section",
                        name       = key,
                        file_path  = file_path,
                        start_line = sl + 1,
                        end_line   = el + 1,
                    ))

        if header_lines:
            chunks.insert(0, DocChunk(
                content    = "\n".join(header_lines),
                language   = "yaml",
                chunk_type = "yaml_header",
                name       = data.get("table", Path(file_path).stem),
                file_path  = file_path,
                start_line = (header_start or 0) + 1,
                end_line   = header_end + 1,
            ))

        return chunks

    def _raw_chunk(self, source: str, file_path: str) -> list[DocChunk]:
        """Fallback: treat the entire YAML as a single misc chunk."""
        lines = source.splitlines()
        return [DocChunk(
            content    = source.strip(),
            language   = "yaml",
            chunk_type = "yaml_section",
            name       = Path(file_path).stem,
            file_path  = file_path,
            start_line = 1,
            end_line   = len(lines),
        )]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_doc_file(file_path: str) -> list[DocChunk]:
    """Chunk a Markdown or YAML file and return DocChunk objects."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Cannot read {file_path}: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".md":
        return MarkdownChunker().chunk(source, file_path)
    elif suffix in {".yaml", ".yml"}:
        return YAMLChunker().chunk(source, file_path)
    else:
        raise ValueError(f"Unsupported document extension: {suffix}")


def is_doc_file(path: str | Path) -> bool:
    """Return True if the file should be processed by the doc chunker."""
    return Path(path).suffix.lower() in {".md", ".yaml", ".yml"}
