"""Load PDF and text/Markdown documents into a normalized representation.

Adding a new format later is a matter of adding a loader branch here; nothing
downstream needs to change.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.models.chunk import compute_document_id

_PDF_SUFFIXES = {".pdf"}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = _PDF_SUFFIXES | _TEXT_SUFFIXES


@dataclass(frozen=True)
class Page:
    """A unit of source text. ``page_number`` is 1-based for PDFs, None otherwise."""

    text: str
    page_number: int | None


@dataclass(frozen=True)
class LoadedDocument:
    """A loaded document: stable id, source title, and ordered pages."""

    document_id: str
    source_title: str
    pages: tuple[Page, ...]


def load(path: str | Path) -> LoadedDocument:
    """Load a supported document from disk into a LoadedDocument.

    Raises:
        FileNotFoundError: the path is not an existing file.
        ValueError: unsupported type, empty file, or no extractable text.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    return load_bytes(p.read_bytes(), source_title=p.name)


def load_bytes(content: bytes, *, source_title: str) -> LoadedDocument:
    """Load a supported document from in-memory bytes.

    ``source_title`` is used only to detect the format (by suffix) and as the
    document's display title — never as a filesystem path. Callers handling
    untrusted uploads must pass a sanitized basename.

    Raises:
        ValueError: unsupported type, empty content, or no extractable text.
    """
    suffix = Path(source_title).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if not content:
        raise ValueError("File is empty")

    document_id = compute_document_id(content)
    pages = _load_pdf(content) if suffix in _PDF_SUFFIXES else _load_text(content)

    if not any(page.text.strip() for page in pages):
        raise ValueError("No extractable text found")

    return LoadedDocument(
        document_id=document_id, source_title=source_title, pages=pages
    )


def _load_pdf(content: bytes) -> tuple[Page, ...]:
    reader = PdfReader(io.BytesIO(content))
    pages = [
        Page(text=page.extract_text() or "", page_number=index)
        for index, page in enumerate(reader.pages, start=1)
    ]
    return tuple(pages)


def _load_text(content: bytes) -> tuple[Page, ...]:
    text = content.decode("utf-8", errors="replace")
    return (Page(text=text, page_number=None),)
