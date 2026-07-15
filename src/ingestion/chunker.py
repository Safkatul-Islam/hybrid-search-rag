"""Split a LoadedDocument into Chunks with stable, sequential identity.

Chunking runs per page so page numbers are preserved for citations. Positions
are assigned sequentially across the whole document, so the same input always
yields the same chunk ids.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import LoadedDocument
from src.models.chunk import Chunk, make_chunk_id


def chunk_document(
    document: LoadedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a document into Chunks.

    Args:
        document: the loaded document to split.
        chunk_size: target maximum characters per chunk.
        chunk_overlap: characters of overlap between adjacent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Chunk] = []
    position = 0
    for page in document.pages:
        for piece in splitter.split_text(page.text):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=make_chunk_id(document.document_id, position),
                    text=piece,
                    source_title=document.source_title,
                    position=position,
                    page=page.page_number,
                )
            )
            position += 1
    return chunks
