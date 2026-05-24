from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List
import uuid

import httpx
import fitz
import docx

from chunking.cleaner import clean_text
from chunking.splitter import split_text


@dataclass
class PreprocessedChunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, str]


def extract_text_from_pdf_bytes(data: bytes) -> List[str]:
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def extract_text_from_docx_bytes(data: bytes) -> str:
    document = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_bytes(filename: str, data: bytes) -> List[str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    if lower.endswith(".docx"):
        return [extract_text_from_docx_bytes(data)]
    raise ValueError("Unsupported file type")


def extract_text_from_url(url: str) -> str:
    with httpx.stream("GET", url, timeout=60) as response:
        response.raise_for_status()
        data = b"".join(response.iter_bytes())
    pages = extract_text_from_pdf_bytes(data)
    return "\n\n".join(pages)


def preprocess_document(
    filename: str,
    data: bytes,
    chunk_size: int,
    overlap: int,
) -> List[PreprocessedChunk]:
    doc_id = str(uuid.uuid4())
    pages = extract_text_from_bytes(filename, data)
    chunks: List[PreprocessedChunk] = []

    for page_index, page_text in enumerate(pages, start=1):
        cleaned = clean_text(page_text)
        for chunk_index, chunk in enumerate(split_text(cleaned, chunk_size, overlap)):
            chunk_id = str(uuid.uuid4())
            metadata = {
                "source_name": filename,
                "page": str(page_index),
                "chunk_index": str(chunk_index),
            }
            chunks.append(
                PreprocessedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk,
                    metadata=metadata,
                )
            )

    return chunks
