from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    retrieved_chunks: List[dict]
    confidence_score: float


class UploadResponse(BaseModel):
    document_id: str
    source_name: str
    chunks_indexed: int


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]
