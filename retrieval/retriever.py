from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from embeddings.embedder import Embedder
from retrieval.semantic_search import InMemoryVectorIndex
from retrieval.store import DocumentStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_index: InMemoryVectorIndex, document_store: DocumentStore) -> None:
        self.embedder = embedder
        self.vector_index = vector_index
        self.document_store = document_store

    def retrieve(self, query: str, top_k: int, doc_ids: Optional[List[str]] = None) -> List[Dict[str, object]]:
        if not query.strip():
            return []
        query_vector = self.embedder.embed([query])
        results = self.vector_index.search(query_vector, max(top_k * 5, top_k))
        if not doc_ids:
            return results[:top_k]

        filtered: List[Dict[str, object]] = []
        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("doc_id") in doc_ids:
                filtered.append(result)
            if len(filtered) >= top_k:
                break
        return filtered
