from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np


class InMemoryVectorIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, str]] = []

    def add(self, vectors: np.ndarray, texts: List[str], metadatas: List[Dict[str, str]]) -> None:
        if vectors.size == 0:
            return
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, object]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict[str, object]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                }
            )
        return results


def find_relevant_sections(text: str, query: str, max_sections: int = 5) -> str:
    query_words = [
        w.lower()
        for w in query.split()
        if len(w) > 3 and w.lower() not in {"what", "when", "where", "how", "the", "and", "for"}
    ]
    sections: List[str] = []
    current = ""
    for line in text.split("\n"):
        if line.strip() and (
            line.isupper()
            or any(k in line.lower() for k in ["section", "clause", "article", "chapter"])
            or len(line) < 100
        ):
            if current.strip():
                sections.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        sections.append(current.strip())
    scored = sorted(
        ((s, sum(w in s.lower() for w in query_words)) for s in sections),
        key=lambda x: x[1],
        reverse=True,
    )
    return "\n\n---\n\n".join(s for s, _ in scored[:max_sections])
