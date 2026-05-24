from typing import List


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks
