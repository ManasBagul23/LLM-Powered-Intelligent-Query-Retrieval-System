# Architecture

## Pipeline

1. Upload document
2. Extract text
3. Clean and chunk
4. Generate embeddings
5. Index vectors
6. Retrieve top-k chunks
7. Build prompt context
8. Generate LLM response
9. Return JSON with citations

## Notes

- Chunk size: 500-800 tokens
- Overlap: 100-150 tokens
- Vector search: cosine similarity (FAISS)
