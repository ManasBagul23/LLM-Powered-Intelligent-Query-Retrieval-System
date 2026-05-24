# LLM-Powered Intelligent Query Retrieval System

Enterprise-grade Retrieval-Augmented Generation (RAG) platform for PDF and DOCX ingestion, semantic search, and explainable answers with citations.

## Features

- PDF and DOCX ingestion
- Semantic embeddings with FAISS vector search
- Contextual LLM answers with citations
- Multi-document querying
- Async FastAPI endpoints
- Modular, production-ready architecture

## Architecture

User Query
-> FastAPI API Layer
-> Embedding Engine
-> Vector Search (FAISS)
-> Retrieval Layer
-> Prompt Construction
-> LLM Inference
-> Structured JSON Response

## Project Structure

- app/ - application initialization, settings, logging, server
- api/ - REST API endpoints
- embeddings/ - embedding generation
- retrieval/ - semantic search and LLM logic
- chunking/ - preprocessing and chunking
- frontend/ - UI layer (Streamlit or React)
- tests/ - test suite
- docs/ - documentation and diagrams

## Setup

1. Create a virtual environment and install dependencies:

   pip install -r requirements.txt

2. Configure environment variables in a .env file:

   GROQ_KEYS=your_key_1,your_key_2
   EMBEDDING_MODEL=all-MiniLM-L6-v2

3. Start the API server:

   uvicorn main:app --reload

## API Usage

- POST /upload
- POST /query
- GET /health
- DELETE /document/{id}

### Example Query Response

{
  "query": "What is covered under policy section 3?",
  "answer": "...",
  "sources": [{"doc_id": "...", "chunk_id": "...", "page": "2"}],
  "retrieved_chunks": [{"text": "...", "score": 0.82}],
  "confidence_score": 0.91
}

## Screenshots

Add UI screenshots in docs/.

## Sample Outputs

Add sample responses and evaluation results in docs/.

## Future Scope

- Hybrid retrieval (BM25 + embeddings)
- Reranking models
- Streaming responses
- Authentication and authorization
- Metadata filters and ACLs

"# LLM-Powered-Intelligent-Query-Retrieval-System" 
