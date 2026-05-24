from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from api.models import HackathonRequest, QueryRequest, QueryResponse, UploadResponse
from chunking.preprocess import extract_text_from_url, preprocess_document
from retrieval.semantic_search import find_relevant_sections
from retrieval.store import DocumentChunk, DocumentRecord

router = APIRouter()


@router.get("/")
def root() -> dict:
    return {"message": "LLM-Powered Intelligent Query Retrieval System", "status": "running"}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), request: Request = None) -> UploadResponse:
    if request is None:
        raise HTTPException(status_code=500, detail="Request context missing")
    state = request.app.state.rag_state
    settings = request.app.state.settings

    data = await file.read()
    try:
        chunks = preprocess_document(file.filename, data, settings.chunk_size, settings.chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=422, detail="No text extracted from document")

    doc_id = chunks[0].doc_id
    texts = [chunk.text for chunk in chunks]
    vectors = state.embedder.embed(texts)

    metadatas: List[dict] = []
    document_chunks: List[DocumentChunk] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["doc_id"] = chunk.doc_id
        metadata["chunk_id"] = chunk.chunk_id
        metadatas.append(metadata)
        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata=metadata,
            )
        )

    state.vector_index.add(vectors, texts, metadatas)

    record = DocumentRecord(
        doc_id=doc_id,
        source_name=file.filename,
        text="\n\n".join(texts),
        chunks=document_chunks,
    )
    state.document_store.add_document(record)

    return UploadResponse(document_id=doc_id, source_name=file.filename, chunks_indexed=len(chunks))


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request = None) -> QueryResponse:
    if request is None:
        raise HTTPException(status_code=500, detail="Request context missing")
    state = request.app.state.rag_state
    settings = request.app.state.settings

    top_k = req.top_k or settings.default_top_k
    results = state.retriever.retrieve(req.query, top_k, req.doc_ids)

    if not results:
        documents = state.document_store.list_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No documents indexed")
        combined_text = "\n\n".join(doc.text for doc in documents)
        context = find_relevant_sections(combined_text, req.query)
    else:
        context = "\n\n".join(result["text"] for result in results)
        if len(context) > settings.max_context_chars:
            context = context[: settings.max_context_chars] + "\n\n[Context truncated...]"

    llm_response = await state.llm_client.ask(req.query, context)
    answer = llm_response.get("answer", "") if isinstance(llm_response, dict) else str(llm_response)

    retrieved_chunks = [
        {
            "text": result.get("text"),
            "score": result.get("score"),
            "metadata": result.get("metadata", {}),
        }
        for result in results
    ]
    sources = [result.get("metadata", {}) for result in results]
    confidence = 0.0
    if results:
        max_score = max(result.get("score", 0.0) for result in results)
        confidence = max(0.0, min((max_score + 1.0) / 2.0, 1.0))

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        confidence_score=confidence,
    )


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str, request: Request = None) -> dict:
    if request is None:
        raise HTTPException(status_code=500, detail="Request context missing")
    state = request.app.state.rag_state
    deleted = state.document_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/ask")
async def ask(req: QueryRequest, request: Request = None) -> dict:
    if request is None:
        raise HTTPException(status_code=500, detail="Request context missing")
    settings = request.app.state.settings
    state = request.app.state.rag_state

    pdf_path = f"{settings.data_dir}/Dataset1.pdf"
    try:
        with open(pdf_path, "rb") as file_handle:
            data = file_handle.read()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="PDF not found") from exc

    chunks = preprocess_document("Dataset1.pdf", data, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text extracted from document")
    text = "\n\n".join(chunk.text for chunk in chunks)
    context = find_relevant_sections(text, req.query)
    response = await state.llm_client.ask(req.query, context)
    return response


@router.post("/hackrx/run")
async def hackathon(request_body: HackathonRequest, authorization: Optional[str] = Header(None), request: Request = None) -> dict:
    if request is None:
        raise HTTPException(status_code=500, detail="Request context missing")
    state = request.app.state.rag_state
    pdf_text = extract_text_from_url(request_body.documents)
    answers = []
    for question in request_body.questions:
        context = find_relevant_sections(pdf_text, question)
        response = await state.llm_client.ask(question, context)
        answers.append(response.get("answer", "Unable to determine"))
    return {"answers": answers}
