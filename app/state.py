from dataclasses import dataclass

from app.settings import Settings
from embeddings.embedder import Embedder
from retrieval.llm_client import LlmClient
from retrieval.semantic_search import InMemoryVectorIndex
from retrieval.store import DocumentStore
from retrieval.retriever import Retriever


@dataclass
class AppState:
    settings: Settings
    embedder: Embedder
    vector_index: InMemoryVectorIndex
    document_store: DocumentStore
    retriever: Retriever
    llm_client: LlmClient


async def create_state(settings: Settings) -> AppState:
    embedder = Embedder(settings.embedding_model)
    vector_index = InMemoryVectorIndex(embedder.embedding_dim)
    document_store = DocumentStore()
    retriever = Retriever(embedder, vector_index, document_store)
    llm_client = LlmClient(settings)
    return AppState(
        settings=settings,
        embedder=embedder,
        vector_index=vector_index,
        document_store=document_store,
        retriever=retriever,
        llm_client=llm_client,
    )
