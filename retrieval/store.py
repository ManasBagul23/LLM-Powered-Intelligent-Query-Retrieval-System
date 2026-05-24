from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, str]


@dataclass
class DocumentRecord:
    doc_id: str
    source_name: str
    text: str
    chunks: List[DocumentChunk] = field(default_factory=list)


class DocumentStore:
    def __init__(self) -> None:
        self._documents: Dict[str, DocumentRecord] = {}

    def add_document(self, record: DocumentRecord) -> None:
        self._documents[record.doc_id] = record

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._documents.get(doc_id)

    def list_documents(self) -> List[DocumentRecord]:
        return list(self._documents.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False
