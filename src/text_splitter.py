"""
Text chunking module with metadata preservation.
Ensures every chunk retains its original 1-based PDF page reference.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Splits documents into overlapping chunks while preserving page numbers and identifiers."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits page documents into chunks.
        Guarantees that chunk metadata inherits the exact 1-based page number of its parent page.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            page_num = doc.metadata.get("page", 1)
            source_file = doc.metadata.get("source", "unknown")
            page_chunks = self.splitter.split_text(doc.page_content)

            for chunk_idx, text in enumerate(page_chunks):
                chunk_id = f"{source_file}_p{page_num}_c{chunk_idx}"
                chunk_metadata: Dict[str, Any] = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "page": page_num,  # Strictly preserve 1-based page number
                    "char_count": len(text),
                }
                all_chunks.append(Document(page_content=text, metadata=chunk_metadata))

        return all_chunks
