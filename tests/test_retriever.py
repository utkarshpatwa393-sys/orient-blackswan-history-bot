"""
Unit tests for text chunking, ChromaDB storage, and document retrieval.
"""

from src.config import Config
from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentChunker
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever


def test_chunking_preserves_metadata():
    """Verify that chunks retain 1-based page numbers and have valid chunk IDs."""
    loader = PDFLoader(Config.DEFAULT_PDF_PATH)
    docs, _ = loader.load()
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.split_documents(docs)

    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert "page" in chunk.metadata
        assert isinstance(chunk.metadata["page"], int)
        assert chunk.metadata["page"] >= 1
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) > 0


def test_retriever_query_and_scores():
    """Verify that the retriever returns relevant chunks for document-supported queries."""
    vm = VectorStoreManager()
    vs = vm.get_vector_store()
    retriever = DocumentRetriever(vs, k=4)

    results = retriever.retrieve_with_scores("Orient BlackSwan incorporated")
    assert len(results) > 0

    top_doc, score = results[0]
    assert "page" in top_doc.metadata
    assert score >= 0.0  # Chroma L2 distance is non-negative
