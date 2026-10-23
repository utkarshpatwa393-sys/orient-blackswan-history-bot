"""
Unit tests for strict abstention and 'Not found' fallback verification.
"""

from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.rag_chain import RAGChain


def test_empty_question_returns_not_found():
    """Verify that empty or whitespace questions immediately abstain."""
    vm = VectorStoreManager()
    vs = vm.get_vector_store()
    retriever = DocumentRetriever(vs, k=4)
    rag_chain = RAGChain(retriever)

    result = rag_chain.query("   ")
    assert result["answer"] == "Not found"
    assert result["is_found"] is False


def test_out_of_domain_question_abstains():
    """Verify that questions outside the document scope return exact 'Not found'."""
    vm = VectorStoreManager()
    vs = vm.get_vector_store()
    retriever = DocumentRetriever(vs, k=4)
    rag_chain = RAGChain(retriever)

    result = rag_chain.query("Who is the current Minister of Education?")
    assert result["answer"] == "Not found"
    assert result["is_found"] is False
