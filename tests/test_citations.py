"""
Unit tests for citation extraction and evidence quotation verification.
"""

from langchain_core.documents import Document
from src.citations import CitationManager


def test_extract_page_numbers():
    """Verify regex extraction of single and multiple page numbers."""
    assert CitationManager.extract_page_numbers("PDF page 12") == [12]
    assert CitationManager.extract_page_numbers("Source: pages 1, 12 and 15") == [1, 12, 15]
    assert CitationManager.extract_page_numbers("No pages listed") == []


def test_verify_evidence_in_context():
    """Verify that quotations are verified strictly against retrieved documents."""
    doc1 = Document(
        page_content="The medieval period witnessed the emergence of new political structures and trade.",
        metadata={"page": 12},
    )
    doc2 = Document(
        page_content="Orient BlackSwan was incorporated in 1982 in Hyderabad.",
        metadata={"page": 1},
    )
    docs = [doc1, doc2]

    # Valid quotations from text
    assert CitationManager.verify_evidence_in_context("emergence of new political structures", docs) is True
    assert CitationManager.verify_evidence_in_context("incorporated in 1982", docs) is True

    # Hallucinated or non-existent quotation
    assert CitationManager.verify_evidence_in_context("Albert Einstein published in 1905", docs) is False
    assert CitationManager.verify_evidence_in_context("", docs) is False
    assert CitationManager.verify_evidence_in_context("None", docs) is False


def test_format_citation():
    """Verify standardized string formatting."""
    assert CitationManager.format_citation([12]) == "PDF page 12"
    assert CitationManager.format_citation([1, 12]) == "PDF pages 1, 12"
    assert CitationManager.format_citation([]) == "PDF page unspecified"
