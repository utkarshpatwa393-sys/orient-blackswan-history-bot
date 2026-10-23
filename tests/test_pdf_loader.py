"""
Unit tests for PDF loading and page metadata preservation.
"""

import pytest
from pathlib import Path
from src.pdf_loader import PDFLoader, calculate_file_hash
from src.config import Config


def test_pdf_exists_and_loads():
    """Verify that the catalogue exists, loads all pages, and preserves 1-based numbers."""
    pdf_path = Config.DEFAULT_PDF_PATH
    assert pdf_path.exists(), f"PDF does not exist at {pdf_path}"

    loader = PDFLoader(pdf_path)
    valid, msg = loader.verify_file()
    assert valid is True

    documents, stats = loader.load()
    assert stats["total_pages"] >= 1
    assert stats["processed_pages"] >= 1
    assert len(documents) == stats["processed_pages"]

    # Check 1-based page numbers
    for idx, doc in enumerate(documents, start=1):
        assert doc.metadata["page"] == idx
        assert "source" in doc.metadata
        assert "file_hash" in doc.metadata
        assert len(doc.page_content.strip()) > 0


def test_missing_pdf_handling(tmp_path):
    """Verify that non-existent PDF files trigger clear actionable error message."""
    non_existent = tmp_path / "does_not_exist.pdf"
    loader = PDFLoader(non_existent)
    valid, msg = loader.verify_file()
    assert valid is False
    assert "not found" in msg.lower()

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_file_hash_calculation():
    """Verify that SHA-256 hash calculation is deterministic."""
    pdf_path = Config.DEFAULT_PDF_PATH
    hash1 = calculate_file_hash(pdf_path)
    hash2 = calculate_file_hash(pdf_path)
    assert hash1 == hash2
    assert len(hash1) == 64
