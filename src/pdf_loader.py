"""
PDF loader and text extraction module.
Preserves 1-based page numbers and attaches rich metadata.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def calculate_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for index invalidation and tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


class PDFLoader:
    """Loads PDF documents with strict page-number preservation and metadata."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def verify_file(self) -> Tuple[bool, str]:
        """Verify that the PDF file exists and is accessible."""
        if not self.file_path.exists():
            return False, f"PDF file not found at '{self.file_path}'. Please place 'History_2024-25.pdf' in the 'data/' directory."
        if not self.file_path.is_file():
            return False, f"Target path '{self.file_path}' is a directory, not a PDF file."
        if self.file_path.stat().st_size == 0:
            return False, f"PDF file at '{self.file_path}' is empty (0 bytes)."
        return True, "PDF file verified successfully."

    def load(self) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Extract text from every page, preserving 1-based page numbers.
        Returns list of Document objects and ingestion stats.
        """
        valid, msg = self.verify_file()
        if not valid:
            raise FileNotFoundError(msg)

        if PYMUPDF_AVAILABLE:
            return self._load_with_pymupdf()
        elif PYPDF_AVAILABLE:
            return self._load_with_pypdf()
        else:
            raise ImportError("Neither 'pymupdf' nor 'pypdf' is installed. Please install 'pymupdf'.")

    def _load_with_pymupdf(self) -> Tuple[List[Document], Dict[str, Any]]:
        doc = fitz.open(str(self.file_path))
        total_pages = len(doc)
        documents: List[Document] = []
        empty_pages: List[int] = []
        file_hash = calculate_file_hash(self.file_path)

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-based page numbering
            page = doc[page_idx]
            text = page.get_text("text").strip()

            if not text:
                empty_pages.append(page_num)
                continue

            metadata = {
                "source": self.file_path.name,
                "file_path": str(self.file_path),
                "page": page_num,
                "total_pages": total_pages,
                "file_hash": file_hash,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        doc.close()

        stats = {
            "source_file": self.file_path.name,
            "total_pages": total_pages,
            "processed_pages": len(documents),
            "empty_pages": empty_pages,
            "file_hash": file_hash,
            "scanned_warning": len(empty_pages) > 0 and len(documents) == 0,
        }
        return documents, stats

    def _load_with_pypdf(self) -> Tuple[List[Document], Dict[str, Any]]:
        reader = PdfReader(str(self.file_path))
        total_pages = len(reader.pages)
        documents: List[Document] = []
        empty_pages: List[int] = []
        file_hash = calculate_file_hash(self.file_path)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-based page numbering
            text = (page.extract_text() or "").strip()

            if not text:
                empty_pages.append(page_num)
                continue

            metadata = {
                "source": self.file_path.name,
                "file_path": str(self.file_path),
                "page": page_num,
                "total_pages": total_pages,
                "file_hash": file_hash,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        stats = {
            "source_file": self.file_path.name,
            "total_pages": total_pages,
            "processed_pages": len(documents),
            "empty_pages": empty_pages,
            "file_hash": file_hash,
            "scanned_warning": len(empty_pages) > 0 and len(documents) == 0,
        }
        return documents, stats

    def get_page_text(self, page_number: int) -> str:
        """Fetch exact text for a given 1-based page number."""
        valid, _ = self.verify_file()
        if not valid:
            return ""

        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(self.file_path))
            if 1 <= page_number <= len(doc):
                text = doc[page_number - 1].get_text("text")
                doc.close()
                return text
            doc.close()
            return ""
        elif PYPDF_AVAILABLE:
            reader = PdfReader(str(self.file_path))
            if 1 <= page_number <= len(reader.pages):
                return reader.pages[page_number - 1].extract_text() or ""
            return ""
        return ""
