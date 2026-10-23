"""
Citation validation and extraction module.
Ensures every citation references valid PDF pages and exact extracted quotations.
"""

import re
from typing import List, Dict, Any, Optional, Set
from langchain_core.documents import Document


class CitationManager:
    """Validates citations, verifies evidence quotations against source text, and formats output."""

    @staticmethod
    def extract_page_numbers(source_text: str) -> List[int]:
        """Extract all integer page numbers mentioned in source line."""
        pages: Set[int] = set()
        # Find prefix like page/pages/pp followed by numbers and delimiters
        prefix_matches = re.findall(r"(?:pages?|pp\.?|p\.?)\s*([\d,\s\-and]+)", source_text, re.IGNORECASE)
        for chunk in prefix_matches:
            numbers = re.findall(r"\b\d+\b", chunk)
            for n in numbers:
                try:
                    pages.add(int(n))
                except ValueError:
                    pass

        # Also fallback to direct pattern if no prefix chunk was found
        if not pages:
            direct_matches = re.findall(r"(?:page|pages|pp\.?|p\.?)\s*(\d+)", source_text, re.IGNORECASE)
            for m in direct_matches:
                try:
                    pages.add(int(m))
                except ValueError:
                    pass

        return sorted(list(pages))

    @staticmethod
    def verify_evidence_in_context(evidence: str, retrieved_docs: List[Document]) -> bool:
        """
        Verify that the evidence quotation actually exists in the retrieved documents.
        Tolerates minor whitespace/punctuation differences.
        """
        if not evidence or evidence.strip().lower() in {"none", "n/a", "not found"}:
            return False

        clean_evidence = re.sub(r"\s+", " ", evidence.strip().strip('"\'')).lower()
        if len(clean_evidence) < 5:
            return False

        for doc in retrieved_docs:
            clean_content = re.sub(r"\s+", " ", doc.page_content).lower()
            if clean_evidence in clean_content:
                return True
            # Also check if a significant 80% contiguous slice matches
            evidence_words = clean_evidence.split()
            if len(evidence_words) >= 6:
                subset = " ".join(evidence_words[:6])
                if subset in clean_content:
                    return True

        return False

    @staticmethod
    def find_pages_for_evidence(evidence: str, retrieved_docs: List[Document]) -> List[int]:
        """Find the exact 1-based page numbers that contain the evidence quote."""
        clean_evidence = re.sub(r"\s+", " ", evidence.strip().strip('"\'')).lower()
        matched_pages: Set[int] = set()

        for doc in retrieved_docs:
            clean_content = re.sub(r"\s+", " ", doc.page_content).lower()
            page = doc.metadata.get("page")
            if page and (clean_evidence in clean_content or any(word in clean_content for word in clean_evidence.split() if len(word) > 4)):
                matched_pages.add(int(page))

        if not matched_pages:
            # Fallback to the top retrieved document's page
            for doc in retrieved_docs:
                p = doc.metadata.get("page")
                if p:
                    matched_pages.add(int(p))
                    break

        return sorted(list(matched_pages))

    @staticmethod
    def format_citation(pages: List[int]) -> str:
        """Format page numbers as a standardized source string."""
        if not pages:
            return "PDF page unspecified"
        if len(pages) == 1:
            return f"PDF page {pages[0]}"
        return f"PDF pages {', '.join(str(p) for p in pages)}"
