"""
Document retriever with relevance scoring and threshold filtering.
"""

from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import Config


class DocumentRetriever:
    """Retrieves relevant chunks with similarity scores and relevance checks."""

    def __init__(
        self,
        vector_store: Chroma,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.k = k or Config.RETRIEVER_K
        self.score_threshold = score_threshold if score_threshold is not None else Config.SCORE_THRESHOLD

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k documents for a query."""
        results_with_scores = self.retrieve_with_scores(query)
        return [doc for doc, _ in results_with_scores]

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve chunks with distance/similarity scores.
        Chroma returns L2 distance (lower is closer/better).
        """
        query = query.strip()
        if not query:
            return []

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.k,
            )
            return results
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return []

    def check_relevance(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> bool:
        """
        Heuristic relevance check before passing to LLM.
        Verifies that at least one keyword or semantic match is present.
        """
        if not retrieved_docs:
            return False

        # In Chroma L2 distance, smaller distance = higher similarity
        # If best distance is exceedingly large (e.g. > 1.8), retrieval has no relevance
        best_score = retrieved_docs[0][1]
        if best_score > 1.85:
            return False

        # Check for query keyword overlap
        query_words = set(query.lower().split())
        # Strip common stopwords
        stopwords = {"what", "is", "the", "of", "a", "an", "in", "which", "who", "for", "to", "at", "by", "on"}
        key_terms = query_words - stopwords

        if not key_terms:
            return True

        combined_text = " ".join([doc.page_content.lower() for doc, _ in retrieved_docs])
        has_overlap = any(term in combined_text for term in key_terms)
        return has_overlap
