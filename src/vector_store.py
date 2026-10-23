"""
ChromaDB persistent vector store management.
Handles indexing, persistence, deduplication, and hash-based invalidation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import Config
from src.embeddings import get_embedding_function


class VectorStoreManager:
    """Manages persistent ChromaDB vector store with hash tracking."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.persist_dir = Path(persist_dir or Config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_model = embedding_model or Config.GOOGLE_EMBEDDING_MODEL
        self.metadata_file = self.persist_dir / "index_meta.json"
        self._vector_store: Optional[Chroma] = None

    def get_metadata(self) -> Dict[str, Any]:
        """Read saved indexing metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        """Save indexing metadata."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(meta, f, indent=2)

    def is_indexed(self, expected_file_hash: Optional[str] = None) -> bool:
        """
        Check if the persistent vector store exists and matches expected file hash.
        """
        meta = self.get_metadata()
        if not meta or meta.get("chunk_count", 0) == 0:
            return False

        if expected_file_hash:
            return meta.get("file_hash") == expected_file_hash

        # Check if chroma sqlite file exists
        sqlite_file = self.persist_dir / "chroma.sqlite3"
        return sqlite_file.exists()

    def get_vector_store(self) -> Chroma:
        """Load or return the active Chroma vector store instance."""
        if self._vector_store is not None:
            return self._vector_store

        embeddings = get_embedding_function(model=self.embedding_model)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )
        return self._vector_store

    def index_documents(
        self,
        chunks: List[Document],
        source_stats: Dict[str, Any],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Embeds and stores chunks in ChromaDB.
        Reuses existing index if hash matches and force_rebuild is False.
        """
        file_hash = source_stats.get("file_hash", "")

        if not force_rebuild and self.is_indexed(expected_file_hash=file_hash):
            meta = self.get_metadata()
            return {
                "status": "reused",
                "message": "Existing vector store is up to date. Re-indexing skipped.",
                "chunk_count": meta.get("chunk_count", len(chunks)),
                "total_pages": meta.get("total_pages", source_stats.get("total_pages", 0)),
                "file_hash": file_hash,
            }

        # Clear existing collection if rebuilding
        if force_rebuild:
            self.clear_index()

        embeddings = get_embedding_function(model=self.embedding_model)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Chroma handles batch indexing
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
        )

        index_meta = {
            "source_file": source_stats.get("source_file", "History_2024-25.pdf"),
            "file_hash": file_hash,
            "total_pages": source_stats.get("total_pages", 0),
            "processed_pages": source_stats.get("processed_pages", 0),
            "chunk_count": len(chunks),
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
        }
        self.save_metadata(index_meta)

        return {
            "status": "indexed",
            "message": f"Successfully indexed {len(chunks)} chunks across {source_stats.get('total_pages', 0)} pages.",
            "chunk_count": len(chunks),
            "total_pages": source_stats.get("total_pages", 0),
            "file_hash": file_hash,
        }

    def clear_index(self) -> None:
        """Clear the current collection and reset cache."""
        import shutil
        self._vector_store = None
        if self.persist_dir.exists():
            for item in self.persist_dir.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
