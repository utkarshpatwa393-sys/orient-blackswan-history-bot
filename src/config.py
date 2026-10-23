"""
Configuration module for Orient BlackSwan History Bot.
Loads environment variables safely and defines project-wide defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class Config:
    """Application configuration and environment manager."""

    # Project Paths
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DEFAULT_PDF_PATH: Path = PROJECT_ROOT / os.getenv("PDF_PATH", "data/History_2024-25.pdf")
    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    SUBMISSION_DIR: Path = PROJECT_ROOT / "submission"
    SCREENSHOTS_DIR: Path = PROJECT_ROOT / "screenshots"

    # Vector DB / Collection
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "orient_blackswan_history")

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))

    # Retrieval Configuration
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 4))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", 0.35))

    # Google Gemini Models
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite").strip()
    GOOGLE_EMBEDDING_MODEL: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

    @classmethod
    def validate_api_key(cls) -> bool:
        """Check if Google API key is configured."""
        return bool(cls.GOOGLE_API_KEY and len(cls.GOOGLE_API_KEY) > 10)

    @classmethod
    def get_masked_api_key(cls) -> str:
        """Return masked key for diagnostic logs without leaking secret."""
        if not cls.GOOGLE_API_KEY:
            return "NOT CONFIGURED"
        if len(cls.GOOGLE_API_KEY) <= 8:
            return "********"
        return f"{cls.GOOGLE_API_KEY[:4]}...{cls.GOOGLE_API_KEY[-4:]}"

    @classmethod
    def get_pdf_path(cls) -> Path:
        """Return configured PDF path."""
        return cls.DEFAULT_PDF_PATH
