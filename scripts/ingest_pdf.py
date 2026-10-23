"""
Standalone ingestion script for Orient BlackSwan History Catalogue PDF.
Processes pages, extracts metadata, creates overlapping chunks, and stores in ChromaDB.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentChunker
from src.vector_store import VectorStoreManager


def ingest_pdf(pdf_path: Path, force: bool = False):
    """Run the complete ingestion pipeline."""
    print("=" * 65)
    print("  ORIENT BLACKSWAN HISTORY BOT - PDF INGESTION PIPELINE")
    print("=" * 65)
    print(f"Target PDF Path: {pdf_path}")
    print(f"Chroma DB Path : {Config.CHROMA_PERSIST_DIR}")
    print(f"Collection Name: {Config.COLLECTION_NAME}")
    print(f"Embedding Model: {Config.GOOGLE_EMBEDDING_MODEL}")
    print("-" * 65)

    # Step 1: Verify file existence
    loader = PDFLoader(pdf_path)
    valid, msg = loader.verify_file()
    if not valid:
        print(f"\n[ERROR] {msg}")
        print("\nAction Required:")
        print(f"1. Copy your 'History_2024-25.pdf' file into: {Config.DATA_DIR}")
        print("2. Re-run: python scripts/ingest_pdf.py\n")
        sys.exit(1)

    # Step 2: Load and extract text
    print("\n[1/4] Extracting text and page metadata from PDF...")
    documents, stats = loader.load()

    print(f"  • Total Pages Found    : {stats['total_pages']}")
    print(f"  • Processed Text Pages : {stats['processed_pages']}")
    print(f"  • Empty/Skipped Pages  : {len(stats['empty_pages'])} {stats['empty_pages'] if stats['empty_pages'] else ''}")
    print(f"  • Document SHA-256     : {stats['file_hash'][:16]}...")

    if stats.get("scanned_warning"):
        print("[WARNING] No extractable text found. Document may be scanned and require OCR.")

    # Step 3: Split into chunks
    print("\n[2/4] Splitting text into overlapping chunks...")
    chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunks = chunker.split_documents(documents)
    print(f"  • Chunks Created       : {len(chunks)}")
    print(f"  • Chunk Size           : {Config.CHUNK_SIZE} characters")
    print(f"  • Chunk Overlap        : {Config.CHUNK_OVERLAP} characters")

    # Step 4: Index into ChromaDB
    print("\n[3/4] Indexing into persistent ChromaDB vector store...")
    vector_manager = VectorStoreManager()
    result = vector_manager.index_documents(
        chunks=chunks,
        source_stats=stats,
        force_rebuild=force,
    )

    # Step 5: Report Results
    print("\n[4/4] Ingestion Summary:")
    print(f"  • Status               : {result['status'].upper()}")
    print(f"  • Details              : {result['message']}")
    print(f"  • Total Chunks Stored  : {result['chunk_count']}")
    print("=" * 65)
    print("  INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into ChromaDB for Orient BlackSwan History Bot.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(Config.DEFAULT_PDF_PATH),
        help="Path to the PDF catalogue file (default: data/History_2024-25.pdf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild index even if document hash has not changed.",
    )
    args = parser.parse_args()
    ingest_pdf(Path(args.pdf), force=args.force)
