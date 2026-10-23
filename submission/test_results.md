# Automated Test Results Report

**Project:** Project 4 — Orient BlackSwan History Bot  
**Test Framework:** `pytest 8.4.2` with Python 3.13 / LangChain / ChromaDB  
**Execution Date:** 2026-10-23  
**Status:** All 10 Tests Passed (100% Success Rate)

---

## 1. Test Execution Summary

| Test File | Test Case | Target Capability | Result |
|:---|:---|:---|:---:|
| `tests/test_citations.py` | `test_extract_page_numbers` | Regex parsing of single & multi-page citations | **PASSED** |
| `tests/test_citations.py` | `test_verify_evidence_in_context` | Strict validation of quoted evidence in text | **PASSED** |
| `tests/test_citations.py` | `test_format_citation` | Standardized citation string formatting | **PASSED** |
| `tests/test_not_found.py` | `test_empty_question_returns_not_found` | Immediate abstention on empty input | **PASSED** |
| `tests/test_not_found.py` | `test_out_of_domain_question_abstains` | Strict `Not found` output on out-of-scope query | **PASSED** |
| `tests/test_pdf_loader.py` | `test_pdf_exists_and_loads` | 1-based page metadata extraction & page count | **PASSED** |
| `tests/test_pdf_loader.py` | `test_missing_pdf_handling` | Clean error messages when PDF is missing | **PASSED** |
| `tests/test_pdf_loader.py` | `test_file_hash_calculation` | Deterministic SHA-256 fingerprinting | **PASSED** |
| `tests/test_retriever.py` | `test_chunking_preserves_metadata` | Overlapping chunks inherit 1-based page numbers | **PASSED** |
| `tests/test_retriever.py` | `test_retriever_query_and_scores` | ChromaDB vector search & similarity distances | **PASSED** |

---

## 2. Ingestion Pipeline Statistics

- **Document Ingested:** `data/History_2024-25.pdf`
- **Total Pages:** 80 pages
- **Processed Pages:** 80 pages
- **Chunks Generated:** 81 chunks (1,000 character size, 150 character overlap)
- **Vector Database:** ChromaDB Persistent Collection (`orient_blackswan_history`)
- **Embedding Dimensions:** 3,072 (`models/gemini-embedding-001`)

---

## 3. LMS Evaluation Verification

- **Q1 (Price Update Date):** Grounded -> `1 April 2024` (Page 1)
- **Q2 (Incorporation Year):** Grounded -> `1982` (Page 1)
- **Q3 (Physics Textbook):** Abstention -> `Not found` (Catalogue is History only)
- **Q4 (Minister of Education):** Abstention -> `Not found` (Absent from document)
- **Q5 (Best History Book):** Abstention -> `Not found` (Subjective; no endorsement in catalogue)
