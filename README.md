# Project 4 — Orient BlackSwan History Bot
## Document-Grounded Academic RAG System | Python + LangChain + ChromaDB + Google Gemini

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-orange.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/Model-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **"Ask. Explore. Learn."** — A document-grounded AI research assistant designed strictly for the *Orient BlackSwan History Catalogue 2024–25 (~80 pages)*.

---

## 1. Project Overview

**Orient BlackSwan History Bot** is a Retrieval-Augmented Generation (RAG) system built to answer natural-language academic queries grounded exclusively in the *Orient BlackSwan History Catalogue 2024–25*.

Traditional Large Language Models (LLMs) are prone to hallucinations, out-of-date answers, and invented citations. This application solves that problem by:
1. Indexing the actual PDF catalogue into **ChromaDB** using **Google Gemini Embeddings** (`models/gemini-embedding-001`).
2. Retrieving the most relevant document chunks for each user query.
3. Providing **strict grounding** via **Google Gemini** (`gemini-3.5-flash-lite`), accompanied by exact 1-based PDF page numbers and verbatim supporting quotations.
4. Enforcing an uncompromising **abstention policy**: whenever an answer cannot be explicitly verified in the PDF catalogue, the application returns exactly `Not found`.

---

## 2. Problem Statement & Assignment Objectives

### Problem Statement
Build a RAG system (`Python` + `ChromaDB` + `LangChain` + `Google Gemini`) over the Orient BlackSwan History catalogue PDF (`History_2024-25.pdf`, ~80 pages). The bot must answer queries with exact page references and respond with `Not found` whenever the required information is absent from the catalogue.

### Core Objectives
- **Strict Grounding:** Zero reliance on external knowledge, training assumptions, or internet searches.
- **Accurate Citations:** Citing original 1-based PDF page numbers.
- **Verifiable Evidence:** Extracting exact quotations from the retrieved context.
- **Strict Abstention:** Returning `Not found` for out-of-domain or ambiguous questions.
- **Academic UI/UX:** A split-screen interface featuring an interactive PDF document reader alongside the History Bot chat.
- **College Viva-Ready:** Clean code modularity, automated test coverage, and documentation.

---

## 3. The 5 LMS Assignment Questions & Verified Answers

The assignment requires addressing five questions from the LMS form. All answers have been verified against the catalogue using the live RAG pipeline:

| Q# | Assignment Question | Grounded Answer | Source Page | Evidence / Rationale | RAG Decision |
|:---|:---|:---|:---:|:---|:---:|
| **Q1** | **On what date were the prices in this catalogue last updated?** | The prices in the catalogue were last updated on **1 April 2024**. | **Page 1** | *"The prices in this catalogue were last updated on 1 April 2024. All prices listed in this catalogue are effective from 1 April 2024..."* | ✅ **Supported** |
| **Q2** | **In which year was the publisher (Orient BlackSwan) incorporated?** | Orient BlackSwan was incorporated in **1982**. | **Page 1** | *"Orient BlackSwan was incorporated in 1982. Orient BlackSwan Private Limited is one of India's leading independent academic publishers."* | ✅ **Supported** |
| **Q3** | **What is the price of a Physics textbook?** | `Not found` | **N/A** | The catalogue strictly covers History, Archaeology, and South Asian Studies. No physics textbooks exist in the document. | 🛡️ **Abstained (`Not found`)** |
| **Q4** | **Who is the current Minister of Education?** | `Not found` | **N/A** | The document is an academic book catalogue, not a government directory. The model refuses to use outside general knowledge. | 🛡️ **Abstained (`Not found`)** |
| **Q5** | **Which is the "best" history book for a beginner?** | `Not found` | **N/A** | "Best" is subjective. The catalogue presents academic titles without declaring an official beginner recommendation. | 🛡️ **Abstained (`Not found`)** |

---

## 4. System Architecture

The following diagram illustrates how raw PDF documents are processed, embedded, indexed, retrieved, and validated:

```text
                 History_2024-25.pdf (~80 pages)
                                |
                                v
                      PyMuPDF PDF Loader
                 (1-based page metadata extraction)
                                |
                                v
                     Document Chunker
                 (1,000 char size / 150 overlap)
                                |
                                v
                   Google Gemini Embeddings
                 (models/gemini-embedding-001)
                                |
                                v
                 ChromaDB Persistent Store
               (Local storage in ./chroma_db)
                                |
                                |
User Question ------------------+
       |
       v
Query Embedding
       |
       v
ChromaDB Retriever (k=4, L2 distance)
       |
       v
Relevance & Distance Validation Filter
       |
       v
Relevant Context Chunks with Page Tags
       |
       v
LangChain + Google Gemini (gemini-3.5-flash-lite)
(Strict Prompt Grounding & Instruction Following)
       |
       v
Citation Verifier & Evidence Matcher
       |
       +-------------------------------+
       |                               |
       v                               v
[Grounded Answer + Citations]    [Not found]
       |                               |
       +---------------+---------------+
                       |
                       v
            Streamlit Editorial UI
```

---

## 5. Technology Stack

- **Programming Language:** Python 3.11+ / 3.12 / 3.13
- **RAG Orchestration:** `LangChain` (`langchain`, `langchain-core`, `langchain-community`)
- **Vector Database:** `ChromaDB` (`langchain-chroma`, `chromadb`)
- **LLM & Embeddings:** `langchain-google-genai` (Google Gemini 3.5 Flash Lite + Gemini Embedding 001)
- **PDF Extraction:** `PyMuPDF` (`fitz`) with strict 1-based page metadata
- **Frontend UI:** `Streamlit` with custom editorial CSS styling
- **Environment Management:** `python-dotenv`
- **Testing Suite:** `pytest` (10 automated unit and integration tests)

---

## 6. Project Directory Structure

```text
GEN-AI_MIDSEM/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive documentation
├── .env                        # Local environment secrets (excluded from Git)
├── .env.example                # Configuration template
├── .gitignore                  # Git exclusion rules
├── pytest.ini                  # Pytest configuration
│
├── data/
│   ├── .gitkeep
│   └── History_2024-25.pdf     # Source catalogue PDF (~80 pages)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Environment configuration & secret masking
│   ├── pdf_loader.py           # PyMuPDF extractor with 1-based page tracking
│   ├── text_splitter.py        # RecursiveCharacterTextSplitter with chunk IDs
│   ├── embeddings.py           # Google Gemini embeddings wrapper
│   ├── vector_store.py         # Persistent ChromaDB manager & SHA-256 caching
│   ├── retriever.py            # Similarity search & distance thresholding
│   ├── citations.py            # Quotation verification against context
│   └── rag_chain.py            # Strict system prompt & abstention logic
│
├── scripts/
│   ├── ingest_pdf.py           # Standalone ingestion & indexing pipeline
│   ├── evaluate.py             # Evaluator for the 5 LMS assignment questions
│   ├── generate_sample_catalogue.py # High-fidelity 80-page catalogue generator
│   └── generate_architecture_diagram.py # Architecture diagram generator
│
├── tests/
│   ├── test_pdf_loader.py      # Tests for PDF loading & SHA-256
│   ├── test_retriever.py       # Tests for chunking & ChromaDB retrieval
│   ├── test_citations.py       # Tests for citation & quote validation
│   └── test_not_found.py       # Tests for strict abstention & 'Not found'
│
├── chroma_db/                  # Persistent vector storage
│   ├── chroma.sqlite3
│   ├── index_meta.json
│   └── .gitkeep
│
├── screenshots/                # Real application screenshots
│   ├── 01_home.png
│   ├── 02_answer_with_citation.png
│   ├── 03_not_found.png
│   ├── 04_lms_questions.png
│   └── 05_architecture.png
│
└── submission/
    ├── lms_answers.md          # Markdown report of the 5 LMS answers
    └── test_results.md         # Automated test execution report
```

---

## 7. Installation & Setup Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/utkarshpatwa393/orient-blackswan-history-bot.git
cd orient-blackswan-history-bot
```

### Step 2: Create and Activate a Virtual Environment
```bash
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell):
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and set your credentials:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
GOOGLE_MODEL=gemini-3.5-flash-lite
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
CHROMA_PERSIST_DIRECTORY=./chroma_db
PDF_PATH=./data/History_2024-25.pdf
COLLECTION_NAME=orient_blackswan_history
RETRIEVER_K=4
```

> **Security Note:** `.env` is listed in `.gitignore`. Never commit API keys or private credentials to GitHub.

---

## 8. PDF Ingestion & Indexing

Place `History_2024-25.pdf` into the `data/` directory, then run:

```bash
python scripts/ingest_pdf.py
```

### Ingestion Output Example:
```text
=================================================================
  ORIENT BLACKSWAN HISTORY BOT - PDF INGESTION PIPELINE
=================================================================
Target PDF Path: data/History_2024-25.pdf
Chroma DB Path : ./chroma_db
Collection Name: orient_blackswan_history
Embedding Model: models/gemini-embedding-001
-----------------------------------------------------------------

[1/4] Extracting text and page metadata from PDF...
  • Total Pages Found    : 80
  • Processed Text Pages : 80
  • Empty/Skipped Pages  : 0
  • Document SHA-256     : 6b37eccecd5ec523...

[2/4] Splitting text into overlapping chunks...
  • Chunks Created       : 81
  • Chunk Size           : 1000 characters
  • Chunk Overlap        : 150 characters

[3/4] Indexing into persistent ChromaDB vector store...

[4/4] Ingestion Summary:
  • Status               : INDEXED
  • Details              : Successfully indexed 81 chunks across 80 pages.
  • Total Chunks Stored  : 81
=================================================================
  INGESTION COMPLETED SUCCESSFULLY
=================================================================
```

> **Smart Caching:** Re-running `python scripts/ingest_pdf.py` detects the matching SHA-256 file hash and reuses the index immediately without unnecessary API costs. To force re-indexing, pass the `--force` flag.

---

## 9. Running the Streamlit Application

Start the interactive web application with:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Key Features in the Interface:
- **Academic Editorial Theme:** Warm off-white background, deep navy typography, and burgundy accents inspired by classical book layouts.
- **Split-Screen Reader:** The left panel displays the actual PDF page text with navigation buttons (`◀ Prev`, `Next ▶`), page jumper, and evidence highlighter.
- **History Bot Chat:** The right panel displays the conversational assistant with online status, user messages, grounded responses, and clickable citation pill buttons (`📄 Page X ➔`).
- **One-Click Jump:** Clicking any citation pill automatically jumps the document reader to the cited page and highlights the evidence quote.
- **Suggested Question Pills:** Quick buttons for the 5 LMS assignment questions.

---

## 10. Automated Testing

Run the automated test suite with:

```bash
pytest tests/ -v
```

### Test Results:
```text
============================= test session starts ==============================
collected 10 items

tests/test_citations.py::test_extract_page_numbers PASSED                [ 10%]
tests/test_citations.py::test_verify_evidence_in_context PASSED          [ 20%]
tests/test_citations.py::test_format_citation PASSED                     [ 30%]
tests/test_not_found.py::test_empty_question_returns_not_found PASSED    [ 40%]
tests/test_not_found.py::test_out_of_domain_question_abstains PASSED     [ 50%]
tests/test_pdf_loader.py::test_pdf_exists_and_loads PASSED               [ 60%]
tests/test_pdf_loader.py::test_missing_pdf_handling PASSED               [ 70%]
tests/test_pdf_loader.py::test_file_hash_calculation PASSED              [ 80%]
tests/test_retriever.py::test_chunking_preserves_metadata PASSED         [ 90%]
tests/test_retriever.py::test_retriever_query_and_scores PASSED          [100%]

======================== 10 passed in 5.40s ========================
```

---

## 11. Application Screenshots

### 1. Application Home Screen
![Application Home Screen](screenshots/01_home.png)

### 2. Verified Answer with Page Citation
![Answer with Citation](screenshots/02_answer_with_citation.png)

### 3. Out-of-Domain Abstention (`Not found`)
![Not Found State](screenshots/03_not_found.png)

### 4. Multi-Turn Session with Assignment Questions
![LMS Questions Session](screenshots/04_lms_questions.png)

### 5. System RAG Architecture
![Architecture Diagram](screenshots/05_architecture.png)

---

## 12. Viva & Presentation Guide (For Engineering Students)

### Frequently Asked Questions in Viva:

**Q1: Why use RAG instead of directly asking Gemini?**  
*Answer:* General LLMs lack knowledge of private or proprietary internal documents (such as this 2024–25 publisher catalogue) and hallucinate prices, dates, or page numbers. RAG retrieves the exact text chunks from the vector database and provides them as ground truth in the prompt context.

**Q2: How does the system guarantee 1-based page numbers?**  
*Answer:* In `src/pdf_loader.py`, PyMuPDF indexes 0-based page indices (`page_idx`) and adds 1 (`page_num = page_idx + 1`). This metadata is preserved during chunking in `src/text_splitter.py` and saved directly in ChromaDB.

**Q3: How is hallucination prevented when a question is outside the document?**  
*Answer:* Multi-layer defense:
1. **Distance Thresholding:** In `src/retriever.py`, if retrieved chunks have large L2 distance or lack semantic overlap, the query is rejected early.
2. **Strict System Prompt:** Instructs Gemini to output *only* `Not found` if the answer is missing.
3. **Application Fallback:** In `src/rag_chain.py`, if the model output fails quotation verification or claims unsupported facts, the application overrides the output and returns `Not found`.

---

## 13. Git Setup & Push Instructions

To push this project to your GitHub repository:

```bash
# Initialize git if needed
git init

# Verify .gitignore protects .env and vector storage
git status

# Stage all project files
git add .

# Create initial commit
git commit -m "feat: complete Orient BlackSwan History Bot RAG application"

# Add your GitHub remote repository
git remote add origin https://github.com/<your-username>/orient-blackswan-history-bot.git
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## 14. License & Academic Attribution
Developed as an academic submission for the Generative AI Mid-Semester Examination (**Project 4 — Orient BlackSwan History Bot**).  
Document copyright belongs to **Orient BlackSwan Private Limited**.
