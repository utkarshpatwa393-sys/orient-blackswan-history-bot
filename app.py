"""
Orient BlackSwan History Bot - Redesigned Academic Editorial Interface
A document-grounded RAG research assistant built with LangChain, ChromaDB, and Google Gemini.
Adheres strictly to the Orient BlackSwan Editorial Design System.
"""

import sys
import re
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.config import Config
from src.pdf_loader import PDFLoader
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.rag_chain import RAGChain
from src.citations import CitationManager


# -----------------------------------------------------------------------------
# Page Configuration & Editorial CSS Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

EDITORIAL_CSS = """
<style>
/* Import Classical & Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

/* Exact Design System Tokens */
:root {
    --bg-main: #F7F5F0;
    --surface-main: #FFFFFF;
    --surface-secondary: #F1EEE7;
    --primary-navy: #192C46;
    --primary-burgundy: #8D2947;
    --burgundy-hover: #742039;
    --burgundy-active: #651A32;
    --text-main: #242B35;
    --text-secondary: #707783;
    --border-color: #E3DED5;
    --success-green: #287A57;
    --error-red: #B93838;
    --focus-ring: #C58A9D;
}

/* Global Reset & Prevent Mobile Tap Flashes */
*, *::before, *::after {
    -webkit-tap-highlight-color: transparent !important;
    box-sizing: border-box;
}

body, .stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Main Container Max-Width Alignment */
.main .block-container {
    max-width: 1140px !important;
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
}

/* ========================================================================= */
/* CRITICAL FIX: EXPLICIT INTERACTION STATES FOR ALL BUTTONS (NO BLACK FLASH)*/
/* ========================================================================= */

/* General / Secondary Buttons */
.stButton > button,
button[kind="secondary"],
div[data-testid="stBaseButton-secondary"] {
    background-color: var(--surface-main) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    line-height: 1.4 !important;
    box-shadow: 0 1px 2px rgba(25, 44, 70, 0.04) !important;
    transition: background-color 160ms ease,
                border-color 160ms ease,
                color 160ms ease,
                box-shadow 160ms ease,
                transform 160ms ease !important;
    outline: none !important;
}

.stButton > button:hover,
button[kind="secondary"]:hover,
div[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--surface-secondary) !important;
    color: var(--primary-burgundy) !important;
    border-color: var(--focus-ring) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 6px rgba(141, 41, 71, 0.08) !important;
}

.stButton > button:active,
button[kind="secondary"]:active,
div[data-testid="stBaseButton-secondary"]:active {
    background-color: #E6E1D7 !important;
    color: var(--primary-burgundy) !important;
    border-color: var(--primary-burgundy) !important;
    transform: translateY(0px) !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.06) !important;
}

.stButton > button:focus,
.stButton > button:focus-visible,
button[kind="secondary"]:focus,
button[kind="secondary"]:focus-visible,
div[data-testid="stBaseButton-secondary"]:focus,
div[data-testid="stBaseButton-secondary"]:focus-visible {
    background-color: var(--surface-main) !important;
    color: var(--primary-burgundy) !important;
    outline: 3px solid var(--focus-ring) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 2px #FFFFFF !important;
}

/* Primary Action Buttons (Send / Form Submit) */
div[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
div[data-testid="stBaseButton-primary"] {
    background-color: var(--primary-burgundy) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--primary-burgundy) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.3rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 2px 5px rgba(141, 41, 71, 0.25) !important;
    transition: background-color 160ms ease,
                border-color 160ms ease,
                box-shadow 160ms ease,
                transform 160ms ease !important;
    outline: none !important;
}

div[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primary"]:hover {
    background-color: var(--burgundy-hover) !important;
    color: #FFFFFF !important;
    border-color: var(--burgundy-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(141, 41, 71, 0.35) !important;
}

div[data-testid="stFormSubmitButton"] > button:active,
button[kind="primary"]:active {
    background-color: var(--burgundy-active) !important;
    color: #FFFFFF !important;
    border-color: var(--burgundy-active) !important;
    transform: translateY(0px) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
}

div[data-testid="stFormSubmitButton"] > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus-visible,
button[kind="primary"]:focus,
button[kind="primary"]:focus-visible {
    background-color: var(--primary-burgundy) !important;
    color: #FFFFFF !important;
    outline: 3px solid var(--focus-ring) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 2px #FFFFFF !important;
}

/* Disabled State */
button:disabled,
button[disabled] {
    background-color: var(--surface-secondary) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-color) !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Form Input Field Styling */
div[data-baseweb="input"] {
    background-color: var(--surface-main) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    transition: border-color 160ms ease, box-shadow 160ms ease !important;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.03) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: var(--primary-burgundy) !important;
    box-shadow: 0 0 0 3px rgba(197, 138, 157, 0.35) !important;
}

div[data-baseweb="input"] input {
    color: var(--text-main) !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
    background-color: transparent !important;
}

/* ========================================================================= */
/* PREMIUM HEADER                                                            */
/* ========================================================================= */
.editorial-header {
    background: var(--surface-main);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.25rem 1.8rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(25, 44, 70, 0.03);
}

.header-brand-block {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.header-emblem {
    width: 48px;
    height: 48px;
    background: var(--primary-navy);
    color: #FFFFFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 3px 8px rgba(25, 44, 70, 0.15);
}

.header-title-main {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--primary-navy);
    line-height: 1.15;
    margin: 0;
    letter-spacing: 0.5px;
}

.header-title-main span {
    color: var(--primary-burgundy);
}

.header-subtitle-text {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-top: 3px;
}

.header-status-side {
    text-align: right;
}

.header-tagline-text {
    font-family: 'Newsreader', Georgia, serif;
    font-style: italic;
    font-size: 1.1rem;
    color: var(--primary-navy);
    margin-bottom: 4px;
}

.header-badge-ready {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #EBF5F0;
    color: var(--success-green);
    border: 1px solid #C4E2D3;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.header-badge-dot {
    width: 7px;
    height: 7px;
    background: var(--success-green);
    border-radius: 50%;
}

/* ========================================================================= */
/* WELCOME SECTION                                                           */
/* ========================================================================= */
.welcome-hero-card {
    background: var(--surface-main);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(25, 44, 70, 0.02);
}

.welcome-hero-title {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--primary-navy);
    margin-bottom: 0.5rem;
}

.welcome-hero-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 1.6rem;
    max-width: 820px;
}

.suggested-label-small {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.8rem;
}

/* Suggested Question Cards */
.suggested-card {
    background: var(--surface-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    transition: all 160ms ease;
    cursor: pointer;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.suggested-card:hover {
    background: #FAF8F5;
    border-color: var(--focus-ring);
    box-shadow: 0 4px 10px rgba(141, 41, 71, 0.08);
    transform: translateY(-2px);
}

.suggested-card-num {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--primary-burgundy);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.suggested-card-text {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--primary-navy);
    line-height: 1.45;
}

.suggested-card-arrow {
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: var(--primary-burgundy);
    font-weight: 600;
}

/* ========================================================================= */
/* CHAT MESSAGES & CONVERSATION                                              */
/* ========================================================================= */
.chat-conversation-area {
    margin-bottom: 1.5rem;
}

.bubble-user-row {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.2rem;
}

.bubble-user-card {
    background: var(--surface-secondary);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    border-radius: 14px 14px 2px 14px;
    padding: 0.9rem 1.25rem;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.03);
}

.bubble-assistant-row {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1.4rem;
}

.bubble-assistant-card {
    background: var(--surface-main);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    border-radius: 14px 14px 14px 2px;
    padding: 1.3rem 1.5rem;
    max-width: 90%;
    box-shadow: 0 2px 8px rgba(25, 44, 70, 0.03);
}

.assistant-answer-text {
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--text-main);
    margin-bottom: 0.9rem;
}

/* ========================================================================= */
/* CITATION CARD (SUPPORTED ANSWER)                                          */
/* ========================================================================= */
.citation-box-card {
    background: #FAF8F5;
    border-left: 4px solid var(--primary-burgundy);
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin-top: 0.85rem;
}

.citation-header-line {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}

.citation-badge-source {
    background: var(--primary-navy);
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 2px 7px;
    border-radius: 4px;
}

.citation-meta-file {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--primary-navy);
}

.citation-page-pill {
    background: #F8EBF0;
    color: var(--primary-burgundy);
    border: 1px solid #E3BDCB;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
    margin-left: auto;
}

.citation-quote-preview {
    font-family: 'Newsreader', Georgia, serif;
    font-style: italic;
    font-size: 0.92rem;
    color: #4A5568;
    line-height: 1.5;
    margin-top: 0.35rem;
}

/* ========================================================================= */
/* NOT FOUND PANEL                                                           */
/* ========================================================================= */
.not-found-panel {
    background: var(--surface-secondary);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--text-secondary);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}

.not-found-status-title {
    font-weight: 700;
    color: var(--text-main);
    font-size: 0.92rem;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.not-found-explanation {
    color: var(--text-secondary);
    font-size: 0.86rem;
    line-height: 1.5;
}

/* ========================================================================= */
/* SIDEBAR METRICS & STYLING                                                 */
/* ========================================================================= */
.sidebar-panel-box {
    background: var(--surface-main);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-panel-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--primary-navy);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.sidebar-stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.83rem;
    padding: 0.3rem 0;
    border-bottom: 1px dashed #EDE8DE;
}

.sidebar-stat-item:last-child {
    border-bottom: none;
}

.sidebar-stat-key {
    color: var(--text-secondary);
}

.sidebar-stat-val {
    font-weight: 600;
    color: var(--primary-navy);
}

/* Academic Book Document Reader Surface */
.book-paper-sheet {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 2.2rem 2.6rem;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #242B35;
    box-shadow: 0 3px 12px rgba(25, 44, 70, 0.04);
    min-height: 520px;
}

.book-paper-header {
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
}

.book-paper-footer {
    border-top: 1px solid var(--border-color);
    padding-top: 1rem;
    margin-top: 2rem;
    text-align: center;
    font-family: 'Cinzel', serif;
    font-size: 0.88rem;
    color: var(--text-secondary);
}

.book-evidence-highlight {
    background-color: #F8EBF0;
    border-left: 3px solid var(--primary-burgundy);
    padding: 0.8rem 1rem;
    margin: 1rem 0;
    border-radius: 0 6px 6px 0;
    color: var(--primary-burgundy);
    font-weight: 500;
}
</style>
"""

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""

if "selected_query_to_run" not in st.session_state:
    st.session_state.selected_query_to_run = None


# -----------------------------------------------------------------------------
# Backend RAG Pipeline Loader (Preserved intact)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Orient BlackSwan vector database...")
def get_rag_pipeline():
    """Initializes and caches the persistent vector store and grounded RAG chain."""
    vm = VectorStoreManager()
    if not vm.is_indexed():
        loader = PDFLoader(Config.DEFAULT_PDF_PATH)
        valid, _ = loader.verify_file()
        if valid:
            docs, stats = loader.load()
            from src.text_splitter import DocumentChunker
            chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
            chunks = chunker.split_documents(docs)
            vm.index_documents(chunks, stats, force_rebuild=False)

    vs = vm.get_vector_store()
    retriever = DocumentRetriever(vs, k=Config.RETRIEVER_K)
    chain = RAGChain(retriever)
    return vm, chain


# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
vm_check = VectorStoreManager()
is_indexed = vm_check.is_indexed()
meta = vm_check.get_metadata()
total_doc_pages = int(meta.get("total_pages", 80))

st.markdown(
    f"""
    <div class="editorial-header">
        <div class="header-brand-block">
            <div class="header-emblem">📖</div>
            <div>
                <h1 class="header-title-main">Orient BlackSwan <span>History Bot</span></h1>
                <div class="header-subtitle-text">Grounded Academic AI Research Assistant</div>
            </div>
        </div>
        <div class="header-status-side">
            <div class="header-tagline-text">Ask. Explore. Learn.</div>
            <div class="header-badge-ready">
                <span class="header-badge-dot"></span>
                <span>{'Index Ready' if is_indexed else 'Indexing Required'}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Knowledge Source, Index Management & About
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-panel-box">
            <div class="sidebar-panel-title">📚 Knowledge Source</div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Document</span>
                <span class="sidebar-stat-val">History_2024-25.pdf</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Scope</span>
                <span class="sidebar-stat-val">History Catalogue</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Indexed Pages</span>
                <span class="sidebar-stat-val">""" + str(total_doc_pages) + """ pp</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Total Chunks</span>
                <span class="sidebar-stat-val">""" + str(meta.get("chunk_count", 81)) + """</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_loader = PDFLoader(Config.DEFAULT_PDF_PATH)
    file_exists, _ = pdf_loader.verify_file()

    st.markdown(
        """
        <div class="sidebar-panel-box">
            <div class="sidebar-panel-title">⚙️ Index Management</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        if st.button("🔄 Rebuild Index", use_container_width=True, help="Rebuild vector index from scratch"):
            with st.spinner("Rebuilding persistent ChromaDB vector store..."):
                docs, stats = pdf_loader.load()
                from src.text_splitter import DocumentChunker
                chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                chunks = chunker.split_documents(docs)
                vm_check.index_documents(chunks, stats, force_rebuild=True)
                st.cache_resource.clear()
                st.success("Rebuilt index!")
                st.rerun()

    with col_idx2:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Reset the conversation history"):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        f"""
        <div class="sidebar-panel-box">
            <div class="sidebar-panel-title">🛡️ Grounding & Model</div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">LLM Model</span>
                <span class="sidebar-stat-val">{Config.GOOGLE_MODEL}</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Embeddings</span>
                <span class="sidebar-stat-val">{Config.GOOGLE_EMBEDDING_MODEL.split('/')[-1]}</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">Vector DB</span>
                <span class="sidebar-stat-val">ChromaDB</span>
            </div>
            <div class="sidebar-stat-item">
                <span class="sidebar-stat-key">API Key</span>
                <span class="sidebar-stat-val">{Config.get_masked_api_key()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Orient BlackSwan History Bot • College Project")


# -----------------------------------------------------------------------------
# Main Application Tabs: Research Assistant & Catalogue Reader
# -----------------------------------------------------------------------------
tab_chat, tab_reader = st.tabs(["💬 Research Assistant", "📖 Catalogue Document Reader"])

# Initialize RAG Pipeline
try:
    vm, rag_chain = get_rag_pipeline()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


# =============================================================================
# TAB 1: RESEARCH ASSISTANT (Centered, spacious, premium chat)
# =============================================================================
with tab_chat:
    # 1. Welcome Section (Rendered when no conversation has taken place)
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-hero-card">
                <div class="welcome-hero-title">Your History Research Assistant</div>
                <div class="welcome-hero-desc">
                    Ask questions about the <strong>Orient BlackSwan History Catalogue 2024–25</strong> and receive 
                    answers grounded exclusively in the document, with verified page references and supporting quotations.
                </div>
                <div class="suggested-label-small">SUGGESTED RESEARCH QUESTIONS:</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3 Elegant Clickable Suggested Question Cards
        sugg_col1, sugg_col2, sugg_col3 = st.columns(3)
        with sugg_col1:
            st.markdown(
                """
                <div class="suggested-card">
                    <div>
                        <div class="suggested-card-num">Question 1</div>
                        <div class="suggested-card-text">When were the catalogue prices last updated?</div>
                    </div>
                    <div class="suggested-card-arrow">Ask Question ➔</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Submit Q1", key="btn_q1_hero", use_container_width=True):
                st.session_state.selected_query_to_run = "On what date were the prices in this catalogue last updated?"
                st.rerun()

        with sugg_col2:
            st.markdown(
                """
                <div class="suggested-card">
                    <div>
                        <div class="suggested-card-num">Question 2</div>
                        <div class="suggested-card-text">In which year was Orient BlackSwan incorporated?</div>
                    </div>
                    <div class="suggested-card-arrow">Ask Question ➔</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Submit Q2", key="btn_q2_hero", use_container_width=True):
                st.session_state.selected_query_to_run = "In which year was the publisher (Orient BlackSwan) incorporated?"
                st.rerun()

        with sugg_col3:
            st.markdown(
                """
                <div class="suggested-card">
                    <div>
                        <div class="suggested-card-num">Question 3</div>
                        <div class="suggested-card-text">Find a Physics textbook and its listed price.</div>
                    </div>
                    <div class="suggested-card-arrow">Ask Question ➔</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Submit Q3", key="btn_q3_hero", use_container_width=True):
                st.session_state.selected_query_to_run = "What is the price of a Physics textbook?"
                st.rerun()

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Chat Conversation History
    st.markdown('<div class="chat-conversation-area">', unsafe_allow_html=True)
    for msg_idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="bubble-user-row">
                    <div class="bubble-user-card">{msg["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            res = msg.get("data", {})
            is_found = res.get("is_found", False)
            answer = res.get("answer", msg.get("content", ""))
            pages = res.get("pages", [])
            evidence = res.get("evidence", "")

            # If Answer is 'Not found'
            if not is_found or answer.strip().lower() in {"not found", "not found."}:
                st.markdown(
                    """
                    <div class="bubble-assistant-row">
                        <div class="bubble-assistant-card">
                            <div class="not-found-panel">
                                <div class="not-found-status-title">
                                    <span>📄</span> Not found
                                </div>
                                <div class="not-found-explanation">
                                    The catalogue does not provide sufficient evidence to answer this question.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                pages_str = ", ".join(str(p) for p in pages) if pages else "1"
                st.markdown(
                    f"""
                    <div class="bubble-assistant-row">
                        <div class="bubble-assistant-card">
                            <div class="assistant-answer-text">{answer}</div>
                            <div class="citation-box-card">
                                <div class="citation-header-line">
                                    <span class="citation-badge-source">Source</span>
                                    <span class="citation-meta-file">History_2024-25.pdf</span>
                                    <span class="citation-page-pill">PDF Page {pages_str}</span>
                                </div>
                                <div class="citation-quote-preview">"{evidence}"</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if evidence:
                    with st.expander(f"🔎 Full Verified Evidence Details (Page {pages_str})", expanded=False):
                        st.markdown(f"**Exact Quotation from Retrieved Context:**\n> {evidence}")
                        st.caption(f"Verified against local vector chunks on PDF Page {pages_str}")

    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Question Entry Form (Always positioned cleanly at the bottom)
    with st.form("main_qa_form", clear_on_submit=True):
        f_col1, f_col2 = st.columns([5, 1])
        with f_col1:
            typed_query = st.text_input(
                "Ask a question about the catalogue...",
                placeholder="Ask a question about the catalogue…",
                label_visibility="collapsed",
                key="input_question_box",
            )
        with f_col2:
            send_btn = st.form_submit_button("Send ➔", use_container_width=True)

    # Execute pending query
    query_to_process = st.session_state.selected_query_to_run or (typed_query if send_btn else None)
    if query_to_process:
        st.session_state.selected_query_to_run = None
        clean_q = query_to_process.strip()
        if clean_q:
            # 1. Add user query to chat history
            st.session_state.messages.append({"role": "user", "content": clean_q})

            # 2. Invoke grounded RAG chain
            with st.spinner("Retrieving document chunks and verifying citations..."):
                response = rag_chain.query(clean_q)

            # 3. Auto-sync Document Reader to cited page
            if response["is_found"] and response.get("pages"):
                first_page = response["pages"][0]
                st.session_state.current_page = first_page
                st.session_state.highlight_text = response.get("evidence", "")

            # 4. Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "data": response,
            })
            st.rerun()


# =============================================================================
# TAB 2: CATALOGUE DOCUMENT READER (Full academic page viewer)
# =============================================================================
with tab_reader:
    r_col_nav, r_col_stat = st.columns([3, 1])
    with r_col_nav:
        n_col1, n_col2, n_col3 = st.columns([1, 2, 1])
        with n_col1:
            if st.button("◀ Previous Page", key="btn_prev_pg", use_container_width=True):
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
        with n_col2:
            page_val = st.number_input(
                "Jump to Page",
                min_value=1,
                max_value=total_doc_pages,
                value=st.session_state.current_page,
                label_visibility="collapsed",
                key="page_number_input",
            )
            if page_val != st.session_state.current_page:
                st.session_state.current_page = page_val
                st.rerun()
        with n_col3:
            if st.button("Next Page ▶", key="btn_next_pg", use_container_width=True):
                if st.session_state.current_page < total_doc_pages:
                    st.session_state.current_page += 1
                    st.rerun()

    with r_col_stat:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 0.88rem; font-weight: 600; color: #707783; padding-top: 0.4rem;">
                Page {st.session_state.current_page} of {total_doc_pages}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Read text from current page
    p_text = pdf_loader.get_page_text(st.session_state.current_page)
    if not p_text:
        p_text = f"Orient BlackSwan History Catalogue 2024–25\n\nPage {st.session_state.current_page}\n\nScholarly monographs and university history textbooks."

    # Highlight quoted evidence if present
    hl_text = st.session_state.get("highlight_text", "").strip()
    hl_clean = re.sub(r'["\']', '', hl_text)

    if hl_clean and len(hl_clean) > 8 and hl_clean.lower() in p_text.lower():
        start_pos = p_text.lower().find(hl_clean.lower())
        end_pos = start_pos + len(hl_clean)
        before = p_text[:start_pos].replace('\n', '<br>')
        match = p_text[start_pos:end_pos]
        after = p_text[end_pos:].replace('\n', '<br>')
        rendered_page_content = (
            f"{before}<div class='book-evidence-highlight'>{match} "
            f"<span style='font-size:0.75rem; font-weight:700; color:#8D2947; margin-left:6px;'>[Cited Source]</span></div>{after}"
        )
    else:
        rendered_page_content = p_text.replace("\n", "<br>")

    # Render Book Paper Surface
    st.markdown(
        f"""
        <div class="book-paper-sheet">
            <div class="book-paper-header">
                <span>ORIENT BLACKSWAN HISTORY CATALOGUE 2024–25</span>
                <span>PAGE {st.session_state.current_page} / {total_doc_pages}</span>
            </div>
            <div>
                {rendered_page_content}
            </div>
            <div class="book-paper-footer">
                — {st.session_state.current_page} —
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
