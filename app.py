"""
Orient BlackSwan History Bot - Polished Academic Editorial Interface
A document-grounded RAG research assistant built with LangChain, ChromaDB, and Google Gemini.
Adheres strictly to the light academic design system: zero black flashes, light sidebar, crystal-clear tabs.
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
# Streamlit Configuration & Comprehensive CSS Architecture
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

EDITORIAL_THEME_CSS = """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

/* Exact Design System Tokens */
:root {
    --bg-page: #F7F5F0;
    --surface-card: #FFFFFF;
    --surface-secondary: #F1EEE7;
    --primary-navy: #192C46;
    --primary-burgundy: #8D2947;
    --burgundy-hover: #742039;
    --burgundy-active: #651A32;
    --burgundy-pale: #FDF2F4;
    --burgundy-pale-border: #F5D5DE;
    --text-primary: #242B35;
    --text-secondary: #707783;
    --border-color: #E3DED5;
    --success-green: #287A57;
    --success-bg: #EBF5F0;
    --focus-ring: #C58A9D;
}

/* Global Reset */
*, *::before, *::after {
    -webkit-tap-highlight-color: transparent !important;
    box-sizing: border-box;
}

/* Background & Body */
body, .stApp {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Main Container Max-Width Alignment */
.main .block-container {
    max-width: 1140px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
}

/* ========================================================================= */
/* 1. SIDEBAR: FORCE LIGHT WARM IVORY BACKGROUND (NO DARK CHARCOAL)         */
/* ========================================================================= */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child,
div[data-testid="stSidebarContent"] {
    background-color: #F1EEE7 !important;
    background: #F1EEE7 !important;
    color: var(--text-primary) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] caption {
    color: var(--text-secondary) !important;
}

/* Sidebar Custom Cards */
.sidebar-section-card {
    background: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 0.95rem 1rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.03) !important;
}

.sidebar-section-title {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: var(--primary-navy) !important;
    margin-bottom: 0.65rem !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.sidebar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    padding: 0.28rem 0;
    border-bottom: 1px dashed #EDE8DE;
}

.sidebar-row:last-child {
    border-bottom: none;
}

.sidebar-row-label {
    color: var(--text-secondary);
}

.sidebar-row-value {
    font-weight: 600;
    color: var(--primary-navy);
}

/* ========================================================================= */
/* 2. TABS: FIX INVISIBLE TEXT (BOTH ACTIVE & INACTIVE CLEARLY READABLE)      */
/* ========================================================================= */
div[data-testid="stTabs"] {
    background: transparent !important;
    margin-bottom: 1.2rem !important;
}

div[data-baseweb="tab-list"] {
    background-color: #F1EEE7 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border-color) !important;
    gap: 6px !important;
}

/* Inactive Tab: Clear Dark Slate Text on Warm Ivory */
button[data-baseweb="tab"] {
    color: #4A5568 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    background-color: transparent !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.3rem !important;
    transition: all 160ms ease !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span {
    color: #4A5568 !important;
}

/* Inactive Tab Hover */
button[data-baseweb="tab"]:hover {
    color: var(--primary-burgundy) !important;
    background-color: #FFFFFF !important;
}

button[data-baseweb="tab"]:hover p,
button[data-baseweb="tab"]:hover span {
    color: var(--primary-burgundy) !important;
}

/* Active Tab: Bold White Text on Solid Burgundy */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--primary-burgundy) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(141, 41, 71, 0.25) !important;
}

button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span {
    color: #FFFFFF !important;
}

/* Remove default Streamlit red underline bar */
div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ========================================================================= */
/* 3. BUTTONS: COMPREHENSIVE ANTI-BLACK FIX ON CLICK / TAP / FOCUS          */
/* ========================================================================= */

/* Secondary / Standard Buttons (Card triggers, Navigation, Reset) */
.stButton > button,
button[kind="secondary"],
div[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 9px !important;
    padding: 0.52rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
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
    background-color: #F1EEE7 !important;
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
    background-color: #FFFFFF !important;
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
    padding: 0.6rem 1.25rem !important;
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

/* Input Fields */
div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 11px !important;
    transition: border-color 160ms ease, box-shadow 160ms ease !important;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.03) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: var(--primary-burgundy) !important;
    box-shadow: 0 0 0 3px rgba(197, 138, 157, 0.35) !important;
}

div[data-baseweb="input"] input {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
    background-color: transparent !important;
}

/* ========================================================================= */
/* 4. COMPACT & RESPONSIVE HEADER                                            */
/* ========================================================================= */
.compact-header {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.95rem 1.4rem;
    margin-bottom: 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(25, 44, 70, 0.03);
    flex-wrap: wrap;
    gap: 0.8rem;
}

.header-brand-wrap {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}

.header-emblem-badge {
    width: 42px;
    height: 42px;
    background: var(--primary-navy);
    color: #FFFFFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.header-title-text {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--primary-navy);
    line-height: 1.15;
    margin: 0;
}

.header-title-text span {
    color: var(--primary-burgundy);
}

.header-subtitle-line {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: 2px;
}

.header-right-side {
    display: flex;
    align-items: center;
    gap: 1.1rem;
}

.header-tagline-quote {
    font-family: 'Newsreader', Georgia, serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--primary-navy);
}

.ready-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--success-bg);
    color: var(--success-green);
    border: 1px solid #C4E2D3;
    padding: 0.25rem 0.65rem;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 700;
}

.ready-pill-dot {
    width: 7px;
    height: 7px;
    background: var(--success-green);
    border-radius: 50%;
}

/* ========================================================================= */
/* 5. WELCOME HERO & SUGGESTED QUESTIONS (NO DUPLICATE BUTTONS)              */
/* ========================================================================= */
.welcome-box {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 1px 4px rgba(25, 44, 70, 0.02);
}

.welcome-heading {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--primary-navy);
    margin-bottom: 0.4rem;
}

.welcome-summary {
    color: var(--text-secondary);
    font-size: 0.92rem;
    line-height: 1.55;
    margin-bottom: 0.4rem;
}

.suggested-section-title {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 1.1rem 0 0.7rem 0;
}

/* Suggested Question Card Container */
.question-card-outer {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.1rem 1.15rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.02);
    transition: all 160ms ease;
}

.question-card-outer:hover {
    border-color: var(--focus-ring);
    box-shadow: 0 3px 8px rgba(141, 41, 71, 0.08);
}

.question-card-tag {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--primary-burgundy);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
}

.question-card-query {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--primary-navy);
    line-height: 1.45;
    margin-bottom: 1rem;
    flex-grow: 1;
}

/* ========================================================================= */
/* 6. CHAT CONVERSATION MESSAGES & CITATION CARDS                            */
/* ========================================================================= */
.user-msg-row {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}

.user-msg-bubble {
    background: var(--burgundy-pale);
    color: var(--primary-navy);
    border: 1px solid var(--burgundy-pale-border);
    border-radius: 14px 14px 3px 14px;
    padding: 0.85rem 1.2rem;
    max-width: 82%;
    font-size: 0.94rem;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(25, 44, 70, 0.03);
}

.assistant-msg-row {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1.3rem;
}

.assistant-msg-bubble {
    background: #FFFFFF;
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: 14px 14px 14px 3px;
    padding: 1.2rem 1.4rem;
    max-width: 90%;
    box-shadow: 0 2px 6px rgba(25, 44, 70, 0.03);
}

.assistant-answer-content {
    font-size: 0.96rem;
    line-height: 1.65;
    color: var(--text-primary);
    margin-bottom: 0.85rem;
}

/* Grounded Citation Card */
.citation-panel {
    background: #F1EEE7;
    border-left: 4px solid var(--primary-burgundy);
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin-top: 0.85rem;
}

.citation-header-flex {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
    flex-wrap: wrap;
}

.citation-source-label {
    background: var(--primary-navy);
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 2px 7px;
    border-radius: 4px;
}

.citation-file-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--primary-navy);
}

.citation-page-badge {
    background: #F8EBF0;
    color: var(--primary-burgundy);
    border: 1px solid #E3BDCB;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
    margin-left: auto;
}

.citation-quote-italic {
    font-family: 'Newsreader', Georgia, serif;
    font-style: italic;
    font-size: 0.92rem;
    color: #4A5568;
    line-height: 1.5;
    margin-top: 0.35rem;
}

/* Neutral Not Found Panel */
.not-found-container {
    background: #F1EEE7;
    border: 1px solid var(--border-color);
    border-left: 4px solid #707783;
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.2rem;
    margin-top: 0.4rem;
}

.not-found-heading {
    font-weight: 700;
    color: var(--text-primary);
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.2rem;
}

.not-found-body {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.45;
}

/* ========================================================================= */
/* 7. DOCUMENT READER SHEET                                                  */
/* ========================================================================= */
.reader-paper-box {
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 2.2rem 2.5rem;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--text-primary);
    box-shadow: 0 2px 8px rgba(25, 44, 70, 0.03);
    min-height: 520px;
}

.reader-header-line {
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

.reader-highlight-callout {
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

st.markdown(EDITORIAL_THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""

if "active_question_to_run" not in st.session_state:
    st.session_state.active_question_to_run = None


# -----------------------------------------------------------------------------
# Backend RAG Pipeline Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to persistent vector store...")
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
# Header: Compact & Fully Responsive
# -----------------------------------------------------------------------------
vm_check = VectorStoreManager()
is_indexed = vm_check.is_indexed()
meta = vm_check.get_metadata()
total_doc_pages = int(meta.get("total_pages", 80))

st.markdown(
    f"""
    <div class="compact-header">
        <div class="header-brand-wrap">
            <div class="header-emblem-badge">📖</div>
            <div>
                <h1 class="header-title-text">Orient BlackSwan <span>History Bot</span></h1>
                <div class="header-subtitle-line">Grounded Academic AI Research Assistant</div>
            </div>
        </div>
        <div class="header-right-side">
            <div class="header-tagline-quote">Ask. Explore. Learn.</div>
            <div class="ready-pill">
                <span class="ready-pill-dot"></span>
                <span>{'Index Ready' if is_indexed else 'Indexing Required'}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Light Warm Ivory Theme with Clean White Cards
# -----------------------------------------------------------------------------
with st.sidebar:
    pdf_loader = PDFLoader(Config.DEFAULT_PDF_PATH)
    file_exists, _ = pdf_loader.verify_file()

    st.markdown(
        f"""
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">📚 Knowledge Source</div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Document</span>
                <span class="sidebar-row-value">History_2024-25.pdf</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Availability</span>
                <span class="sidebar-row-value">{'✅ Verified' if file_exists else '⚠️ Missing'}</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Indexed Pages</span>
                <span class="sidebar-row-value">{total_doc_pages} pages</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Vector Chunks</span>
                <span class="sidebar-row-value">{meta.get('chunk_count', 81)} chunks</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">⚙️ Index Management</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn_rebuild, col_btn_clear = st.columns(2)
    with col_btn_rebuild:
        if st.button("🔄 Rebuild", use_container_width=True, help="Re-index the PDF into ChromaDB"):
            with st.spinner("Re-indexing..."):
                docs, stats = pdf_loader.load()
                from src.text_splitter import DocumentChunker
                chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                chunks = chunker.split_documents(docs)
                vm_check.index_documents(chunks, stats, force_rebuild=True)
                st.cache_resource.clear()
                st.success("Index rebuilt!")
                st.rerun()

    with col_btn_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation messages"):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        f"""
        <div class="sidebar-section-card">
            <div class="sidebar-section-title">🛡️ Grounding & Model</div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Model</span>
                <span class="sidebar-row-value">{Config.GOOGLE_MODEL}</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Embeddings</span>
                <span class="sidebar-row-value">{Config.GOOGLE_EMBEDDING_MODEL.split('/')[-1]}</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">Vector Store</span>
                <span class="sidebar-row-value">ChromaDB</span>
            </div>
            <div class="sidebar-row">
                <span class="sidebar-row-label">API Key</span>
                <span class="sidebar-row-value">{Config.get_masked_api_key()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Orient BlackSwan History Bot • College Project")


# -----------------------------------------------------------------------------
# Main Tabs: High Contrast & Crystal Clear
# -----------------------------------------------------------------------------
tab_research, tab_catalogue = st.tabs(["💬 Research Assistant", "📖 Catalogue Document Reader"])

# Initialize RAG Pipeline
try:
    vm, rag_chain = get_rag_pipeline()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


# =============================================================================
# TAB 1: RESEARCH ASSISTANT (Chat & Welcome Hero)
# =============================================================================
with tab_research:
    # 1. Welcome Section (Displayed only before user starts asking questions)
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-box">
                <div class="welcome-heading">Your History Research Assistant</div>
                <div class="welcome-summary">
                    Ask questions about the <strong>Orient BlackSwan History Catalogue 2024–25</strong> and receive 
                    evidence-based answers with verified page references.
                </div>
            </div>
            <div class="suggested-section-title">SUGGESTED RESEARCH QUESTIONS:</div>
            """,
            unsafe_allow_html=True,
        )

        # 3 Clean Cards with EXACTLY ONE button each (No duplicate controls!)
        card_col1, card_col2, card_col3 = st.columns(3)
        with card_col1:
            st.markdown(
                """
                <div class="question-card-outer">
                    <div class="question-card-tag">Question 1</div>
                    <div class="question-card-query">When were the catalogue prices last updated?</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Ask question →", key="btn_q1_single", use_container_width=True):
                st.session_state.active_question_to_run = "On what date were the prices in this catalogue last updated?"
                st.rerun()

        with card_col2:
            st.markdown(
                """
                <div class="question-card-outer">
                    <div class="question-card-tag">Question 2</div>
                    <div class="question-card-query">In which year was Orient BlackSwan incorporated?</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Ask question →", key="btn_q2_single", use_container_width=True):
                st.session_state.active_question_to_run = "In which year was the publisher (Orient BlackSwan) incorporated?"
                st.rerun()

        with card_col3:
            st.markdown(
                """
                <div class="question-card-outer">
                    <div class="question-card-tag">Question 3</div>
                    <div class="question-card-query">Find a Physics textbook and its listed price.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Ask question →", key="btn_q3_single", use_container_width=True):
                st.session_state.active_question_to_run = "What is the price of a Physics textbook?"
                st.rerun()

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # 2. Chat Conversation History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-msg-row">
                    <div class="user-msg-bubble">{msg["content"]}</div>
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
                    <div class="assistant-msg-row">
                        <div class="assistant-msg-bubble">
                            <div class="not-found-container">
                                <div class="not-found-heading">
                                    <span>📄</span> Not found
                                </div>
                                <div class="not-found-body">
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
                    <div class="assistant-msg-row">
                        <div class="assistant-msg-bubble">
                            <div class="assistant-answer-content">{answer}</div>
                            <div class="citation-panel">
                                <div class="citation-header-flex">
                                    <span class="citation-source-label">Source</span>
                                    <span class="citation-file-title">History_2024-25.pdf</span>
                                    <span class="citation-page-badge">PDF Page {pages_str}</span>
                                </div>
                                <div class="citation-quote-italic">"{evidence}"</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if evidence:
                    with st.expander(f"🔎 Supporting Quotation Details (Page {pages_str})", expanded=False):
                        st.markdown(f"**Verbatim Extract from Document:**\n> {evidence}")
                        st.caption(f"Cited from {Config.DEFAULT_PDF_PATH.name}, Page {pages_str}")

    # 3. Question Input Form (Send button with paper plane icon)
    with st.form("chat_input_form", clear_on_submit=True):
        col_text, col_submit = st.columns([5, 1])
        with col_text:
            typed_input = st.text_input(
                "Ask a question about the catalogue…",
                placeholder="Ask a question about the catalogue…",
                label_visibility="collapsed",
                key="chat_text_box",
            )
        with col_submit:
            form_sent = st.form_submit_button("Send ➔", use_container_width=True)

    # Process query
    chosen_query = st.session_state.active_question_to_run or (typed_input if form_sent else None)
    if chosen_query:
        st.session_state.active_question_to_run = None
        clean_text = chosen_query.strip()
        if clean_text:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": clean_text})

            # Query RAG Chain
            with st.spinner("Retrieving document chunks and grounding answer..."):
                response = rag_chain.query(clean_text)

            # Auto-sync document viewer to cited page
            if response["is_found"] and response.get("pages"):
                st.session_state.current_page = response["pages"][0]
                st.session_state.highlight_text = response.get("evidence", "")

            # Append assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "data": response,
            })
            st.rerun()


# =============================================================================
# TAB 2: CATALOGUE DOCUMENT READER (Full academic page viewer)
# =============================================================================
with tab_catalogue:
    nav_col, stat_col = st.columns([3, 1])
    with nav_col:
        p_prev, p_select, p_next = st.columns([1, 2, 1])
        with p_prev:
            if st.button("◀ Previous Page", key="btn_reader_prev", use_container_width=True):
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
        with p_select:
            selected_pg = st.number_input(
                "Page",
                min_value=1,
                max_value=total_doc_pages,
                value=st.session_state.current_page,
                label_visibility="collapsed",
                key="reader_page_selector",
            )
            if selected_pg != st.session_state.current_page:
                st.session_state.current_page = selected_pg
                st.rerun()
        with p_next:
            if st.button("Next Page ▶", key="btn_reader_next", use_container_width=True):
                if st.session_state.current_page < total_doc_pages:
                    st.session_state.current_page += 1
                    st.rerun()

    with stat_col:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 0.88rem; font-weight: 600; color: #707783; padding-top: 0.4rem;">
                Page {st.session_state.current_page} of {total_doc_pages}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Read page text
    raw_page_text = pdf_loader.get_page_text(st.session_state.current_page)
    if not raw_page_text:
        raw_page_text = f"Orient BlackSwan History Catalogue 2024–25\n\nPage {st.session_state.current_page}\n\nScholarly monographs and university history textbooks."

    # Highlight quoted evidence if present
    highlight_str = st.session_state.get("highlight_text", "").strip()
    clean_hl = re.sub(r'["\']', '', highlight_str)

    if clean_hl and len(clean_hl) > 8 and clean_hl.lower() in raw_page_text.lower():
        start_idx = raw_page_text.lower().find(clean_hl.lower())
        end_idx = start_idx + len(clean_hl)
        before_txt = raw_page_text[:start_idx].replace('\n', '<br>')
        matched_txt = raw_page_text[start_idx:end_idx]
        after_txt = raw_page_text[end_idx:].replace('\n', '<br>')
        rendered_body = (
            f"{before_txt}<div class='reader-highlight-callout'>{matched_txt} "
            f"<span style='font-size:0.75rem; font-weight:700; color:#8D2947; margin-left:6px;'>[Cited Source]</span></div>{after_txt}"
        )
    else:
        rendered_body = raw_page_text.replace("\n", "<br>")

    # Render Book Paper Surface
    st.markdown(
        f"""
        <div class="reader-paper-box">
            <div class="reader-header-line">
                <span>ORIENT BLACKSWAN HISTORY CATALOGUE 2024–25</span>
                <span>PAGE {st.session_state.current_page} / {total_doc_pages}</span>
            </div>
            <div>
                {rendered_body}
            </div>
            <div style="border-top: 1px solid #E3DED5; padding-top: 1rem; margin-top: 2rem; text-align: center; font-family: 'Cinzel', serif; font-size: 0.88rem; color: #707783;">
                — {st.session_state.current_page} —
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
