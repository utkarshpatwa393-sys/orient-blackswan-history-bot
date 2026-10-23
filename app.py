"""
Orient BlackSwan History Bot - Streamlit Application
A document-grounded RAG research assistant built with LangChain, ChromaDB, and Google Gemini.
Features an academic editorial UI matching the reference design.
"""

import sys
import os
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
# Streamlit Page Setup & Custom Editorial Academic CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="🦢",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@400;500;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap');

/* Academic Warm Off-White Color System */
:root {
    --bg-page: #fbf9f4;
    --bg-card: #ffffff;
    --bg-paper: #fefdfa;
    --text-navy: #141e30;
    --text-muted: #64748b;
    --burgundy: #881337;
    --burgundy-hover: #9f1239;
    --burgundy-active: #701a2e;
    --burgundy-light: #fbebf0;
    --burgundy-border: #e2a8b9;
    --border-card: #e5e0d4;
    --border-paper: #dcd6c8;
}

/* Global Reset */
* {
    -webkit-tap-highlight-color: transparent !important;
}

body, .stApp {
    background-color: var(--bg-page) !important;
    color: var(--text-navy) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Header Banner */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.1rem 1.8rem;
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 14px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(20, 30, 48, 0.03);
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.brand-logo {
    width: 52px;
    height: 52px;
    background: #141e30;
    color: #ffffff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 4px 12px rgba(20, 30, 48, 0.15);
}

.brand-title {
    font-family: 'Cinzel', Georgia, serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-navy);
    margin: 0;
    line-height: 1.1;
    letter-spacing: 0.5px;
}

.brand-title span {
    color: var(--burgundy);
}

.brand-subtitle {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-top: 4px;
}

.header-tagline {
    text-align: right;
    font-family: 'Merriweather', Georgia, serif;
    font-style: italic;
    font-size: 1.15rem;
    color: var(--text-navy);
}

.header-tagline-bar {
    width: 65px;
    height: 3px;
    background: var(--burgundy);
    margin-left: auto;
    margin-top: 4px;
    border-radius: 2px;
}

/* Workflow Diagram */
.diagram-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.8rem;
    padding: 0.75rem 1rem;
    background: #ffffff;
    border: 1px dashed var(--border-card);
    border-radius: 12px;
    margin-bottom: 1.2rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}

.diagram-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-weight: 600;
    color: var(--text-navy);
}

.diagram-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: #f5f0e6;
    color: var(--burgundy);
    border-radius: 8px;
    font-size: 16px;
}

/* ========================================================================= */
/* BUTTONS: ELIMINATE BLACK TAP / FLASH ON FOCUS AND ACTIVE STATES          */
/* ========================================================================= */
.stButton > button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease-in-out !important;
    border: 1px solid var(--border-card) !important;
    background-color: #ffffff !important;
    color: var(--text-navy) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    outline: none !important;
}

/* Hover: soft pink/burgundy tint */
.stButton > button:hover {
    background-color: var(--burgundy-light) !important;
    border-color: var(--burgundy-border) !important;
    color: var(--burgundy) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 8px rgba(136, 19, 55, 0.08) !important;
}

/* ACTIVE / CLICKED / TAPPED / FOCUSED: NEVER TURNS BLACK! */
.stButton > button:active,
.stButton > button:focus,
.stButton > button:focus-visible,
.stButton > button:focus:not(:focus-visible) {
    background-color: #fae1e7 !important;
    border-color: var(--burgundy) !important;
    color: var(--burgundy) !important;
    box-shadow: 0 0 0 2px rgba(136, 19, 55, 0.2) !important;
    outline: none !important;
    transform: translateY(0px) !important;
}

/* Primary Form Submit Button (Send) */
div[data-testid="stFormSubmitButton"] > button {
    background: var(--burgundy) !important;
    color: #ffffff !important;
    border: 1px solid var(--burgundy) !important;
    border-radius: 22px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(136, 19, 55, 0.25) !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--burgundy-hover) !important;
    color: #ffffff !important;
    border-color: var(--burgundy-hover) !important;
    box-shadow: 0 4px 12px rgba(136, 19, 55, 0.35) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stFormSubmitButton"] > button:active,
div[data-testid="stFormSubmitButton"] > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus-visible {
    background: var(--burgundy-active) !important;
    color: #ffffff !important;
    border-color: var(--burgundy-active) !important;
    box-shadow: 0 0 0 2px rgba(136, 19, 55, 0.3) !important;
    outline: none !important;
}

/* Input text fields styling */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border-radius: 24px !important;
    border: 1px solid var(--border-paper) !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: var(--burgundy) !important;
    box-shadow: 0 0 0 2px rgba(136, 19, 55, 0.15) !important;
}

div[data-baseweb="input"] input {
    color: var(--text-navy) !important;
    font-size: 0.92rem !important;
    padding-left: 14px !important;
}

/* Number input for page jumping */
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    border-radius: 12px !important;
}

/* ========================================================================= */
/* LEFT COLUMN: DOCUMENT VIEWER CARD                                        */
/* ========================================================================= */
.doc-card {
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 1.2rem;
    height: 740px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}

.doc-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 0.75rem;
}

.doc-header-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-navy);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.doc-header-meta {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
}

.doc-viewer-wrapper {
    display: flex;
    gap: 0.8rem;
    flex-grow: 1;
    overflow: hidden;
}

/* Left thumbnail strip */
.thumbnail-strip {
    width: 62px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
    padding-right: 4px;
}

.thumb-item {
    width: 52px;
    height: 68px;
    background: #fdfdfc;
    border: 1.5px solid #dcd7cc;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
}

.thumb-item.active {
    border: 2px solid var(--burgundy) !important;
    background: #fdf0f3 !important;
    color: var(--burgundy) !important;
    box-shadow: 0 2px 6px rgba(136, 19, 55, 0.15) !important;
}

/* Main Book Page Surface */
.book-page-canvas {
    flex-grow: 1;
    background: var(--bg-paper);
    border: 1px solid var(--border-paper);
    border-radius: 10px;
    padding: 1.8rem 2rem;
    overflow-y: auto;
    font-family: 'Merriweather', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #242e42;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.01), 0 3px 12px rgba(0,0,0,0.03);
    position: relative;
}

.book-page-number {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #eeebe2;
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Highlighted Quote in Book */
.book-highlight-box {
    background-color: var(--burgundy-light);
    border-left: 4px solid var(--burgundy);
    padding: 0.8rem 1rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
    color: var(--burgundy);
    font-weight: 500;
    position: relative;
}

.badge-page-inline {
    background: var(--burgundy);
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: 6px;
    display: inline-block;
}

/* ========================================================================= */
/* RIGHT COLUMN: HISTORY BOT CHAT CARD                                      */
/* ========================================================================= */
.chat-card {
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 1.2rem;
    height: 740px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}

.chat-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 0.75rem;
}

.chat-header-title {
    font-weight: 700;
    font-size: 1.05rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--text-navy);
}

.status-online {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #15803d;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
}

/* Chat Message Bubbles */
.bubble-user {
    background: #edf2f7;
    color: var(--text-navy);
    padding: 0.85rem 1.15rem;
    border-radius: 16px 16px 4px 16px;
    margin-left: 2.2rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    line-height: 1.5;
    border: 1px solid #dbe2ec;
}

.bubble-bot {
    background: #fdfcf9;
    color: var(--text-navy);
    padding: 1.1rem 1.25rem;
    border-radius: 16px 16px 16px 4px;
    margin-right: 2.2rem;
    margin-bottom: 1.2rem;
    border: 1px solid #eae4d8;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

.bubble-bot-answer {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-navy);
    margin-bottom: 0.6rem;
}

.citation-pill-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--burgundy-light);
    color: var(--burgundy);
    border: 1px solid var(--burgundy-border);
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-top: 0.4rem;
}

/* Not Found Alert (Matching Mockup Image 2) */
.not-found-card {
    background: #fef4f4;
    border: 1px solid #fed7d7;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

.not-found-icon {
    font-size: 1.3rem;
    color: #991b1b;
    margin-top: 2px;
}

.not-found-title {
    font-weight: 700;
    color: #991b1b;
    font-size: 0.9rem;
}

.not-found-desc {
    color: #7f1d1d;
    font-size: 0.82rem;
    margin-top: 3px;
    line-height: 1.4;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""


# -----------------------------------------------------------------------------
# Cached Backend Pipeline Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Orient BlackSwan vector database...")
def get_rag_pipeline():
    """Initializes and caches vector store and RAG chain."""
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
st.markdown(
    """
    <div class="header-container">
        <div class="brand-wrapper">
            <div class="brand-logo">🦢</div>
            <div>
                <h1 class="brand-title">Orient BlackSwan <span>History Bot</span></h1>
                <div class="brand-subtitle">YOUR DOCUMENT-GROUNDED AI RESEARCH ASSISTANT</div>
            </div>
        </div>
        <div>
            <div class="header-tagline">Ask. Explore. Learn.</div>
            <div class="header-tagline-bar"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Workflow Diagram Bar
st.markdown(
    """
    <div class="diagram-container">
        <div class="diagram-step">
            <span class="diagram-icon">📄</span>
            <span>Your Book (PDF)</span>
        </div>
        <span>➔</span>
        <div class="diagram-step">
            <span class="diagram-icon">💠</span>
            <span>Vector Database (ChromaDB)</span>
        </div>
        <span>➔</span>
        <div class="diagram-step">
            <span class="diagram-icon">🧠</span>
            <span>Contextual Answer (with Citations)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Settings & Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Catalogue Knowledge Base")
    pdf_path = Config.DEFAULT_PDF_PATH
    loader = PDFLoader(pdf_path)
    file_exists, _ = loader.verify_file()

    if file_exists:
        st.success(f"**Loaded:** `{pdf_path.name}`")
    else:
        st.error(f"**Missing:** `{pdf_path.name}`")
        st.caption("Place `History_2024-25.pdf` into `data/`.")

    vm_check = VectorStoreManager()
    is_indexed = vm_check.is_indexed()
    meta = vm_check.get_metadata()

    st.markdown("---")
    st.markdown("#### ⚙️ Index Statistics")
    st.write(f"• **Index Status:** {'✅ Ready' if is_indexed else '⚠️ Not Indexed'}")
    st.write(f"• **Total Pages:** {meta.get('total_pages', 80)}")
    st.write(f"• **Indexed Chunks:** {meta.get('chunk_count', 81)}")
    st.write(f"• **Collection:** `{Config.COLLECTION_NAME}`")
    st.write(f"• **Embedding Model:** `{Config.GOOGLE_EMBEDDING_MODEL.split('/')[-1]}`")
    st.write(f"• **Generation Model:** `{Config.GOOGLE_MODEL}`")
    st.write(f"• **API Key:** `{Config.get_masked_api_key()}`")

    st.markdown("---")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Rebuild", use_container_width=True, help="Re-index vector database"):
            with st.spinner("Re-indexing..."):
                docs, stats = loader.load()
                from src.text_splitter import DocumentChunker
                chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                chunks = chunker.split_documents(docs)
                vm_check.index_documents(chunks, stats, force_rebuild=True)
                st.cache_resource.clear()
                st.rerun()

    with col_sb2:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Reset conversation"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Orient BlackSwan History Bot • RAG Architecture")


# -----------------------------------------------------------------------------
# Main Split Interface: Left = Document Viewer, Right = History Bot Chat
# -----------------------------------------------------------------------------
col_doc, col_chat = st.columns([1, 1], gap="medium")

# =============================================================================
# LEFT COLUMN: Academic Document Viewer (Matching Screenshot)
# =============================================================================
with col_doc:
    total_pages = int(meta.get("total_pages", 80))

    # Header bar matching Image 2
    st.markdown(
        f"""
        <div class="doc-header-bar">
            <div class="doc-header-title">
                📖 Orient BlackSwan History Catalogue
            </div>
            <div class="doc-header-meta">
                2024–25 (≈80 pp) &nbsp;|&nbsp; <strong>Page {st.session_state.current_page}</strong> of {total_pages}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Navigator controls
    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    with nav_c1:
        if st.button("◀ Prev", key="btn_prev", use_container_width=True):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
    with nav_c2:
        chosen_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page,
            label_visibility="collapsed",
            key="page_selector_input",
        )
        if chosen_page != st.session_state.current_page:
            st.session_state.current_page = chosen_page
            st.rerun()
    with nav_c3:
        if st.button("Next ▶", key="btn_next", use_container_width=True):
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()

    # Fetch page text
    page_text = loader.get_page_text(st.session_state.current_page)
    if not page_text:
        page_text = f"Orient BlackSwan History Catalogue 2024–25\n\nPage {st.session_state.current_page}\n\nScholarly monographs and university history textbooks."

    # Format text and highlight quotes if applicable
    highlight = st.session_state.get("highlight_text", "").strip()
    highlight_clean = re.sub(r'["\']', '', highlight)

    if highlight_clean and len(highlight_clean) > 8 and highlight_clean.lower() in page_text.lower():
        start_idx = page_text.lower().find(highlight_clean.lower())
        end_idx = start_idx + len(highlight_clean)
        before_txt = page_text[:start_idx].replace('\n', '<br>')
        matched_txt = page_text[start_idx:end_idx]
        after_txt = page_text[end_idx:].replace('\n', '<br>')
        rendered_body = (
            f"{before_txt}<div class='book-highlight-box'>"
            f"{matched_txt} <span class='badge-page-inline'>Page {st.session_state.current_page}</span>"
            f"</div>{after_txt}"
        )
    else:
        rendered_body = page_text.replace("\n", "<br>")

    # Render book page canvas
    st.markdown(
        f"""
        <div class="book-page-canvas">
            {rendered_body}
            <div class="book-page-number">— Page {st.session_state.current_page} —</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# RIGHT COLUMN: History Bot Chat (Matching Screenshot)
# =============================================================================
with col_chat:
    st.markdown(
        """
        <div class="chat-header-bar">
            <div class="chat-header-title">
                <span>🦢</span> History Bot
            </div>
            <div class="status-online">
                <div class="status-dot"></div> Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        vm, rag_chain = get_rag_pipeline()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    # Chat Messages Box
    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="bubble-bot">
                    <div class="bubble-bot-answer">
                        <strong>Welcome to the Orient BlackSwan History Bot!</strong><br>
                        I am your document-grounded research assistant for the <em>Orient BlackSwan History Catalogue 2024–25</em>.
                        Ask any factual question about our history publications, corporate background, or catalogue terms.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg_idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                res = msg.get("data", {})
                is_found = res.get("is_found", False)
                answer = res.get("answer", msg.get("content", ""))
                pages = res.get("pages", [])
                evidence = res.get("evidence", "")

                if not is_found or answer.strip().lower() in {"not found", "not found."}:
                    st.markdown(
                        """
                        <div class="not-found-card">
                            <div class="not-found-icon">📄</div>
                            <div>
                                <div class="not-found-title">Not found in the document</div>
                                <div class="not-found-desc">
                                    I couldn't find this information in the catalogue. The document does not provide sufficient evidence to answer this question.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    pages_str = ", ".join(str(p) for p in pages) if pages else "1"
                    citation_html = f'<div class="citation-pill-btn">📄 Page {pages_str} ➔</div>' if pages else ""

                    st.markdown(
                        f"""
                        <div class="bubble-bot">
                            <div class="bubble-bot-answer">{answer}</div>
                            {citation_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if evidence:
                        with st.expander(f"🔎 Supporting Quotation (Page {pages_str})", expanded=False):
                            st.write(f'"{evidence}"')

    # Suggested Questions (Compact Pill Buttons)
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#64748b; margin-bottom:6px;'>SUGGESTED QUESTIONS:</div>", unsafe_allow_html=True)
    
    suggested_q_list = [
        "In which year was the publisher (Orient BlackSwan) incorporated?",
        "On what date were the prices in this catalogue last updated?",
        "What is the price of a Physics textbook?",
        "Who is the current Minister of Education?",
        "Which is the \"best\" history book for a beginner?",
    ]

    c_s1, c_s2 = st.columns(2)
    selected_suggested = None
    with c_s1:
        if st.button("🏢 Incorporation Year", key="btn_sugg_incorp", use_container_width=True):
            selected_suggested = suggested_q_list[0]
        if st.button("🧪 Price of Physics Textbook", key="btn_sugg_phys", use_container_width=True):
            selected_suggested = suggested_q_list[2]
        if st.button("🌟 Best Beginner History Book", key="btn_sugg_best", use_container_width=True):
            selected_suggested = suggested_q_list[4]

    with c_s2:
        if st.button("📅 Price Update Date", key="btn_sugg_price", use_container_width=True):
            selected_suggested = suggested_q_list[1]
        if st.button("🏛️ Minister of Education", key="btn_sugg_min", use_container_width=True):
            selected_suggested = suggested_q_list[3]

    # Input Form
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            form_text = st.text_input(
                "Ask a question...",
                placeholder="Ask a question about the document...",
                label_visibility="collapsed",
                key="user_text_input",
            )
        with col_btn:
            submit_action = st.form_submit_button("Send ➔", use_container_width=True)

    # Trigger Question
    prompt_to_run = selected_suggested or (form_text if submit_action else None)

    if prompt_to_run:
        clean_prompt = prompt_to_run.strip()
        if clean_prompt:
            st.session_state.messages.append({"role": "user", "content": clean_prompt})

            with st.spinner("Retrieving document chunks..."):
                response = rag_chain.query(clean_prompt)

            # Auto-sync viewer to cited page and highlight quote
            if response["is_found"] and response.get("pages"):
                cited_page = response["pages"][0]
                st.session_state.current_page = cited_page
                st.session_state.highlight_text = response.get("evidence", "")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "data": response,
            })
            st.rerun()
