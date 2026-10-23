"""
Orient BlackSwan History Bot - Streamlit Application
A document-grounded RAG research assistant built with LangChain, ChromaDB, and Google Gemini.
"""

import sys
import os
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
# Streamlit Page Setup & Custom Editorial CSS
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
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@400;500;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap');

/* Academic Warm Off-White Theme */
:root {
    --bg-page: #fbf9f5;
    --bg-card: #ffffff;
    --text-navy: #141e30;
    --text-muted: #5a6578;
    --burgundy: #881337;
    --burgundy-light: #fbebf0;
    --burgundy-border: #e2a8b9;
    --border-card: #e5e0d4;
    --accent-gold: #c59b27;
}

body, .stApp {
    background-color: var(--bg-page);
    color: var(--text-navy);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Header Styles */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 1.8rem;
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(20, 30, 48, 0.03);
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
    font-size: 1.8rem;
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
    letter-spacing: 1.8px;
    color: var(--text-muted);
    margin-top: 4px;
}

.header-tagline {
    text-align: right;
    font-family: 'Merriweather', Georgia, serif;
    font-style: italic;
    font-size: 1.1rem;
    color: var(--text-navy);
}

.header-tagline-bar {
    width: 60px;
    height: 3px;
    background: var(--burgundy);
    margin-left: auto;
    margin-top: 4px;
    border-radius: 2px;
}

/* Diagram Bar */
.diagram-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    padding: 0.8rem 1rem;
    background: #ffffff;
    border: 1px dashed var(--border-card);
    border-radius: 10px;
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
    background: #f4efe6;
    color: var(--burgundy);
    border-radius: 8px;
    font-size: 16px;
}

/* Document Viewer (Left Pane) */
.doc-card {
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 1.2rem;
    height: 720px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.doc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 0.8rem;
}

.doc-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-navy);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.doc-badge {
    background: #f0ece3;
    padding: 0.25rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-navy);
}

.doc-content-box {
    flex-grow: 1;
    overflow-y: auto;
    background: #fdfcf9;
    border: 1px solid #ede8de;
    border-radius: 8px;
    padding: 1.4rem;
    font-family: 'Merriweather', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #2b3547;
    white-space: pre-wrap;
}

.doc-highlight {
    background-color: var(--burgundy-light);
    border-left: 4px solid var(--burgundy);
    padding: 0.6rem 0.8rem;
    margin: 0.8rem 0;
    border-radius: 0 6px 6px 0;
    color: var(--burgundy);
    font-weight: 500;
}

.page-indicator-badge {
    display: inline-block;
    background: var(--burgundy);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: 6px;
    vertical-align: middle;
}

/* Chat Pane (Right Pane) */
.chat-container-card {
    background: #ffffff;
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 1.2rem;
    height: 720px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 0.8rem;
}

.chat-title {
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

.chat-history-box {
    flex-grow: 1;
    overflow-y: auto;
    padding-right: 0.5rem;
    margin-bottom: 0.8rem;
}

/* Chat Bubbles */
.bubble-user {
    background: #f0f4f8;
    color: var(--text-navy);
    padding: 0.8rem 1.1rem;
    border-radius: 14px 14px 4px 14px;
    margin-left: 2rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    border: 1px solid #dce4ec;
}

.bubble-bot {
    background: #fbf9f4;
    color: var(--text-navy);
    padding: 1.1rem;
    border-radius: 14px 14px 14px 4px;
    margin-right: 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid #ebe4d8;
}

.bubble-answer {
    font-size: 0.95rem;
    line-height: 1.55;
    margin-bottom: 0.6rem;
    color: #141e30;
}

.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--burgundy-light);
    color: var(--burgundy);
    border: 1px solid var(--burgundy-border);
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    text-decoration: none;
    margin-top: 0.4rem;
}

/* Not Found Alert */
.not-found-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
}

.not-found-title {
    font-weight: 700;
    color: #991b1b;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.not-found-desc {
    color: #7f1d1d;
    font-size: 0.82rem;
    margin-top: 4px;
}

/* Suggested Question Chips */
.pill-button {
    background: #f4efe6;
    border: 1px solid #dfd8c8;
    color: var(--text-navy);
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.75rem;
    border-radius: 16px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    display: inline-block;
    cursor: pointer;
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

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"


# -----------------------------------------------------------------------------
# Cached Backend Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to persistent vector store...")
def get_rag_pipeline():
    """Initializes and caches vector store and RAG chain."""
    vm = VectorStoreManager()
    if not vm.is_indexed():
        # Check if PDF exists to auto-ingest
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
# App Header Component
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

# Architecture Workflow Diagram
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
# Sidebar: Document Management & Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Catalogue Knowledge Base")
    pdf_path = Config.DEFAULT_PDF_PATH
    loader = PDFLoader(pdf_path)
    file_exists, file_msg = loader.verify_file()

    if file_exists:
        st.success(f"**Loaded:** `{pdf_path.name}`")
    else:
        st.error(f"**Missing:** `{pdf_path.name}`")
        st.caption("Place `History_2024-25.pdf` into the `data/` directory.")

    # Ingestion Status Card
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
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Re-indexing PDF chunks into ChromaDB..."):
                docs, stats = loader.load()
                from src.text_splitter import DocumentChunker
                chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                chunks = chunker.split_documents(docs)
                vm_check.index_documents(chunks, stats, force_rebuild=True)
                st.cache_resource.clear()
                st.success("Re-indexed!")
                st.rerun()

    with col_sb2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Orient BlackSwan History Bot • Final Year RAG Project")


# -----------------------------------------------------------------------------
# Main Split Interface: Left = Document Viewer, Right = Chat Bot
# -----------------------------------------------------------------------------
col_doc, col_chat = st.columns([1, 1], gap="medium")

# =============================================================================
# LEFT COLUMN: Academic Document Viewer
# =============================================================================
with col_doc:
    st.markdown(
        f"""
        <div class="doc-header">
            <div class="doc-title">
                📖 Orient BlackSwan History Catalogue 2024–25
            </div>
            <div class="doc-badge">
                Page {st.session_state.current_page} of {meta.get('total_pages', 80)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Navigator
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("◀ Prev", key="btn_prev", use_container_width=True):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
    with nav_col2:
        selected_p = st.number_input(
            "Jump to Page",
            min_value=1,
            max_value=int(meta.get("total_pages", 80)),
            value=st.session_state.current_page,
            label_visibility="collapsed",
            key="page_input",
        )
        if selected_p != st.session_state.current_page:
            st.session_state.current_page = selected_p
            st.rerun()
    with nav_col3:
        if st.button("Next ▶", key="btn_next", use_container_width=True):
            if st.session_state.current_page < meta.get("total_pages", 80):
                st.session_state.current_page += 1
                st.rerun()

    # Fetch Page Text
    current_page_text = loader.get_page_text(st.session_state.current_page)
    if not current_page_text:
        current_page_text = (
            f"[Page {st.session_state.current_page}]\n\n"
            "This catalogue page contains listings of academic publications, monographs, and textbooks."
        )

    # Highlight evidence quote if applicable
    highlight = st.session_state.get("highlight_text", "")
    content_html = ""
    if highlight and highlight.lower() in current_page_text.lower():
        # Insert highlight block
        start_idx = current_page_text.lower().find(highlight.lower())
        end_idx = start_idx + len(highlight)
        before = current_page_text[:start_idx]
        matched = current_page_text[start_idx:end_idx]
        after = current_page_text[end_idx:]
        content_html = (
            f"{before}<div class='doc-highlight'>{matched} "
            f"<span class='page-indicator-badge'>Page {st.session_state.current_page}</span></div>{after}"
        )
    else:
        content_html = current_page_text.replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="doc-content-box">
            {content_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# RIGHT COLUMN: History Bot Chat
# =============================================================================
with col_chat:
    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-title">
                <span>🦢</span> History Bot
            </div>
            <div class="status-online">
                <div class="status-dot"></div> Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize RAG Pipeline
    try:
        vm, rag_chain = get_rag_pipeline()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

    # Container for Chat Messages
    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="bubble-bot">
                    <div class="bubble-answer">
                        <strong>Welcome to the Orient BlackSwan History Bot!</strong><br>
                        I am your document-grounded research assistant for the <em>Orient BlackSwan History Catalogue 2024–25</em>.
                        Ask any factual question about our history publications, corporate background, or catalogue terms.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                res = msg.get("data", {})
                is_found = res.get("is_found", False)
                answer = res.get("answer", msg.get("content", ""))
                pages = res.get("pages", [])
                evidence = res.get("evidence", "")

                if not is_found or answer.strip().lower() == "not found":
                    st.markdown(
                        """
                        <div class="not-found-card">
                            <div class="not-found-title">
                                <span>📄</span> Not found in the document
                            </div>
                            <div class="not-found-desc">
                                I couldn't find this information in the catalogue. The document does not provide sufficient evidence to answer this question.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    pages_str = ", ".join(str(p) for p in pages) if pages else "1"
                    citation_html = (
                        f'<div class="citation-pill">📄 Page {pages_str} ➔</div>'
                        if pages
                        else ""
                    )

                    st.markdown(
                        f"""
                        <div class="bubble-bot">
                            <div class="bubble-answer">{answer}</div>
                            {citation_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if evidence:
                        with st.expander(f"🔎 Supporting Quotation (Page {pages_str})", expanded=False):
                            st.write(f'"{evidence}"')

    # Suggested Questions Pills
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#5a6578; margin-bottom:6px;'>SUGGESTED QUESTIONS:</div>", unsafe_allow_html=True)
    suggested_questions = [
        "In which year was the publisher (Orient BlackSwan) incorporated?",
        "On what date were the prices in this catalogue last updated?",
        "What is the price of a Physics textbook?",
        "Who is the current Minister of Education?",
        "Which is the \"best\" history book for a beginner?",
    ]

    # Render quick pill buttons in a compact row
    c_p1, c_p2 = st.columns([1, 1])
    clicked_suggested = None
    with c_p1:
        if st.button("🏢 Incorporation Year", key="sugg_1", use_container_width=True):
            clicked_suggested = suggested_questions[0]
        if st.button("🧪 Price of Physics Textbook", key="sugg_3", use_container_width=True):
            clicked_suggested = suggested_questions[2]
        if st.button("🌟 Best Beginner History Book", key="sugg_5", use_container_width=True):
            clicked_suggested = suggested_questions[4]

    with c_p2:
        if st.button("📅 Price Update Date", key="sugg_2", use_container_width=True):
            clicked_suggested = suggested_questions[1]
        if st.button("🏛️ Minister of Education", key="sugg_4", use_container_width=True):
            clicked_suggested = suggested_questions[3]

    # Input Box Form
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Ask a question about the document...",
                placeholder="Ask a question about the catalogue...",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_btn:
            submit_btn = st.form_submit_button("Send ➔", use_container_width=True)

    # Process Query
    query_to_run = clicked_suggested or (user_input if submit_btn else None)

    if query_to_run:
        clean_q = query_to_run.strip()
        if clean_q:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": clean_q})

            # Run RAG Chain
            with st.spinner("Retrieving document chunks and grounding answer..."):
                response = rag_chain.query(clean_q)

            # Auto-sync document viewer to cited page if found
            if response["is_found"] and response.get("pages"):
                first_page = response["pages"][0]
                st.session_state.current_page = first_page
                st.session_state.highlight_text = response.get("evidence", "")

            # Append assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "data": response,
            })
            st.rerun()
