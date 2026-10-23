"""
Orient BlackSwan History Bot — Academic Research Assistant
RAG chatbot grounded on History_2024-25.pdf using LangChain + ChromaDB + Google Gemini.
Full light-ivory theme. Zero black anywhere. Premium academic editorial design.
"""

import sys
import re
from pathlib import Path

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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

:root {
  --bg: #F7F5F0;
  --surface: #FFFFFF;
  --surface-2: #F1EEE7;
  --navy: #192C46;
  --burgundy: #8D2947;
  --burg-h: #742039;
  --burg-a: #651A32;
  --burg-pale: #FDF2F4;
  --burg-border: #F0CCDA;
  --txt: #242B35;
  --txt2: #6B7280;
  --border: #E2DDD4;
  --green: #1E7A52;
  --green-bg: #E8F5EE;
  --focus: #C08EA0;
  --shadow-sm: 0 1px 3px rgba(25,44,70,.06);
  --shadow-md: 0 4px 12px rgba(25,44,70,.08);
  --r-sm: 8px;
  --r-md: 12px;
}

*, *::before, *::after {
  box-sizing: border-box;
  -webkit-tap-highlight-color: transparent !important;
}

html, body,
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main,
.main > div {
  background-color: var(--bg) !important;
  color: var(--txt) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── KILL THE BLACK HEADER BAR at the top ── */
[data-testid="stHeader"],
[data-testid="stHeader"] > div,
header[data-testid="stHeader"] {
  background: var(--surface) !important;
  background-color: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  box-shadow: none !important;
}

/* Deploy button area */
[data-testid="stToolbar"],
[data-testid="stToolbar"] > div,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"] {
  background: var(--surface) !important;
  background-color: var(--surface) !important;
  color: var(--txt2) !important;
}

/* Top-right action buttons (Deploy, overflow menu) */
[data-testid="stToolbarActions"],
[data-testid="stToolbarActions"] button,
[data-testid="stToolbarActions"] svg {
  color: var(--txt2) !important;
  fill: var(--txt2) !important;
}

.main .block-container {
  max-width: 1200px !important;
  padding: 1rem 2rem 4rem !important;
  background: var(--bg) !important;
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  color: var(--txt) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--txt) !important; }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--txt2) !important;
}

[data-testid="stTabs"] { background: transparent !important; }
[data-baseweb="tab-list"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 4px !important;
  gap: 4px !important;
}
button[data-baseweb="tab"] {
  background: transparent !important;
  color: #4A5568 !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  border-radius: var(--r-sm) !important;
  border: none !important;
  padding: 0.45rem 1.25rem !important;
  transition: all 150ms ease !important;
  outline: none !important;
}
button[data-baseweb="tab"] * { color: inherit !important; }
button[data-baseweb="tab"]:hover { background: var(--surface) !important; color: var(--burgundy) !important; }
button[data-baseweb="tab"]:hover * { color: var(--burgundy) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--burgundy) !important;
  color: #FFF !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 6px rgba(141,41,71,.28) !important;
}
button[data-baseweb="tab"][aria-selected="true"] * { color: #FFF !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { background: transparent !important; padding: 0 !important; }

.stButton > button,
button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important;
  color: var(--txt) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  padding: 0.5rem 1rem !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 150ms ease !important;
  outline: none !important;
  -webkit-tap-highlight-color: transparent !important;
}
.stButton > button:hover, button[kind="secondary"]:hover {
  background: var(--surface-2) !important;
  color: var(--burgundy) !important;
  border-color: var(--focus) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active, button[kind="secondary"]:active {
  background: #E6E0D6 !important;
  color: var(--burgundy) !important;
  border-color: var(--burgundy) !important;
  transform: none !important;
}
.stButton > button:focus, .stButton > button:focus-visible,
button[kind="secondary"]:focus, button[kind="secondary"]:focus-visible {
  background: var(--surface) !important;
  color: var(--burgundy) !important;
  outline: 3px solid var(--focus) !important;
  outline-offset: 2px !important;
  box-shadow: none !important;
}

[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: var(--burgundy) !important;
  color: #FFF !important;
  border: 1px solid var(--burgundy) !important;
  border-radius: var(--r-sm) !important;
  padding: 0.55rem 1.2rem !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  box-shadow: 0 2px 8px rgba(141,41,71,.3) !important;
  transition: all 150ms ease !important;
  outline: none !important;
  -webkit-tap-highlight-color: transparent !important;
}
[data-testid="stFormSubmitButton"] > button:hover, button[kind="primary"]:hover {
  background: var(--burg-h) !important;
  border-color: var(--burg-h) !important;
  color: #FFF !important;
  transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] > button:active, button[kind="primary"]:active {
  background: var(--burg-a) !important;
  border-color: var(--burg-a) !important;
  color: #FFF !important;
  transform: none !important;
}
[data-testid="stFormSubmitButton"] > button:focus,
[data-testid="stFormSubmitButton"] > button:focus-visible,
button[kind="primary"]:focus {
  background: var(--burgundy) !important;
  color: #FFF !important;
  outline: 3px solid var(--focus) !important;
  outline-offset: 2px !important;
}

[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-baseweb="input"]:focus-within {
  border-color: var(--burgundy) !important;
  box-shadow: 0 0 0 3px rgba(197,138,157,.3) !important;
}
[data-baseweb="input"] input, [data-testid="stNumberInput"] input {
  background: transparent !important;
  color: var(--txt) !important;
  font-size: 0.92rem !important;
  padding: 0.6rem 0.85rem !important;
}
[data-testid="stNumberInput"] button {
  background: var(--surface-2) !important;
  color: var(--txt) !important;
  border: none !important;
}
[data-testid="stNumberInput"] button:hover {
  background: var(--border) !important;
  color: var(--navy) !important;
}

[data-testid="stSpinner"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  color: var(--txt) !important;
}
[data-testid="stSpinner"] * { color: var(--txt) !important; }

[data-testid="stForm"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 0.75rem 1rem !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
}
[data-testid="stExpander"] summary {
  background: var(--surface-2) !important;
  color: var(--txt) !important;
  border-radius: var(--r-sm) !important;
}
[data-testid="stExpander"] summary * { color: var(--txt) !important; }

h1,h2,h3,h4,h5,h6 { color: var(--navy) !important; }
p, li, .stMarkdown p { color: var(--txt) !important; }
hr { border-color: var(--border) !important; }

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
  background: var(--surface-2) !important;
  color: var(--navy) !important;
}

[data-baseweb="select"] > div { background: var(--surface) !important; border-color: var(--border) !important; color: var(--txt) !important; }
[data-baseweb="menu"], [data-baseweb="popover"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt) !important;
  border-radius: var(--r-sm) !important;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"] { background: transparent !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #C8C2B7; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #A89F92; }

/* ── Component classes ── */
.obs-header {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1rem 1.5rem;
  margin-bottom: 1.1rem;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 0.75rem;
  box-shadow: var(--shadow-sm);
}
.obs-brand { display: flex; align-items: center; gap: 0.9rem; }
.obs-icon {
  width: 44px; height: 44px;
  background: var(--navy); color: #FFF;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; flex-shrink: 0;
}
.obs-title {
  font-family: 'Cinzel', Georgia, serif;
  font-size: 1.4rem; font-weight: 700;
  color: var(--navy) !important; margin: 0; line-height: 1.15;
}
.obs-title em { font-style: normal; color: var(--burgundy); }
.obs-subtitle { font-size: 0.73rem; font-weight: 600; color: var(--txt2); letter-spacing: 0.5px; margin-top: 2px; }
.obs-right { display: flex; align-items: center; gap: 1rem; }
.obs-tagline { font-family: 'Newsreader', Georgia, serif; font-style: italic; font-size: 1rem; color: var(--navy); }
.obs-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: var(--green-bg); color: var(--green);
  border: 1px solid #A8D9C2;
  padding: 3px 10px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700;
}
.obs-pill-dot { width: 7px; height: 7px; background: var(--green); border-radius: 50%; }

.sb-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 0.9rem 1rem; margin-bottom: 0.8rem;
  box-shadow: var(--shadow-sm);
}
.sb-card-title {
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 1.1px; text-transform: uppercase;
  color: var(--navy) !important; margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 0.4rem;
}
.sb-row {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.8rem; padding: 0.26rem 0;
  border-bottom: 1px dashed var(--border); color: var(--txt);
}
.sb-row:last-child { border-bottom: none; }
.sb-label { color: var(--txt2) !important; }
.sb-value { font-weight: 600; color: var(--navy) !important; }

.welcome-box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem; box-shadow: var(--shadow-sm);
}
.welcome-title {
  font-family: 'Cinzel', Georgia, serif;
  font-size: 1.3rem; font-weight: 700; color: var(--navy) !important; margin-bottom: 0.35rem;
}
.welcome-body { font-size: 0.9rem; color: var(--txt2); line-height: 1.6; }
.welcome-body strong { color: var(--txt); }
.sq-label {
  font-size: 0.71rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: var(--txt2); margin: 1rem 0 0.65rem;
}

.sq-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 1rem 1.1rem 0.8rem;
  height: 100%; display: flex; flex-direction: column;
  box-shadow: var(--shadow-sm);
  transition: border-color 150ms, box-shadow 150ms;
}
.sq-card:hover { border-color: var(--focus); box-shadow: 0 3px 10px rgba(141,41,71,.1); }
.sq-tag { font-size: 0.7rem; font-weight: 700; color: var(--burgundy); text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.35rem; }
.sq-query { font-size: 0.9rem; font-weight: 600; color: var(--navy); line-height: 1.45; margin-bottom: 0.9rem; flex: 1; }

.user-row { display: flex; justify-content: flex-end; margin-bottom: 1rem; }
.user-bubble {
  background: var(--burg-pale); color: var(--navy);
  border: 1px solid var(--burg-border);
  border-radius: 14px 14px 4px 14px;
  padding: 0.8rem 1.2rem; max-width: 80%;
  font-size: 0.93rem; line-height: 1.5; box-shadow: var(--shadow-sm);
}

.asst-row { display: flex; justify-content: flex-start; margin-bottom: 1.4rem; }
.asst-bubble {
  background: var(--surface); color: var(--txt);
  border: 1px solid var(--border);
  border-radius: 14px 14px 14px 4px;
  padding: 1.15rem 1.4rem; max-width: 90%; box-shadow: var(--shadow-sm);
}
.asst-text { font-size: 0.94rem; line-height: 1.65; color: var(--txt); margin-bottom: 0.8rem; }

.cite-panel {
  background: var(--surface-2);
  border-left: 4px solid var(--burgundy);
  border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 0.8rem 1rem; margin-top: 0.8rem;
}
.cite-header { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.3rem; }
.cite-src-badge { background: var(--navy); color: #FFF; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; padding: 2px 7px; border-radius: 4px; }
.cite-file { font-size: 0.8rem; font-weight: 600; color: var(--navy); }
.cite-page { background: #F8EBF0; color: var(--burgundy); border: 1px solid #E0BAC9; font-size: 0.76rem; font-weight: 700; padding: 2px 9px; border-radius: 12px; margin-left: auto; }
.cite-quote { font-family: 'Newsreader', Georgia, serif; font-style: italic; font-size: 0.9rem; color: #4A5568; line-height: 1.5; margin-top: 0.3rem; }

.nf-panel { background: var(--surface-2); border: 1px solid var(--border); border-left: 4px solid var(--txt2); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 0.9rem 1.1rem; }
.nf-head { font-weight: 700; color: var(--txt); font-size: 0.9rem; display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.2rem; }
.nf-body { font-size: 0.83rem; color: var(--txt2); line-height: 1.45; }

.reader-box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 2rem 2.5rem;
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1rem; line-height: 1.8; color: var(--txt);
  box-shadow: var(--shadow-md); min-height: 520px;
}
.reader-header { border-bottom: 1px solid var(--border); padding-bottom: 0.7rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 600; color: var(--txt2); }
.reader-highlight { background: #F8EBF0; border-left: 3px solid var(--burgundy); padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0; color: var(--burgundy); font-weight: 500; }
.reader-footer { border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 2rem; text-align: center; font-family: 'Cinzel', serif; font-size: 0.85rem; color: var(--txt2); }
</style>
"""

st.markdown(MASTER_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""
if "active_question" not in st.session_state:
    st.session_state.active_question = None


# ─────────────────────────────────────────────────────────────────────────────
# Backend: cached RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to vector store...")
def get_rag_pipeline():
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


# ─────────────────────────────────────────────────────────────────────────────
# Load meta
# ─────────────────────────────────────────────────────────────────────────────
_vm_meta = VectorStoreManager()
is_indexed = _vm_meta.is_indexed()
meta = _vm_meta.get_metadata()
total_pages = int(meta.get("total_pages", 80))


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
pill_label = "Index Ready" if is_indexed else "Indexing Required"
st.markdown(f"""
<div class="obs-header">
  <div class="obs-brand">
    <div class="obs-icon">📖</div>
    <div>
      <h1 class="obs-title">Orient BlackSwan <em>History Bot</em></h1>
      <div class="obs-subtitle">Grounded Academic AI Research Assistant</div>
    </div>
  </div>
  <div class="obs-right">
    <div class="obs-tagline">Ask. Explore. Learn.</div>
    <div class="obs-pill">
      <span class="obs-pill-dot"></span>
      <span>{pill_label}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    pdf_loader = PDFLoader(Config.DEFAULT_PDF_PATH)
    file_ok, _ = pdf_loader.verify_file()
    avail = "✅ Verified" if file_ok else "⚠️ Missing"

    st.markdown(f"""
    <div class="sb-card">
      <div class="sb-card-title">📚 Knowledge Source</div>
      <div class="sb-row"><span class="sb-label">Document</span><span class="sb-value">History_2024-25.pdf</span></div>
      <div class="sb-row"><span class="sb-label">File Status</span><span class="sb-value">{avail}</span></div>
      <div class="sb-row"><span class="sb-label">Pages Indexed</span><span class="sb-value">{total_pages} pages</span></div>
      <div class="sb-row"><span class="sb-label">Vector Chunks</span><span class="sb-value">{meta.get('chunk_count', 81)} chunks</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-card"><div class="sb-card-title">⚙️ Index Management</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Rebuild", use_container_width=True):
            with st.spinner("Re-indexing..."):
                docs, stats = pdf_loader.load()
                from src.text_splitter import DocumentChunker
                chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                chunks = chunker.split_documents(docs)
                _vm_meta.index_documents(chunks, stats, force_rebuild=True)
                st.cache_resource.clear()
                st.success("Index rebuilt!")
                st.rerun()
    with c2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(f"""
    <div class="sb-card">
      <div class="sb-card-title">🛡️ Model & Grounding</div>
      <div class="sb-row"><span class="sb-label">LLM</span><span class="sb-value">{Config.GOOGLE_MODEL}</span></div>
      <div class="sb-row"><span class="sb-label">Embeddings</span><span class="sb-value">{Config.GOOGLE_EMBEDDING_MODEL.split('/')[-1]}</span></div>
      <div class="sb-row"><span class="sb-label">Vector Store</span><span class="sb-value">ChromaDB</span></div>
      <div class="sb-row"><span class="sb-label">API Key</span><span class="sb-value">{Config.get_masked_api_key()}</span></div>
    </div>
    <div class="sb-card">
      <div class="sb-card-title">ℹ️ About</div>
      <div class="sb-row"><span class="sb-label">Framework</span><span class="sb-value">LangChain + Streamlit</span></div>
      <div class="sb-row"><span class="sb-label">Database</span><span class="sb-value">ChromaDB (persistent)</span></div>
      <div class="sb-row"><span class="sb-label">Assignment</span><span class="sb-value">GEN-AI MIDSEM</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Orient BlackSwan History Bot • GEN-AI Project")


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_reader = st.tabs(["💬 Research Assistant", "📖 Document Reader"])

try:
    vm, rag_chain = get_rag_pipeline()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESEARCH ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
          <div class="welcome-title">Your History Research Assistant</div>
          <div class="welcome-body">
            Ask questions about the <strong>Orient BlackSwan History Catalogue 2024–25</strong>
            and receive evidence-based answers with verified page references.
          </div>
        </div>
        <div class="sq-label">Suggested Research Questions</div>
        """, unsafe_allow_html=True)

        SUGGESTED = [
            ("Question 1", "When were the catalogue prices last updated?",
             "On what date were the prices in this catalogue last updated?"),
            ("Question 2", "In which year was Orient BlackSwan incorporated?",
             "In which year was the publisher (Orient BlackSwan) incorporated?"),
            ("Question 3", "List a book about Modern Indian History with its price.",
             "List a book about Modern Indian History with its author and price."),
        ]

        cols = st.columns(3)
        for col, (tag, display, query) in zip(cols, SUGGESTED):
            with col:
                st.markdown(f"""
                <div class="sq-card">
                  <div class="sq-tag">{tag}</div>
                  <div class="sq-query">{display}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Ask this →", key=f"sq_{tag}", use_container_width=True):
                    st.session_state.active_question = query
                    st.rerun()

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-row">
              <div class="user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            res = msg.get("data", {})
            is_found = res.get("is_found", False)
            answer = res.get("answer", msg.get("content", ""))
            pages = res.get("pages", [])
            evidence = res.get("evidence", "")
            pages_str = ", ".join(str(p) for p in pages) if pages else "—"

            if not is_found or answer.strip().lower() in {"not found", "not found."}:
                st.markdown("""
                <div class="asst-row">
                  <div class="asst-bubble">
                    <div class="nf-panel">
                      <div class="nf-head">📄 Not found in document</div>
                      <div class="nf-body">The catalogue does not contain sufficient evidence to answer this question.</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                safe_answer = answer.replace("<", "&lt;").replace(">", "&gt;")
                safe_evidence = evidence.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="asst-row">
                  <div class="asst-bubble">
                    <div class="asst-text">{safe_answer}</div>
                    <div class="cite-panel">
                      <div class="cite-header">
                        <span class="cite-src-badge">Source</span>
                        <span class="cite-file">History_2024-25.pdf</span>
                        <span class="cite-page">PDF Page {pages_str}</span>
                      </div>
                      <div class="cite-quote">"{safe_evidence}"</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if evidence:
                    with st.expander(f"🔎 Supporting extract (Page {pages_str})", expanded=False):
                        st.markdown(f"**Verbatim from document:**\n> {evidence}")
                        st.caption(f"Cited from History_2024-25.pdf — Page {pages_str}")

    with st.form("query_form", clear_on_submit=True):
        ic, sc = st.columns([5, 1])
        with ic:
            typed = st.text_input(
                "Question",
                placeholder="Ask anything about the History catalogue...",
                label_visibility="collapsed",
                key="query_input",
            )
        with sc:
            submitted = st.form_submit_button("Send ➔", use_container_width=True)

    query = st.session_state.active_question or (typed if submitted else None)
    if query:
        st.session_state.active_question = None
        q = query.strip()
        if q:
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Retrieving and grounding answer..."):
                response = rag_chain.query(q)
            if response["is_found"] and response.get("pages"):
                st.session_state.current_page = response["pages"][0]
                st.session_state.highlight_text = response.get("evidence", "")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "data": response,
            })
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT READER
# ═════════════════════════════════════════════════════════════════════════════
with tab_reader:
    nav_col, pg_col = st.columns([3, 1])

    with nav_col:
        lc, mc, rc = st.columns([1, 2, 1])
        with lc:
            if st.button("◀ Prev", key="prev_page", use_container_width=True):
                if st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
        with mc:
            sel = st.number_input(
                "Page", min_value=1, max_value=total_pages,
                value=st.session_state.current_page,
                label_visibility="collapsed", key="page_sel",
            )
            if sel != st.session_state.current_page:
                st.session_state.current_page = sel
                st.rerun()
        with rc:
            if st.button("Next ▶", key="next_page", use_container_width=True):
                if st.session_state.current_page < total_pages:
                    st.session_state.current_page += 1
                    st.rerun()

    with pg_col:
        st.markdown(
            f'<div style="text-align:right;font-size:.85rem;font-weight:600;color:var(--txt2);padding-top:.45rem">'
            f'Page {st.session_state.current_page} of {total_pages}</div>',
            unsafe_allow_html=True,
        )

    raw_text = pdf_loader.get_page_text(st.session_state.current_page)
    if not raw_text:
        raw_text = (f"Orient BlackSwan History Catalogue 2024-25\n\n"
                    f"Page {st.session_state.current_page}\n\n"
                    f"Scholarly monographs and university history textbooks.")

    hl = re.sub(r'["\']', '', st.session_state.get("highlight_text", "").strip())
    if hl and len(hl) > 8 and hl.lower() in raw_text.lower():
        s = raw_text.lower().find(hl.lower())
        e = s + len(hl)
        before = raw_text[:s].replace("\n", "<br>")
        match = raw_text[s:e]
        after = raw_text[e:].replace("\n", "<br>")
        body = (f"{before}"
                f"<div class='reader-highlight'>{match} "
                f"<span style='font-size:.72rem;font-weight:700;color:#8D2947;margin-left:6px'>[Cited Source]</span></div>"
                f"{after}")
    else:
        body = raw_text.replace("\n", "<br>")

    st.markdown(f"""
    <div class="reader-box">
      <div class="reader-header">
        <span>ORIENT BLACKSWAN HISTORY CATALOGUE 2024-25</span>
        <span>PAGE {st.session_state.current_page} / {total_pages}</span>
      </div>
      <div>{body}</div>
      <div class="reader-footer">— {st.session_state.current_page} —</div>
    </div>
    """, unsafe_allow_html=True)
