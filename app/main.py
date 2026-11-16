import hashlib
import re
import tempfile
from pathlib import Path

import streamlit as st
from transformers import AutoTokenizer

from src.ingestion.document_processor import process_pdf
from src.generation.financial_qa import FinancialQuestionAnsweringPipeline


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinDoc AI · Financial Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "active_document": None,
    "qa_pipeline": None,
    "active_document_hash": None,
    "pending_upload_hash": None,
    "chat_history": [],
    "queued_question": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_uploaded_document(uploaded_file, file_hash):
    file_name = Path(uploaded_file.name).name

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / file_name
        pdf_path.write_bytes(uploaded_file.getvalue())

        package = process_pdf(
            pdf_path=pdf_path,
            tokenizer=load_tokenizer(),
        )

    qa_pipeline = FinancialQuestionAnsweringPipeline(
        metadata=package["chunks"],
        top_k=5,
    )

    st.session_state.active_document = package
    st.session_state.qa_pipeline = qa_pipeline
    st.session_state.active_document_hash = file_hash
    st.session_state.chat_history = []


# ============================================================
# DISPLAY HELPERS
# ============================================================

def clean_answer_citations(answer: str) -> str:
    """Convert raw chunk citations into clean, scannable page pills."""
    cleaned = re.sub(
        r"\[Page:\s*(\d+)\s*\|\s*Chunk:\s*[A-Za-z0-9_.-]+\]",
        r'<span class="citation-pill">Page \1</span>',
        answer,
    )
    cleaned = re.sub(
        r"\[Page:\s*(\d+)\]",
        r'<span class="citation-pill">Page \1</span>',
        cleaned,
    )
    return cleaned


def render_loader(animation_type="processing"):
    if animation_type == "processing":
        animation = """
        <div class="loader-wrapper">
            <div class="loader-orbit">
                <div class="orbit orbit-one"></div>
                <div class="orbit orbit-two"></div>
                <div class="loader-core">✦</div>
            </div>
            <div class="loader-label">Indexing financial tables &amp; passages...</div>
        </div>
        """
    else:
        animation = """
        <div class="thinking-bar">
            <div class="answer-animation">
                <span></span><span></span><span></span><span></span>
            </div>
            <span class="thinking-text">Analyzing document context...</span>
        </div>
        """

    st.markdown(animation, unsafe_allow_html=True)


# ============================================================
# EXECUTIVE DESIGN SYSTEM (CSS)
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean SaaS Background */
    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(99, 102, 241, 0.06), transparent 32%),
            radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.05), transparent 28%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Main Container Width & Padding */
    .block-container {
        max-width: 920px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }

    /* Top Brand Bar */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }

    .brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0;
    }

    .brand-gradient {
        background: linear-gradient(115deg, #4f46e5 0%, #0ea5e9 55%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 4px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .sidebar-brand {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Compact Active Document Card in Sidebar */
    .doc-card {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.22);
        background: rgba(99, 102, 241, 0.04);
        margin-top: 8px;
    }

    .doc-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        word-break: break-word;
        margin-bottom: 8px;
        display: flex;
        align-items: flex-start;
        gap: 6px;
    }

    .doc-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        padding-top: 8px;
        border-top: 1px solid rgba(148, 163, 184, 0.18);
    }

    .doc-meta-item {
        display: flex;
        flex-direction: column;
    }

    .doc-meta-val {
        font-size: 0.88rem;
        font-weight: 700;
    }

    .doc-meta-lbl {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Hero Empty State */
    .hero-box {
        padding: 2rem 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(255, 255, 255, 0.02);
        margin-bottom: 1.25rem;
    }

    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }

    .hero-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 620px;
    }

    .feature-card {
        padding: 1.15rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(148, 163, 184, 0.03);
        height: 100%;
    }

    .feature-icon {
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }

    .feature-text {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.86rem;
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #4f46e5, #0ea5e9);
        border: none;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.22);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.3);
    }

    .stButton > button[kind="secondary"] {
        border: 1px solid rgba(148, 163, 184, 0.28);
        background: rgba(148, 163, 184, 0.04);
        text-align: left;
        justify-content: flex-start;
        padding: 0.65rem 0.9rem;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #6366f1;
        background: rgba(99, 102, 241, 0.06);
    }

    /* Chat Messages & Citation Pills */
    [data-testid="stChatMessage"] {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 0.75rem;
    }

    .citation-pill {
        display: inline-block;
        padding: 1px 8px;
        margin: 0 2px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #4f46e5;
        background: rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.22);
        border-radius: 6px;
        vertical-align: baseline;
    }

    .verified-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #059669;
        margin-top: 6px;
    }

    .unverified-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #d97706;
        margin-top: 6px;
    }

    /* Evidence Expander */
    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        margin-top: 8px;
    }

    .evidence-header {
        font-size: 0.8rem;
        font-weight: 700;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
    }

    /* Loaders */
    .loader-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 28px 12px;
        gap: 14px;
    }

    .loader-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b;
    }

    .loader-orbit {
        position: relative;
        width: 72px;
        height: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .orbit {
        position: absolute;
        border-radius: 50%;
        border: 2px solid transparent;
    }

    .orbit-one {
        inset: 0;
        border-top-color: #4f46e5;
        border-right-color: #0ea5e9;
        animation: rotate 1.5s linear infinite;
    }

    .orbit-two {
        inset: 10px;
        border-bottom-color: #10b981;
        border-left-color: #4f46e5;
        animation: rotate-reverse 1.1s linear infinite;
    }

    .loader-core {
        width: 30px;
        height: 30px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
        background: linear-gradient(135deg, #4f46e5, #0ea5e9);
    }

    .thinking-bar {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(148, 163, 184, 0.04);
        margin: 8px 0;
    }

    .thinking-text {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }

    .answer-animation {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .answer-animation span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4f46e5, #0ea5e9);
        animation: wave 0.9s ease-in-out infinite;
    }

    .answer-animation span:nth-child(2) { animation-delay: 0.12s; }
    .answer-animation span:nth-child(3) { animation-delay: 0.24s; }
    .answer-animation span:nth-child(4) { animation-delay: 0.36s; }

    @keyframes rotate { to { transform: rotate(360deg); } }
    @keyframes rotate-reverse { to { transform: rotate(-360deg); } }
    @keyframes wave {
        0%, 100% { transform: translateY(0); opacity: 0.45; }
        50% { transform: translateY(-6px); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">✦ FinDoc Workspace</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload financial PDF",
        type=["pdf"],
        help=f"Supports annual reports, 10-K filings, and statements up to {MAX_UPLOAD_MB} MB.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            st.error(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
        else:
            if file_hash != st.session_state.pending_upload_hash:
                st.session_state.pending_upload_hash = file_hash

                if file_hash != st.session_state.active_document_hash:
                    st.session_state.active_document = None
                    st.session_state.qa_pipeline = None
                    st.session_state.active_document_hash = None
                    st.session_state.chat_history = []

            is_already_indexed = file_hash == st.session_state.active_document_hash
            button_label = "↻  Re-index Document" if is_already_indexed else "✦  Process Document"

            if st.button(
                button_label,
                type="primary" if not is_already_indexed else "secondary",
                use_container_width=True,
            ):
                loader = st.empty()
                try:
                    with loader.container():
                        render_loader("processing")

                    process_uploaded_document(uploaded_file, file_hash)
                    loader.empty()
                    st.rerun()

                except Exception as error:
                    loader.empty()
                    st.error(f"Processing failed: {error}")

    else:
        if st.session_state.pending_upload_hash is not None:
            st.session_state.active_document = None
            st.session_state.qa_pipeline = None
            st.session_state.active_document_hash = None
            st.session_state.pending_upload_hash = None
            st.session_state.chat_history = []

    # Compact Active Document Metadata Card (Replaces cluttered main-screen overview)
    if st.session_state.active_document is not None:
        doc_info = st.session_state.active_document["document"]
        doc_stats = st.session_state.active_document["statistics"]

        st.markdown("---")
        st.caption("ACTIVE INDEX")
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-card-title">📄 {doc_info['source_file']}</div>
                <div class="doc-meta-grid">
                    <div class="doc-meta-item">
                        <span class="doc-meta-val">{doc_info['total_pages']:,}</span>
                        <span class="doc-meta-lbl">Pages</span>
                    </div>
                    <div class="doc-meta-item">
                        <span class="doc-meta-val">{doc_info['total_words']:,}</span>
                        <span class="doc-meta-lbl">Words</span>
                    </div>
                    <div class="doc-meta-item">
                        <span class="doc-meta-val">{doc_stats['total_chunks']:,}</span>
                        <span class="doc-meta-lbl">Chunks</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.chat_history:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    st.markdown("---")
    st.caption("ENGINE")
    st.markdown("**BM25 Hybrid Retrieval · Gemini AI**")


# ============================================================
# TOP HEADER
# ============================================================

active_doc = st.session_state.active_document
status_html = (
    f'<div class="status-badge"><span class="status-dot"></span>{active_doc["document"]["source_file"]}</div>'
    if active_doc
    else ""
)

st.markdown(
    f"""
    <div class="brand-header">
        <div>
            <div class="brand-title"><span class="brand-gradient">FinDoc AI</span></div>
            <div class="brand-subtitle">Institutional-grade Q&amp;A and citation verification for financial filings.</div>
        </div>
        {status_html}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN WORKSPACE
# ============================================================

document = st.session_state.active_document
qa_pipeline = st.session_state.qa_pipeline

if document is None or qa_pipeline is None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Interrogate complex financial filings in seconds.</div>
            <div class="hero-desc">
                Upload an annual report, earnings release, or balance sheet from the left sidebar.
                FinDoc AI indexes every section and provides verifiable answers anchored to exact page numbers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📑</div>
                <div class="feature-title">Structured Indexing</div>
                <div class="feature-text">Parses multi-page financial PDFs into searchable semantic sections.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Precision Retrieval</div>
                <div class="feature-text">Pinpoints exact figures, footnotes, and ratios across dense reports.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">✓</div>
                <div class="feature-title">Auditable Citations</div>
                <div class="feature-text">Every answer links directly to underlying source passages and pages.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # Suggested Starter Prompts when chat is empty
    if not st.session_state.chat_history:
        st.caption("SUGGESTED QUERIES")
        q_col1, q_col2 = st.columns(2)
        starter_prompts = [
            "What were total revenues and net income for the fiscal year?",
            "Summarize key risk factors and operational headwinds.",
            "How much was returned to shareholders via dividends and buybacks?",
            "What is the company's current cash, debt, and liquidity position?",
        ]
        for idx, prompt_text in enumerate(starter_prompts):
            target_col = q_col1 if idx % 2 == 0 else q_col2
            if target_col.button(f"↗  {prompt_text}", key=f"starter_{idx}", use_container_width=True):
                st.session_state.queued_question = prompt_text
                st.rerun()

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["question"])

        with st.chat_message("assistant"):
            st.markdown(message["answer"], unsafe_allow_html=True)

            validation = message["citation_validation"]
            if validation["valid"]:
                st.markdown('<div class="verified-tag">✓ Sources verified against document</div>', unsafe_allow_html=True)
            elif message["evidence_found"]:
                st.markdown('<div class="unverified-tag">⚠ Partial citation match — verify source text below</div>', unsafe_allow_html=True)

            if message["evidence"]:
                with st.expander(f"View supporting evidence ({len(message['evidence'])} passages)"):
                    for index, chunk in enumerate(message["evidence"], start=1):
                        st.markdown(
                            f'<div class="evidence-header">Source {index} · Page {chunk["page_number"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.write(chunk["text"])
                        if index < len(message["evidence"]):
                            st.divider()

    # Handle Input (either from chat box or starter prompt click)
    user_input = st.chat_input("Ask a financial question about this document...")
    question = st.session_state.queued_question or user_input
    st.session_state.queued_question = None

    if question:
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": "",
                "sources": [],
                "evidence": [],
                "evidence_found": False,
                "citation_validation": {"valid": False},
            }
        )

        current_message = st.session_state.chat_history[-1]
        animation_placeholder = st.empty()

        try:
            with animation_placeholder.container():
                render_loader("answer")

            result = qa_pipeline.answer(question)
            evidence = []

            if result["evidence_found"]:
                retrieval_result = qa_pipeline.rag.run(query=question)
                evidence = retrieval_result["retrieved_chunks"]

            current_message.update(
                {
                    "answer": clean_answer_citations(result["answer"]),
                    "sources": result["sources"],
                    "evidence": evidence,
                    "evidence_found": result["evidence_found"],
                    "citation_validation": result["citation_validation"],
                }
            )

        except Exception as error:
            current_message["answer"] = (
                f"Unable to generate an answer right now. Please try again. (`{error}`)"
            )

        finally:
            animation_placeholder.empty()

        st.rerun()