"""
2_How_It_Works.py — How It Works
==================================
ReadDoc AI | MSc Data Science and Analytics
"""

import streamlit as st

st.set_page_config(
    page_title="How It Works — ReadDoc AI",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{font-size:28px!important;font-weight:700!important;color:#111827!important}
h2{font-size:20px!important;font-weight:600!important;color:#111827!important;
    margin-top:2rem!important;padding-bottom:8px!important;border-bottom:2px solid #EEF2FF!important}
.step-card{border:1px solid #E5E7EB;border-radius:14px;margin-bottom:20px;overflow:hidden}
.step-header{display:flex;align-items:center;gap:14px;padding:16px 22px;
    background:#F9FAFB;border-bottom:1px solid #E5E7EB}
.step-num{background:#1a56db;color:#fff;font-size:13px;font-weight:700;
    width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;flex-shrink:0}
.step-title{font-size:16px;font-weight:700;color:#111827}
.step-sub{font-size:12px;color:#6B7280;margin-top:2px}
.step-body{padding:18px 22px}
.step-desc{font-size:13.5px;color:#374151;line-height:1.75;margin-bottom:14px}
.why-box{background:#EEF2FF;border-radius:8px;padding:12px 16px;
    font-size:13px;color:#1e40af;line-height:1.6}
.why-label{font-size:10.5px;font-weight:700;text-transform:uppercase;
    letter-spacing:.06em;color:#1a56db;margin-bottom:4px}
.code-box{background:#0a0a0a;border-radius:8px;padding:14px 18px;
    font-family:'DM Mono',monospace;font-size:12px;color:#4ade80;
    margin:12px 0;line-height:1.8;overflow-x:auto}
.scoring-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:12px 0}
.score-card{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:10px;
    padding:14px 16px}
.score-name{font-size:13px;font-weight:700;color:#1a56db;margin-bottom:6px}
.score-desc{font-size:12.5px;color:#374151;line-height:1.6}
.score-formula{font-family:'DM Mono',monospace;font-size:11.5px;color:#059669;
    background:#ECFDF5;padding:4px 8px;border-radius:4px;margin-top:6px;
    display:inline-block}
.page-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.page-card{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:10px;
    padding:14px 16px}
.page-name{font-size:13px;font-weight:700;color:#111827;margin-bottom:4px}
.page-desc{font-size:12.5px;color:#6B7280;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.title("How It Works")
st.markdown("A step-by-step walkthrough of the ReadDoc AI pipeline — from document upload to scored evaluation results.")
st.markdown("---")

# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">1</div>
    <div>
      <div class="step-title">Document Upload & Text Extraction</div>
      <div class="step-sub">PyPDF · BeautifulSoup4 · File I/O</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        The user uploads a PDF or HTML file. The system extracts raw text
        page by page using PyPDF for PDFs, or BeautifulSoup4 for HTML files.
        Each extracted page is tagged with its source filename and page number,
        which later appears in answer citations.
    </div>
    <div class="code-box">
Input:  DSML.pdf (500 pages)<br>
Output: [ {text: "...", file: "DSML.pdf", page: 1},<br>
          {text: "...", file: "DSML.pdf", page: 2}, ... ]
    </div>
    <div class="why-box">
        <div class="why-label">Why this matters for the research</div>
        Page-level metadata is preserved throughout the pipeline so that when
        the system retrieves chunks for a question, it can cite exactly which
        pages the answer came from — a prerequisite for faithfulness evaluation.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 2 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">2</div>
    <div>
      <div class="step-title">Text Chunking — The First Experimental Variable</div>
      <div class="step-sub">Character-based sliding window with overlap · Three sizes: 300, 600, 1000 chars</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        Extracted text is split into overlapping chunks using a sliding window.
        Each chunk inherits the page number from the source page it was split from.
        An overlap of 100 characters between adjacent chunks prevents relevant
        content from being cut off at chunk boundaries.
    </div>
    <div class="code-box">
Chunk size 300:  "Supervised learning involves training a model on..."  (300 chars)<br>
Chunk size 600:  "Supervised learning involves training a model on labeled<br>
                  data, where the goal is to learn a mapping between..."  (600 chars)<br>
Chunk size 1000: Larger segment containing more contextual information...
    </div>
    <div class="why-box">
        <div class="why-label">Why chunk size is the key variable</div>
        Smaller chunks produce higher retrieval precision — the retrieved text is
        tightly focused on the query. Larger chunks provide more surrounding context
        per retrieved segment. This trade-off is the primary parameter under
        investigation in this study.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 3 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">3</div>
    <div>
      <div class="step-title">Embedding & FAISS Index Construction</div>
      <div class="step-sub">all-MiniLM-L6-v2 · FAISS IndexFlatL2 · 384-dimensional vectors</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        Every chunk is converted into a 384-dimensional dense vector (embedding) using the
        all-MiniLM-L6-v2 sentence transformer model. These embeddings capture
        semantic meaning rather than exact keywords — enabling retrieval by conceptual
        similarity. All vectors are stored in a FAISS flat L2 index for exact nearest-neighbour
        search. Three separate indexes are built — one per chunk size — and cached to disk
        so subsequent queries load instantly.
    </div>
    <div class="code-box">
"What is supervised learning?" → [0.021, -0.184, 0.093, ...] (384 dims)<br>
FAISS IndexFlatL2: stores all chunk embeddings<br>
Cache: faiss_600_a839ad0ba8.index  (instant reload after first build)
    </div>
    <div class="why-box">
        <div class="why-label">Why FAISS and not a cloud vector database</div>
        FAISS runs locally with no external dependency or cost. For this experimental
        study, exact L2 search guarantees deterministic retrieval — the same query
        always returns the same chunks, making results reproducible across experimental runs.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 4 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">4</div>
    <div>
      <div class="step-title">Semantic Retrieval — The Second Experimental Variable</div>
      <div class="step-sub">FAISS nearest-neighbour search · k = 3, 5, or 10 chunks</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        When a question is submitted, it is embedded using the same model as the chunks.
        FAISS computes L2 distance between the query vector and all chunk vectors,
        returning the top-k most similar chunks. Duplicate chunks from the same page are
        de-duplicated before being passed to the LLM.
    </div>
    <div class="code-box">
Query: "What is PCA?" → embedding → FAISS search<br>
k=3:  returns 3 chunks  (high precision, may miss some relevant content)<br>
k=5:  returns 5 chunks  (balanced)<br>
k=10: returns 10 chunks (broad coverage, potential noise)
    </div>
    <div class="why-box">
        <div class="why-label">Why k matters</div>
        Higher k gives the LLM more context to draw from but also introduces
        less-relevant chunks that can dilute the answer quality and increase
        hallucination risk. Finding the optimal k is the second research sub-question.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 5 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">5</div>
    <div>
      <div class="step-title">Answer Generation — Llama 3.3 via Groq</div>
      <div class="step-sub">llama-3.3-70b-versatile · Groq API · Constrained to document context</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        The retrieved chunks are formatted as context and passed to Llama 3.3 70B via
        the Groq API. A strict system prompt instructs the model to answer using only
        the provided context, cite the source page, and respond "I could not find this
        in your uploaded documents" if the answer is not present in the retrieved chunks.
        This out-of-scope refusal behaviour is tested explicitly in questions Q9–Q10.
    </div>
    <div class="code-box">
System: "Answer using ONLY the provided context. Cite source page."<br>
Context: [Chunk 1: DSML.pdf p.40] ... [Chunk 2: DSML.pdf p.139] ...<br>
Question: "What is the difference between supervised and unsupervised learning?"<br>
Response: **Answer:** In supervised learning... **Source:** DSML.pdf p.40, 139
    </div>
    <div class="why-box">
        <div class="why-label">Why Llama 3.3 70B is held constant</div>
        The LLM is the dependent variable's renderer, not the independent variable.
        Keeping the same model across all configurations ensures that any differences
        in answer quality are attributable to chunk size and k, not to model capability.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 6 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-num">6</div>
    <div>
      <div class="step-title">Human Evaluation & Scoring</div>
      <div class="step-sub">0.0–1.0 scale · Three metrics · Saved to SQLite</div>
    </div>
  </div>
  <div class="step-body">
    <div class="step-desc">
        Each answer is scored by a human evaluator on three dimensions using
        a structured 0.0–1.0 rubric, following the evaluation framework
        established by Es et al. (2023). Scores are saved per question per
        configuration in an SQLite database for subsequent analysis.
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="scoring-row">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="score-card">
        <div class="score-name">Accuracy</div>
        <div class="score-desc">
            Is the answer factually correct based on the document?
            1.00 = Perfect · 0.75 = Good · 0.50 = Moderate
            0.25 = Poor · 0.00 = Wrong
        </div>
        <div class="score-formula">Baseline: vs general knowledge</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="score-card">
        <div class="score-name">Relevance</div>
        <div class="score-desc">
            Does the answer directly address the question asked?
            Penalises off-topic responses or excessive padding.
            1.00 = Directly on-point · 0.00 = Irrelevant
        </div>
        <div class="score-formula">Same scale for baseline + RAG</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="score-card">
        <div class="score-name">Faithfulness</div>
        <div class="score-desc">
            Is every claim grounded in the retrieved chunks?
            Detects hallucination — facts not present in the source.
            1.00 = Fully grounded · 0.00 = Hallucinated
        </div>
        <div class="score-formula">Baseline: always 0.00 (no document)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="why-box" style="margin-top:12px">
        <div class="why-label">Why human evaluation</div>
        Human evaluation with a structured rubric is considered the gold standard
        for NLP answer quality assessment. Automated metrics such as BLEU and ROUGE
        measure lexical overlap rather than semantic correctness or faithfulness,
        making them unsuitable for open-ended Q&A evaluation. This approach follows
        the methodology used in comparable RAG evaluation studies (Es et al., 2023;
        CLAPNQ, 2024; Disco-RAG, 2024).
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Application pages ──────────────────────────────────────────────────────────
st.markdown("## Application Pages")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="page-card">
        <div class="page-name">📋 Project Overview</div>
        <div class="page-desc">Research question, objectives, experimental design, metrics, tech stack, and references.</div>
    </div>
    <br>
    <div class="page-card">
        <div class="page-name">🔍 EDA</div>
        <div class="page-desc">Exploratory analysis of the source document — chunk statistics, length distributions, word frequency, and chunk-size comparisons.</div>
    </div>
    <br>
    <div class="page-card">
        <div class="page-name">🧪 Baseline Test</div>
        <div class="page-desc">Run all 10 questions with zero document context (control condition). Score accuracy and relevance manually. Faithfulness fixed at 0.00.</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="page-card">
        <div class="page-name">⚡ Experiment Runner</div>
        <div class="page-desc">Select any of E1–E9, generate all 10 answers with document context, score accuracy, relevance, and faithfulness per answer.</div>
    </div>
    <br>
    <div class="page-card">
        <div class="page-name">📊 Results Analysis</div>
        <div class="page-desc">Comparison table, four charts (overall, three metrics, chunk size effect, k effect), auto-written findings, and research question answer paragraph. CSV export.</div>
    </div>
    <br>
    <div class="page-card">
        <div class="page-name">💬 ReadDoc AI (app)</div>
        <div class="page-desc">The live RAG chat interface — upload documents, build index for all chunk sizes, chat with instant source citations.</div>
    </div>
    """, unsafe_allow_html=True)