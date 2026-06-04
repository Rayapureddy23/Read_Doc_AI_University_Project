"""
1_Project_Overview.py — Research Project Overview
===================================================
ReadDoc AI | MSc Data Science and Analytics

Shows the supervisor:
  - Research question, aim, objectives
  - Technical approach
  - Project timeline and stages
"""

import streamlit as st

st.set_page_config(page_title="Project Overview — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.hero{background:#1a56db;border-radius:14px;padding:36px 40px;color:#fff;margin-bottom:2rem}
.hero-badge{display:inline-block;background:rgba(255,255,255,.2);font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:16px}
.hero-title{font-size:26px;font-weight:700;line-height:1.35;margin-bottom:10px}
.hero-rq{font-size:15px;opacity:.9;line-height:1.7;font-style:italic;
    border-left:3px solid rgba(255,255,255,.5);padding-left:16px;margin-top:12px}
.sec{font-size:18px;font-weight:600;color:#111827;margin:2rem 0 1rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.card{background:#F8FAFF;border:0.5px solid #E5E9F5;border-radius:12px;
    padding:18px 22px;margin-bottom:10px}
.card-title{font-size:14px;font-weight:600;color:#111827;margin-bottom:6px}
.card-text{font-size:13px;color:#374151;line-height:1.7}
.obj-row{display:flex;gap:12px;align-items:flex-start;padding:10px 0;
    border-bottom:.5px solid #F3F4F6}
.obj-row:last-child{border-bottom:none}
.obj-num{width:28px;height:28px;background:#EEF2FF;color:#1a56db;border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:12px;
    font-weight:600;flex-shrink:0;margin-top:1px}
.obj-text{font-size:13px;color:#374151;line-height:1.6;flex:1}
.stage-row{display:flex;gap:16px;flex-wrap:wrap;margin-top:.5rem}
.stage-card{flex:1;min-width:180px;background:#F8FAFF;border:.5px solid #E5E9F5;
    border-radius:10px;padding:14px 16px}
.stage-num{font-size:11px;font-weight:600;color:#1a56db;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:6px}
.stage-title{font-size:13px;font-weight:600;color:#111827;margin-bottom:6px}
.stage-items{font-size:12px;color:#6B7280;line-height:1.8}
.tech-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tech-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:8px;
    padding:12px 14px;display:flex;align-items:center;gap:10px}
.tech-name{font-size:13px;font-weight:500;color:#111827}
.tech-role{font-size:11.5px;color:#6B7280}
.var-row{display:flex;gap:12px;flex-wrap:wrap;margin:.5rem 0}
.var-card{flex:1;min-width:200px;border-radius:10px;padding:16px 18px}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">MSc Data Science and Analytics</div>
  <div class="hero-title">
      Optimising Retrieval-Augmented Generation for Document<br>
      Question Answering: An Empirical Study of Chunking Strategies,<br>
      Retrieval Depth, and Answer Relevance Using ReadDoc AI
  </div>
  <div class="hero-rq">
      Research Question: How does varying chunk size and retrieval depth in a
      Retrieval-Augmented Generation pipeline affect the accuracy, contextual relevance,
      and faithfulness of answers generated from unstructured documents?
  </div>
</div>
""", unsafe_allow_html=True)

# ── Aim ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Research aim</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
  <div class="card-text">
      The aim of this project is to design, implement, and empirically evaluate
      <b>ReadDoc AI</b> — a Retrieval-Augmented Generation system that enables users
      to query unstructured documents using natural language. The project investigates
      how two key configuration parameters — <b>chunk size</b> and <b>retrieval depth</b>
      — affect the accuracy, contextual relevance, and faithfulness of AI-generated answers,
      and identifies the optimal configuration for real-world document question answering.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Objectives ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Research objectives</div>', unsafe_allow_html=True)

objectives = [
    ("O1", "Review existing literature on RAG, semantic search, dense retrieval, and document QA systems."),
    ("O2", "Design and implement a full RAG pipeline: PDF/HTML ingestion → chunking → embedding → FAISS → LLM generation."),
    ("O3", "Build ReadDoc AI as the experimental platform — a working document chatbot with real-time streaming answers."),
    ("O4", "Design a controlled 3×3 factorial experiment varying chunk size (300/600/1000) and retrieval depth (k=3/5/10)."),
    ("O5", "Conduct exploratory data analysis on the document corpus to understand its structure before experimentation."),
    ("O6", "Run a baseline experiment (LLM with no documents) to establish the control condition."),
    ("O7", "Run all 9 RAG configurations across 20 standardised test questions, scoring each on accuracy, relevance, and faithfulness."),
    ("O8", "Identify the optimal configuration and provide evidence-based recommendations for RAG system design."),
]

for num, text in objectives:
    st.markdown(
        f'<div class="obj-row"><div class="obj-num">{num}</div>'
        f'<div class="obj-text">{text}</div></div>',
        unsafe_allow_html=True,
    )

# ── Experiment variables ────────────────────────────────────────────────────────
st.markdown('<div class="sec">Experiment variables</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="var-card" style="background:#EEF2FF;border:.5px solid #C7D3F5">
      <div style="font-size:12px;font-weight:600;color:#1a56db;margin-bottom:8px">
          INDEPENDENT VARIABLE 1</div>
      <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px">Chunk size</div>
      <div style="font-size:13px;color:#374151;line-height:1.7">
          300 chars — small, precise<br>
          600 chars — balanced<br>
          1000 chars — large, contextual
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="var-card" style="background:#EEF2FF;border:.5px solid #C7D3F5">
      <div style="font-size:12px;font-weight:600;color:#1a56db;margin-bottom:8px">
          INDEPENDENT VARIABLE 2</div>
      <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px">Retrieval depth (k)</div>
      <div style="font-size:13px;color:#374151;line-height:1.7">
          k = 3 — fewer chunks<br>
          k = 5 — balanced<br>
          k = 10 — broader context
      </div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="var-card" style="background:#ECFDF5;border:.5px solid #A7F3D0">
      <div style="font-size:12px;font-weight:600;color:#059669;margin-bottom:8px">
          DEPENDENT VARIABLES</div>
      <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px">Answer quality</div>
      <div style="font-size:13px;color:#374151;line-height:1.7">
          Accuracy (1–5)<br>
          Contextual relevance (1–5)<br>
          Faithfulness (1–5)
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="card" style="margin-top:12px">
  <div class="card-title">Controlled variables — kept the same across all 9 experiments</div>
  <div class="card-text">
      Same LLM model (Llama 3.3 via Groq) &nbsp;·&nbsp;
      Same embedding model (all-MiniLM-L6-v2) &nbsp;·&nbsp;
      Same document (DSML.pdf — Kroese et al., 2020) &nbsp;·&nbsp;
      Same 20 test questions &nbsp;·&nbsp;
      Same scoring rubric
  </div>
</div>
""", unsafe_allow_html=True)

# ── Technology stack ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Technology stack</div>', unsafe_allow_html=True)

tech = [
    ("Streamlit", "Web framework — frontend and UI"),
    ("Llama 3.3", "LLM via Groq API — answer generation"),
    ("all-MiniLM-L6-v2", "Sentence Transformers — text embedding"),
    ("FAISS", "Facebook AI — vector similarity search"),
    ("PyPDF", "PDF text extraction"),
    ("BeautifulSoup4", "HTML text extraction"),
    ("SQLite", "Persistent chat history"),
    ("Python 3.11", "Core programming language"),
]

st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
for name, role in tech:
    st.markdown(
        f'<div class="tech-card"><div><div class="tech-name">{name}</div>'
        f'<div class="tech-role">{role}</div></div></div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# ── Project stages ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Project stages</div>', unsafe_allow_html=True)

st.markdown("""
<div class="stage-row">

  <div class="stage-card" style="border-top:3px solid #1a56db">
    <div class="stage-num">Stage 1 — Now</div>
    <div class="stage-title">Foundation and EDA</div>
    <div class="stage-items">
      RAG pipeline built<br>
      ReadDoc AI working<br>
      EDA completed<br>
      GitHub repo live<br>
      Supervisor demo
    </div>
  </div>

  <div class="stage-card" style="border-top:3px solid #1D9E75">
    <div class="stage-num">Stage 2 — Weeks 2–3</div>
    <div class="stage-title">Experiments</div>
    <div class="stage-items">
      Baseline test<br>
      9 RAG configs<br>
      Score all answers<br>
      Results analysis<br>
      Charts generated
    </div>
  </div>

  <div class="stage-card" style="border-top:3px solid #EF9F27">
    <div class="stage-num">Stage 3 — Weeks 4–6</div>
    <div class="stage-title">Dissertation</div>
    <div class="stage-items">
      Chapters 1–2<br>
      Chapters 3–4<br>
      Chapter 5<br>
      References<br>
      Appendices
    </div>
  </div>

  <div class="stage-card" style="border-top:3px solid #7F77DD">
    <div class="stage-num">Stage 4 — Final week</div>
    <div class="stage-title">Submission</div>
    <div class="stage-items">
      Deploy to cloud<br>
      Final GitHub push<br>
      Proofread<br>
      Submit PDF
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

# ── Key references ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Key references</div>', unsafe_allow_html=True)

refs = [
    ("Lewis et al., 2020", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — NeurIPS", "Foundational RAG paper — defines the architecture this system implements"),
    ("Karpukhin et al., 2020", "Dense Passage Retrieval for Open-Domain QA — EMNLP", "Basis for semantic embedding-based retrieval"),
    ("Reimers & Gurevych, 2019", "Sentence-BERT — EMNLP", "The all-MiniLM-L6-v2 embedding model used in ReadDoc AI"),
    ("Johnson et al., 2021", "Billion-scale similarity search with GPUs — IEEE", "FAISS vector index used in the pipeline"),
    ("Gao et al., 2023", "RAG for Large Language Models: A Survey — arXiv:2312.10997", "Accuracy, faithfulness, and relevance as standard RAG evaluation metrics"),
]

for author, title, relevance in refs:
    st.markdown(
        f'<div class="card" style="margin-bottom:8px">'
        f'<div class="card-title">{author}</div>'
        f'<div class="card-text" style="margin-bottom:4px"><i>{title}</i></div>'
        f'<div class="card-text" style="color:#1a56db;font-size:12px">{relevance}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )