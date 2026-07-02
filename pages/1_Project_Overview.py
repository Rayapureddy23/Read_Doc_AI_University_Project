"""
1_Project_Overview.py — Project Overview
=========================================
"""

import streamlit as st

st.set_page_config(
    page_title="Project Overview — ReadDoc AI",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
h1{font-size:28px!important;font-weight:700!important;color:#111827!important}
h2{font-size:20px!important;font-weight:600!important;color:#111827!important;
    margin-top:2rem!important;padding-bottom:8px!important;border-bottom:2px solid #EEF2FF!important}
h3{font-size:16px!important;font-weight:600!important;color:#374151!important}
.badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:10px}
.rq-box{background:#EEF2FF;border-left:4px solid #1a56db;border-radius:0 12px 12px 0;
    padding:18px 24px;font-size:15px;color:#1e40af;font-style:italic;
    line-height:1.7;margin:1.5rem 0;font-weight:500}
.card{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:12px;
    padding:18px 22px;margin-bottom:12px}
.card-title{font-size:13px;font-weight:700;color:#1a56db;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:6px}
.card-body{font-size:13.5px;color:#374151;line-height:1.7}
.metric-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:1rem 0}
.metric-card{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:10px;
    padding:16px;text-align:center}
.metric-val{font-size:28px;font-weight:700;color:#1a56db}
.metric-label{font-size:12px;color:#6B7280;margin-top:2px}
.tech-tag{display:inline-block;background:#F3F4F6;color:#374151;font-size:12px;
    font-weight:500;padding:3px 10px;border-radius:6px;margin:3px;
    font-family:'DM Mono',monospace}
.objective{background:#F8FAFF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:10px 16px;margin-bottom:8px;font-size:13.5px;color:#374151;line-height:1.6}
.ref-item{font-size:13px;color:#374151;line-height:1.8;margin-bottom:4px;
    padding-left:16px;border-left:2px solid #E5E7EB}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">MSc Data Science</div>', unsafe_allow_html=True)
st.title("ReadDoc AI — Project Overview")
st.markdown("**Optimising Retrieval-Augmented Generation for Document Question Answering: An Empirical Study of Chunking Strategies and Retrieval Depth**")
st.markdown("---")

# ── Research Question ──────────────────────────────────────────────────────────
st.markdown("## Research Question")
st.markdown("""
<div class="rq-box">
"How does varying chunk size and retrieval depth in a Retrieval-Augmented Generation
pipeline affect the accuracy, contextual relevance, and faithfulness of answers
generated from unstructured documents?"
</div>
""", unsafe_allow_html=True)

# ── Overview ───────────────────────────────────────────────────────────────────
st.markdown("## Study Overview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">What this project does</div>
        <div class="card-body">
            ReadDoc AI is a Retrieval-Augmented Generation (RAG) system built to
            empirically investigate how two key pipeline parameters — chunk size
            and retrieval depth (k) — affect the quality of answers generated from
            unstructured PDF documents. It uses semantic vector search (FAISS) to
            retrieve relevant passages and a large language model (Llama 3.3 via
            Groq API) to generate grounded, cited answers.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Why it matters</div>
        <div class="card-body">
            RAG systems are widely deployed in enterprise AI applications, but
            practitioners often set chunking and retrieval parameters without
            systematic evaluation. This study provides empirical evidence on how
            these parameters interact to affect answer quality, helping practitioners
            make informed configuration decisions for document Q&A systems.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Experiment Design ──────────────────────────────────────────────────────────
st.markdown("## Experimental Design")
st.markdown("A **3 × 3 factorial experiment** — three chunk sizes crossed with three retrieval depths — producing nine RAG configurations, plus a zero-retrieval baseline.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">9</div>
        <div class="metric-label">RAG configurations (E1–E9)</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">10</div>
        <div class="metric-label">Test questions per configuration</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">3</div>
        <div class="metric-label">Evaluation metrics</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Independent Variables")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Chunk Size (characters)</div>
        <div class="card-body">
            <b>300</b> — Short segments, high precision, may miss broader context<br>
            <b>600</b> — Mid-range, balances context richness and retrieval precision<br>
            <b>1000</b> — Longer segments, more context per chunk, potential retrieval noise
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Retrieval Depth — k (chunks retrieved)</div>
        <div class="card-body">
            <b>k = 3</b> — Focused retrieval, higher precision, risk of missing content<br>
            <b>k = 5</b> — Balanced breadth and precision<br>
            <b>k = 10</b> — Broad retrieval, higher recall, potential noise introduction
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Experiment Configurations")
import pandas as pd
configs = []
exp_names = ["E1","E2","E3","E4","E5","E6","E7","E8","E9"]
chunk_sizes = [300,300,300,600,600,600,1000,1000,1000]
k_values = [3,5,10,3,5,10,3,5,10]
for i, (name, cs, k) in enumerate(zip(exp_names, chunk_sizes, k_values)):
    configs.append({
        "Experiment": name,
        "Chunk size (chars)": cs,
        "Retrieval depth k": k,
        "Description": f"Chunk {cs} chars, retrieve top {k} chunks"
    })
df = pd.DataFrame(configs)
st.dataframe(df, hide_index=True, use_container_width=True)

# ── Evaluation Metrics ─────────────────────────────────────────────────────────
st.markdown("## Evaluation Metrics (0.0 – 1.0 Scale)")
st.markdown("Each answer is scored by a human evaluator on three dimensions, following the framework established by Es et al. (2023).")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Accuracy</div>
        <div class="card-body">
            Is the answer factually correct based on the document content?
            Judges whether the information conveyed is true and precise.
            <br><br>
            <b>Baseline:</b> scored against general knowledge<br>
            <b>RAG:</b> scored against document ground truth
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Contextual Relevance</div>
        <div class="card-body">
            Does the answer directly address the question asked?
            Penalises off-topic responses, excessive padding, or
            answers that address a related but different question.
            <br><br>
            <b>Scale:</b> 0.00 (irrelevant) → 1.00 (directly on-point)
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">Faithfulness</div>
        <div class="card-body">
            Is every claim in the answer grounded in the retrieved chunks?
            Detects hallucination — information presented as fact that
            does not appear in the source document.
            <br><br>
            <b>Baseline:</b> always 0.00 (no document)<br>
            <b>RAG:</b> 0.00 (hallucinated) → 1.00 (fully grounded)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Objectives ─────────────────────────────────────────────────────────────────
st.markdown("## Research Objectives")
objectives = [
    "Design and implement a complete RAG pipeline with configurable chunk size and retrieval depth parameters.",
    "Establish a zero-retrieval baseline to quantify the value added by document retrieval.",
    "Evaluate nine parameter configurations (3 chunk sizes × 3 k values) using a standardised 10-question test set.",
    "Score all configurations across three metrics — accuracy, contextual relevance, and faithfulness — on a 0.0–1.0 scale.",
    "Identify the optimal chunk size and retrieval depth combination for this document type and question set.",
    "Determine the sensitivity of each metric to changes in chunk size and retrieval depth independently.",
    "Demonstrate that RAG retrieval measurably improves faithfulness over the zero-context baseline.",
    "Provide empirical evidence to support parameter selection decisions in production RAG deployments.",
]
for i, obj in enumerate(objectives, 1):
    st.markdown(f'<div class="objective"><b>O{i}.</b> {obj}</div>', unsafe_allow_html=True)

# ── Tech Stack ─────────────────────────────────────────────────────────────────
st.markdown("## Technology Stack")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Core Pipeline</div>
        <div class="card-body">
            <span class="tech-tag">Python 3.11</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">FAISS IndexFlatL2</span>
            <span class="tech-tag">all-MiniLM-L6-v2</span>
            <span class="tech-tag">sentence-transformers</span>
            <span class="tech-tag">PyPDF</span>
            <span class="tech-tag">BeautifulSoup4</span>
            <span class="tech-tag">SQLite</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Language Model</div>
        <div class="card-body">
            <span class="tech-tag">Llama 3.3 70B</span>
            <span class="tech-tag">Groq API</span>
            <span class="tech-tag">llama-3.3-70b-versatile</span>
            <br><br>
            Llama 3.3 70B via Groq's fast inference API is used for
            all answer generation. The model is held constant across
            all configurations — only chunk size and k vary.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── References ─────────────────────────────────────────────────────────────────
st.markdown("## Key References")
refs = [
    "Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. <i>arXiv:2309.15217</i>.",
    "Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. <i>Advances in Neural Information Processing Systems, 33</i>, 9459–9474.",
    "Kroese, D.P., Botev, Z., Taimre, T., & Vaisman, R. (2020). <i>Data Science and Machine Learning: Mathematical and Statistical Methods.</i> CRC Press. [Source document]",
    "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. <i>EMNLP 2019</i>.",
    "Johnson, J., Douze, M., & Jégou, H. (2021). Billion-scale similarity search with GPUs. <i>IEEE Transactions on Big Data, 7</i>(3), 535–547.",
]
for ref in refs:
    st.markdown(f'<div class="ref-item">{ref}</div>', unsafe_allow_html=True)