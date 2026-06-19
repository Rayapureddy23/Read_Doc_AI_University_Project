"""
8_RAGAS_Evaluation.py — Automated RAG Evaluation using RAGAS
==============================================================

RAGAS (Retrieval Augmented Generation Assessment) automatically scores
RAG answers using an LLM as judge. Produces objective, reproducible scores
for Faithfulness, Answer Relevancy, and Context Precision.

This is stronger than manual scoring because:
- Scores are objective (not subject to human bias)
- Reproducible (same inputs always give same scores)
- Industry standard (used in published RAG research)
- Runs automatically on all 20 questions at once
"""

import streamlit as st
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag
from llm import ask_llama

st.set_page_config(
    page_title="RAGAS Evaluation — ReadDoc AI",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:10px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.score-box{background:#0a0a0a;border-radius:12px;padding:28px 32px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0}
.score-title{font-size:14px;font-weight:600;color:#fff;margin-bottom:16px;
    border-bottom:1px solid #333;padding-bottom:8px;letter-spacing:.08em}
.score-row{display:flex;justify-content:space-between;padding:4px 0;
    font-size:14px;color:#fff}
.score-label{color:#ccc}
.score-val{color:#4ade80;font-weight:600}
.score-overall{color:#facc15;font-weight:700;font-size:16px}
.score-divider{border-top:1px solid #333;margin:8px 0}
.info-box{background:#EEF2FF;border:.5px solid #C7D3F5;border-radius:10px;
    padding:14px 18px;font-size:13px;color:#1e40af;margin-bottom:1.5rem;line-height:1.6}
.metric-compare{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:1rem 0}
.mc{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:14px 16px;text-align:center}
.mc-val{font-size:26px;font-weight:700;color:#1a56db;margin-bottom:4px}
.mc-label{font-size:11px;color:#6B7280}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Automated Evaluation</div>
<div class="page-title">RAGAS evaluation</div>
<div class="page-sub">
    RAGAS (Retrieval Augmented Generation Assessment) automatically scores your
    RAG system using an LLM as judge. Produces objective, reproducible scores
    without manual review — industry standard for RAG evaluation.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>How RAGAS works:</b> For each question it measures (1) whether the answer
    is grounded in the retrieved chunks — <b>Faithfulness</b>, (2) whether the answer
    actually addresses the question — <b>Answer Relevancy</b>, and (3) whether the
    retrieved chunks are relevant to the question — <b>Context Precision</b>.
    These are calculated automatically using an LLM as a scoring judge,
    making them objective and reproducible across all experiments.
</div>
""", unsafe_allow_html=True)

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

if not RAGAS_AVAILABLE:
    st.error("RAGAS is not installed. Run this in your terminal first:")
    st.code("pip install ragas datasets", language="bash")
    st.stop()

# ── Check index ────────────────────────────────────────────────────────────────
status = rag.get_status()
if not status["loaded"]:
    st.warning("No document indexed. Upload your PDF on the main ReadDoc AI page first.")
    st.stop()

# ── Test questions ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec">1. Select questions to evaluate</div>',
            unsafe_allow_html=True)

TEST_QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "What is the bias-variance tradeoff in statistical learning?",
    "How does k-fold cross-validation work?",
    "How does the K-Means algorithm work?",
    "What is Principal Component Analysis (PCA) and what is it used for?",
    "What is the difference between regression and classification?",
    "What evaluation metrics are used for classification models?",
    "What is the difference between Bagging and Random Forests?",
    "How do feed-forward neural networks work and how are they trained?",
    "How does regularization help prevent overfitting?",
    "Why would you use a Random Forest instead of a single decision tree?",
    "What happens to a model when it is too complex or too simple?",
    "What are the two main steps of the EM algorithm?",
    "What are adaptive gradient methods and why are they used in deep learning?",
    "What is the key idea behind how SVMs find a decision boundary?",
    "What is the current version of Python released in 2025?",
    "Who won the FIFA World Cup in 2022?",
    "What is the stock market price of Nvidia today?",
    "What are the system requirements for installing TensorFlow 3.0?",
    "What is the weather forecast for Brisbane this weekend?",
]

col1, col2 = st.columns(2)
with col1:
    top_k = st.select_slider(
        "Retrieval depth (k)",
        options=[3, 5, 10],
        value=5,
    )
with col2:
    num_q = st.slider(
        "Number of questions to evaluate",
        min_value=5, max_value=20, value=20, step=5,
    )

questions_to_run = TEST_QUESTIONS[:num_q]
st.info(f"Will evaluate {len(questions_to_run)} questions with k={top_k}")

# ── Run RAGAS evaluation ───────────────────────────────────────────────────────
st.markdown('<div class="sec">2. Run evaluation</div>', unsafe_allow_html=True)

if st.button("Run RAGAS evaluation", type="primary", use_container_width=True):
    prog = st.progress(0, text="Generating answers...")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, question in enumerate(questions_to_run):
        prog.progress(
            (i + 1) / len(questions_to_run),
            text=f"Question {i+1}/{len(questions_to_run)}: {question[:50]}..."
        )
        chunks  = rag.search(question, top_k=top_k)
        answer  = ask_llama(question, chunks, [])
        contexts = [c["text"] for c in chunks] if chunks else ["No relevant context found."]

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append("")  # not needed for these metrics

    prog.empty()
    st.success("Answers generated. Running RAGAS scoring...")

    # Run RAGAS
    try:
        dataset = Dataset.from_dict(data)
        result  = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        faith  = round(float(result["faithfulness"]), 4)
        relev  = round(float(result["answer_relevancy"]), 4)
        prec   = round(float(result["context_precision"]), 4)
        overall = round((faith + relev + prec) / 3, 4)

        # Save to session state
        st.session_state.ragas_results = {
            "faithfulness":    faith,
            "answer_relevancy": relev,
            "context_precision": prec,
            "overall":          overall,
            "top_k":            top_k,
            "num_questions":    len(questions_to_run),
        }
        st.rerun()

    except Exception as e:
        st.error(f"RAGAS scoring failed: {str(e)}")
        st.info("This can happen if the RAGAS API key is not set. "
                "RAGAS uses OpenAI by default for its LLM judge. "
                "See the alternative below.")

# ── Show results ───────────────────────────────────────────────────────────────
if "ragas_results" in st.session_state:
    r = st.session_state.ragas_results

    st.markdown('<div class="sec">3. Results</div>', unsafe_allow_html=True)

    # Terminal-style display like the image
    st.markdown(f"""
    <div class="score-box">
        <div class="score-title">READDOC AI — RAGAS EVALUATION RESULTS</div>
        <div class="score-row">
            <span class="score-label">Faithfulness</span>
            <span class="score-val">{r['faithfulness']}</span>
        </div>
        <div class="score-row">
            <span class="score-label">Answer Relevancy</span>
            <span class="score-val">{r['answer_relevancy']}</span>
        </div>
        <div class="score-row">
            <span class="score-label">Context Precision</span>
            <span class="score-val">{r['context_precision']}</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">Overall</span>
            <span class="score-overall">{r['overall']}</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">Questions scored: {r['num_questions']}</span>
            <span class="score-label">k = {r['top_k']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown('<div class="metric-compare">', unsafe_allow_html=True)
    for label, val in [
        ("Faithfulness", r["faithfulness"]),
        ("Answer Relevancy", r["answer_relevancy"]),
        ("Context Precision", r["context_precision"]),
    ]:
        colour = "#059669" if val >= 0.8 else "#C2410C" if val < 0.6 else "#1a56db"
        st.markdown(
            f'<div class="mc"><div class="mc-val" style="color:{colour}">{val}</div>'
            f'<div class="mc-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interpretation
    st.markdown('<div class="sec">4. What these scores mean</div>',
                unsafe_allow_html=True)

    interp = []
    if r["faithfulness"] >= 0.8:
        interp.append(f"**Faithfulness {r['faithfulness']}** — Excellent. The system stays grounded in the document and rarely hallucinates. Answers can be trusted to come from the uploaded source.")
    elif r["faithfulness"] >= 0.6:
        interp.append(f"**Faithfulness {r['faithfulness']}** — Moderate. Some answers include content not directly traceable to the document. Consider increasing chunk overlap or retrieval depth.")
    else:
        interp.append(f"**Faithfulness {r['faithfulness']}** — Low. The system is hallucinating frequently. Try a larger chunk size or higher k value.")

    if r["answer_relevancy"] >= 0.8:
        interp.append(f"**Answer Relevancy {r['answer_relevancy']}** — Excellent. Answers directly address the questions asked. The semantic search is retrieving appropriate context.")
    else:
        interp.append(f"**Answer Relevancy {r['answer_relevancy']}** — The system sometimes answers a related but slightly different question. Try a different chunk size.")

    if r["context_precision"] >= 0.8:
        interp.append(f"**Context Precision {r['context_precision']}** — Excellent. FAISS is consistently retrieving the most relevant chunks first. The embedding model is working well for this document.")
    else:
        interp.append(f"**Context Precision {r['context_precision']}** — The retrieved chunks are not always the most relevant ones. Try a smaller chunk size or different k value.")

    for text in interp:
        st.markdown(text)

    # Export
    st.markdown('<div class="sec">5. Export for dissertation</div>',
                unsafe_allow_html=True)
    csv = pd.DataFrame([r]).to_csv(index=False)
    st.download_button(
        "Download RAGAS results (CSV)",
        data=csv,
        file_name="ragas_evaluation_results.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption("Include this in your dissertation appendix alongside your manual evaluation scores.")

# ── If RAGAS fails — alternative ──────────────────────────────────────────────
st.markdown('<div class="sec">Alternative — if RAGAS scoring fails</div>',
            unsafe_allow_html=True)

st.markdown("""
RAGAS uses OpenAI by default as the LLM judge for scoring. If you don't have
an OpenAI API key, you can configure it to use a local or free model instead.
Add this to your `.streamlit/secrets.toml`:
""")
st.code("""
# Use Groq's Llama as the RAGAS judge instead of OpenAI
OPENAI_API_KEY = ""   # leave empty
""", language="toml")

st.markdown("""
Or run RAGAS with your Groq key by setting the judge model before calling evaluate:
""")
st.code("""
import os
os.environ["OPENAI_API_KEY"] = "your-groq-key"

from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile"))
# Pass judge_llm to each metric when initialising
""", language="python")