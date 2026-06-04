"""
ReadDoc AI — Baseline Test Page
Runs all 20 questions through the LLM with zero document context.
This is the control experiment — compare results against RAG answers.
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import ask_baseline

st.set_page_config(
    page_title="Baseline Test — ReadDoc AI",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;padding:2rem 2.5rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px;letter-spacing:-.4px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:2rem;line-height:1.6}
.q-card{background:#F8FAFF;border:0.5px solid #E5E9F5;border-radius:12px;padding:16px 20px;margin-bottom:12px}
.q-label{font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.q-text{font-size:14px;font-weight:500;color:#111827;margin-bottom:10px}
.a-box{background:#fff;border:0.5px solid #E5E9F5;border-radius:8px;padding:12px 16px;font-size:13px;color:#374151;line-height:1.7;white-space:pre-wrap}
.score-row{display:flex;gap:8px;margin-top:10px;align-items:center}
.score-label{font-size:12px;color:#6B7280;min-width:130px}
.cat-header{font-size:13px;font-weight:600;color:#1a56db;background:#EEF2FF;padding:6px 14px;border-radius:20px;margin:1.5rem 0 .8rem;display:inline-block}
.warn-box{background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:14px 18px;font-size:13px;color:#92400E;margin-bottom:1.5rem;line-height:1.6}
.info-box{background:#EEF2FF;border:1px solid #C7D3F5;border-radius:10px;padding:14px 18px;font-size:14px;color:#1e40af;margin-bottom:1.5rem;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Step 4 — Baseline experiment</div>
<div class="page-title">Baseline test — no documents</div>
<div class="page-sub">
    This page asks all 20 test questions to the LLM with zero document context.<br>
    Record the answers and scores in the BASELINE row of your results sheet.<br>
    These scores will be compared against your 9 RAG experiments to show improvement.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>How this works:</b> The LLM (Llama 3.3) answers from general training knowledge only —
    no document is uploaded, no chunks are retrieved. This is your control condition.
    A good RAG system should score significantly higher than these baseline answers.
</div>
""", unsafe_allow_html=True)

# All 20 questions from the experiment
QUESTIONS = {
    "Category 1 — Factual": [
        ("Q1", "What is the difference between supervised and unsupervised learning?"),
        ("Q2", "What is the bias-variance tradeoff in statistical learning?"),
        ("Q3", "How does k-fold cross-validation work?"),
        ("Q4", "How does the K-Means algorithm work?"),
        ("Q5", "What is Principal Component Analysis (PCA) and what is it used for?"),
    ],
    "Category 2 — Multi-fact": [
        ("Q6",  "What is the difference between regression and classification, and when is each used?"),
        ("Q7",  "What evaluation metrics are used for classification models in this book?"),
        ("Q8",  "What is the difference between Bootstrap Aggregation (Bagging) and Random Forests?"),
        ("Q9",  "How do feed-forward neural networks work and how are they trained?"),
        ("Q10", "How does regularization help prevent overfitting and how does it relate to pruning in decision trees?"),
    ],
    "Category 3 — Inferential": [
        ("Q11", "Based on the book, why would you use a Random Forest instead of a single decision tree?"),
        ("Q12", "What does the book suggest happens to a model when it is too complex or too simple?"),
        ("Q13", "Based on the EM algorithm description, what are the two main steps and what does each do?"),
        ("Q14", "What does the book say about adaptive gradient methods and why are they used in deep learning?"),
        ("Q15", "Based on the Support Vector Machine section, what is the key idea behind how SVMs find a decision boundary?"),
    ],
    "Category 4 — Out-of-scope": [
        ("Q16", "What is the current version of Python released in 2025?"),
        ("Q17", "Who won the FIFA World Cup in 2022?"),
        ("Q18", "What is the stock market price of Nvidia today?"),
        ("Q19", "What are the system requirements for installing TensorFlow 3.0?"),
        ("Q20", "What is the weather forecast for Brisbane this weekend?"),
    ],
}

# Session state for answers and scores
if "baseline_answers" not in st.session_state:
    st.session_state.baseline_answers = {}
if "baseline_scores" not in st.session_state:
    st.session_state.baseline_scores = {}

# Run all button
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Run all 20 questions", type="primary", use_container_width=True):
        all_qs = [(qid, q) for qs in QUESTIONS.values() for qid, q in qs]
        prog = st.progress(0, text="Running baseline questions...")
        for i, (qid, question) in enumerate(all_qs):
            with st.spinner(f"Asking {qid}..."):
                try:
                    answer = ask_baseline(question)
                    st.session_state.baseline_answers[qid] = answer
                except Exception as e:
                    st.session_state.baseline_answers[qid] = f"Error: {str(e)}"
            prog.progress((i + 1) / len(all_qs), text=f"Done {i+1} of {len(all_qs)}")
        prog.empty()
        st.success("All 20 questions answered. Score each one below.")
        st.rerun()

with col2:
    if st.button("Clear all answers", use_container_width=True):
        st.session_state.baseline_answers = {}
        st.session_state.baseline_scores = {}
        st.rerun()

st.divider()

# Display questions and answers
all_scores = []
for category, questions in QUESTIONS.items():
    st.markdown(f'<div class="cat-header">{category}</div>', unsafe_allow_html=True)

    for qid, question in questions:
        answer = st.session_state.baseline_answers.get(qid, "")

        with st.container():
            st.markdown(f"""
            <div class="q-card">
                <div class="q-label">{qid}</div>
                <div class="q-text">{question}</div>
            </div>
            """, unsafe_allow_html=True)

            if answer:
                st.markdown(f'<div class="a-box">{answer}</div>', unsafe_allow_html=True)

                col_a, col_r, col_f = st.columns(3)
                with col_a:
                    acc = st.selectbox(
                        "Accuracy",
                        ["-", 1, 2, 3, 4, 5],
                        key=f"acc_{qid}",
                        index=st.session_state.baseline_scores.get(f"acc_{qid}", 0),
                    )
                with col_r:
                    rel = st.selectbox(
                        "Relevance",
                        ["-", 1, 2, 3, 4, 5],
                        key=f"rel_{qid}",
                        index=st.session_state.baseline_scores.get(f"rel_{qid}", 0),
                    )
                with col_f:
                    fai = st.selectbox(
                        "Faithfulness",
                        ["-", 1, 2, 3, 4, 5],
                        key=f"fai_{qid}",
                        index=st.session_state.baseline_scores.get(f"fai_{qid}", 0),
                    )

                if all(x != "-" for x in [acc, rel, fai]):
                    avg = round((acc + rel + fai) / 3, 1)
                    all_scores.append(avg)
                    st.markdown(
                        f'<div style="font-size:12px;color:#6B7280;margin-top:4px">'
                        f'Average for {qid}: <b style="color:#111827">{avg}</b></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size:13px;color:#9CA3AF;padding:8px 0">'
                    'Click "Run all 20 questions" to generate this answer.</div>',
                    unsafe_allow_html=True,
                )

# Overall baseline score summary
if all_scores:
    st.divider()
    overall = round(sum(all_scores) / len(all_scores), 2)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Questions scored", len(all_scores))
    with c2:
        st.metric("Overall baseline avg", f"{overall} / 5")
    with c3:
        st.metric("Questions remaining", 20 - len(all_scores))

    st.markdown(f"""
    <div class="warn-box">
        <b>Record this in your results sheet:</b><br>
        BASELINE row → Overall average: <b>{overall} / 5</b><br>
        This is your control score. Your RAG experiments should score higher than this.
        The bigger the difference, the stronger your dissertation result.
    </div>
    """, unsafe_allow_html=True)