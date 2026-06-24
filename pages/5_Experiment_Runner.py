"""
5_Experiment_Runner.py — Experiment Runner (E1–E9)
=====================================================
Scores on a 0.0–1.0 float scale (standard ML convention).
Same 10 questions as baseline, now WITH document context.
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag
from llm import ask_llama

st.set_page_config(page_title="Experiment Runner — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
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
.q-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:12px;
    padding:16px 20px;margin-bottom:16px}
.q-num{font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.q-text{font-size:14px;font-weight:600;color:#111827;margin-bottom:12px}
.answer-box{background:#fff;border:.5px solid #E5E9F5;border-radius:8px;
    padding:12px 16px;font-size:13px;color:#374151;line-height:1.7;margin-bottom:8px}
.src-tag{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:11px;
    font-weight:500;padding:2px 8px;border-radius:12px;margin:2px 3px 2px 0}
.info-box{background:#EEF2FF;border:.5px solid #C7D3F5;border-radius:10px;
    padding:14px 18px;font-size:13px;color:#1e40af;margin-bottom:1.5rem;line-height:1.6}
.score-summary{background:#0a0a0a;border-radius:10px;padding:20px 24px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0;font-size:13px}
.rubric-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.rubric-pill{background:rgba(255,255,255,0.08);border-radius:6px;padding:3px 10px;
    font-size:11.5px;color:#ccc;font-family:monospace}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">RAG evaluation</div>
<div class="page-title">Experiment runner</div>
<div class="page-sub">
    Same 10 questions WITH document context. Select experiment, generate answers,
    score on 0.0–1.0 scale. Run all 9 configurations (E1–E9).
</div>
""", unsafe_allow_html=True)

status = rag.get_status()
if not status["loaded"]:
    st.warning("No document indexed. Upload your PDF on the main ReadDoc AI page first.")
    st.stop()

EXPERIMENTS = [
    ("E1",300,3),("E2",300,5),("E3",300,10),
    ("E4",600,3),("E5",600,5),("E6",600,10),
    ("E7",1000,3),("E8",1000,5),("E9",1000,10),
]
exp_options = [f"{e} — Chunk {c} chars, k={k}" for e,c,k in EXPERIMENTS]
exp_choice  = st.selectbox("Select experiment", exp_options, index=4)
exp_idx     = exp_options.index(exp_choice)
exp_id, chunk_size, top_k = EXPERIMENTS[exp_idx]

st.info(f"Running {exp_id}: chunk size {chunk_size} chars, retrieval depth k={top_k}")

st.markdown("""
<div class="info-box">
    <b>Scoring scale (0.0–1.0):</b>
    1.00 = Perfect · 0.75 = Good · 0.50 = Moderate · 0.25 = Poor · 0.00 = Completely wrong<br>
    <b>Faithfulness:</b> are all claims traceable to the retrieved chunks? No hallucination?<br>
    <b>Q9–Q10 (out-of-scope):</b> score <b>1.00</b> if it correctly refuses ("not in documents"),
    <b>0.00</b> if it hallucinates a confident answer.
</div>
""", unsafe_allow_html=True)

SCORE_OPTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]

QUESTIONS = [
    {"id": 1,  "text": "What is the difference between supervised and unsupervised learning?", "cat": "Factual"},
    {"id": 2,  "text": "What is the bias-variance tradeoff in statistical learning?",          "cat": "Factual"},
    {"id": 3,  "text": "How does k-fold cross-validation work?",                               "cat": "Factual"},
    {"id": 4,  "text": "How does the K-Means algorithm work?",                                 "cat": "Factual"},
    {"id": 5,  "text": "What is Principal Component Analysis (PCA) and what is it used for?",  "cat": "Factual"},
    {"id": 6,  "text": "Why would you use a Random Forest instead of a single decision tree?", "cat": "Inferential"},
    {"id": 7,  "text": "What happens to a model when it is too complex or too simple?",        "cat": "Inferential"},
    {"id": 8,  "text": "What is the key idea behind how SVMs find a decision boundary?",       "cat": "Inferential"},
    {"id": 9,  "text": "What is the current stock price of Nvidia today?",                     "cat": "Out-of-scope"},
    {"id": 10, "text": "What is the weather forecast for London this weekend?",                 "cat": "Out-of-scope"},
]

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "experiments.db")

def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment   TEXT,
            question_id  INTEGER,
            question     TEXT,
            answer       TEXT,
            sources      TEXT,
            accuracy     REAL,
            relevance    REAL,
            faithfulness REAL,
            UNIQUE(experiment, question_id)
        )
    """)
    conn.commit(); conn.close()

def save_score(exp, qid, question, answer, sources, acc, rel, faith):
    conn = sqlite3.connect(DB)
    conn.execute("""
        INSERT INTO experiment_scores
            (experiment,question_id,question,answer,sources,accuracy,relevance,faithfulness)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(experiment,question_id) DO UPDATE SET
            answer=excluded.answer, sources=excluded.sources,
            accuracy=excluded.accuracy, relevance=excluded.relevance,
            faithfulness=excluded.faithfulness
    """, (exp, qid, question, answer, sources, acc, rel, faith))
    conn.commit(); conn.close()

def load_scores(exp):
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM experiment_scores WHERE experiment=? ORDER BY question_id",
            conn, params=(exp,))
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def _idx(val):
    try: return SCORE_OPTIONS.index(round(float(val), 2))
    except: return 2

init_db()
existing  = load_scores(exp_id)
saved_ids = set(existing["question_id"].tolist()) if not existing.empty else set()

st.markdown('<div class="sec">Questions and answers</div>', unsafe_allow_html=True)

indexed_paths = st.session_state.get("indexed_file_paths")
if st.button(f"Generate all 10 answers for {exp_id}", type="primary", use_container_width=True):
    if indexed_paths and chunk_size not in st.session_state.get("prebuilt_sizes", set()):
        with st.spinner(f"Switching index to chunk size {chunk_size}..."):
            rag.build_index(indexed_paths, chunk_size=chunk_size)
    prog = st.progress(0)
    for i, q in enumerate(QUESTIONS):
        prog.progress((i+1)/10, text=f"Q{q['id']}: {q['text'][:50]}...")
        chunks = rag.search(q["text"], top_k=top_k)
        answer = ask_llama(q["text"], chunks, [])
        src    = " | ".join({f"{c['file_name']} p.{c['page_number']}" for c in chunks})
        st.session_state[f"{exp_id}_ans_{q['id']}"]    = answer
        st.session_state[f"{exp_id}_src_{q['id']}"]    = src
    prog.empty()
    st.success("All answers generated — score each one below.")

for q in QUESTIONS:
    ex  = existing[existing["question_id"] == q["id"]].iloc[0] if q["id"] in saved_ids else None
    ans = st.session_state.get(f"{exp_id}_ans_{q['id']}", ex["answer"] if ex is not None else None)
    src = st.session_state.get(f"{exp_id}_src_{q['id']}", ex["sources"] if ex is not None else "")

    st.markdown(f"""
    <div class="q-card">
        <div class="q-num">Q{q['id']} · {q['cat']}</div>
        <div class="q-text">{q['text']}</div>
    </div>
    """, unsafe_allow_html=True)

    if ans:
        st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
        if src:
            tags = "".join(f'<span class="src-tag">{s.strip()}</span>' for s in src.split("|"))
            st.markdown(f"<b>Sources:</b> {tags}", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([2,2,2,1])
        with col1:
            acc   = st.selectbox("Accuracy",
                [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["accuracy"]) if ex is not None else 2,
                key=f"{exp_id}_acc_{q['id']}")
        with col2:
            rel   = st.selectbox("Relevance",
                [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["relevance"]) if ex is not None else 2,
                key=f"{exp_id}_rel_{q['id']}")
        with col3:
            faith = st.selectbox("Faithfulness",
                [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["faithfulness"]) if ex is not None else 2,
                key=f"{exp_id}_faith_{q['id']}")
        with col4:
            if st.button("Save", key=f"{exp_id}_save_{q['id']}"):
                save_score(exp_id, q["id"], q["text"], ans, src,
                           float(acc), float(rel), float(faith))
                st.rerun()
    else:
        st.caption("Generate answers first.")

df = load_scores(exp_id)
if not df.empty:
    st.markdown('<div class="sec">Results summary</div>', unsafe_allow_html=True)
    avg_acc   = round(float(df["accuracy"].mean()), 4)
    avg_rel   = round(float(df["relevance"].mean()), 4)
    avg_faith = round(float(df["faithfulness"].mean()), 4)
    overall   = round((avg_acc + avg_rel + avg_faith) / 3, 4)
    st.markdown(f"""
    <div class="score-summary">
{exp_id}  (chunk={chunk_size} chars, k={top_k})  [{len(df)}/10 scored]
══════════════════════════════════════════════════
Accuracy         {avg_acc:.4f}
Relevance        {avg_rel:.4f}
Faithfulness     {avg_faith:.4f}
──────────────────────────────────────────────────
Overall          {overall:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df[["question_id","question","accuracy","relevance","faithfulness"]].rename(columns={
            "question_id":"Q","question":"Question",
            "accuracy":"Accuracy","relevance":"Relevance","faithfulness":"Faithfulness"}),
        hide_index=True, use_container_width=True,
        column_config={
            "Accuracy":     st.column_config.ProgressColumn("Accuracy",     min_value=0, max_value=1, format="%.2f"),
            "Relevance":    st.column_config.ProgressColumn("Relevance",    min_value=0, max_value=1, format="%.2f"),
            "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=1, format="%.2f"),
        })
    st.download_button(f"Download {exp_id} CSV", data=df.to_csv(index=False),
        file_name=f"{exp_id}_results.csv", mime="text/csv")