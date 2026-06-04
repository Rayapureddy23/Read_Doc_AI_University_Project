"""
ReadDoc AI — Experiment Runner
Runs all 9 RAG configurations systematically and saves scores to SQLite.
"""

import streamlit as st
import sys, os, json, sqlite3, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag
from llm import ask_llama

st.set_page_config(page_title="Experiment Runner — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;padding:2rem 2.5rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px;letter-spacing:-.4px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.exp-card{border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:16px 20px;margin-bottom:10px;background:#fff}
.exp-card.active{border-color:#1a56db;background:#F8FAFF}
.exp-card.done{border-color:#10B981;background:#F0FDF4}
.exp-title{font-size:14px;font-weight:500;color:#111827;margin-bottom:4px}
.exp-meta{font-size:12px;color:#6B7280}
.badge{display:inline-block;font-size:11px;padding:2px 9px;border-radius:20px;margin-left:8px}
.b-done{background:#ECFDF5;color:#059669}
.b-active{background:#EEF2FF;color:#1a56db}
.b-pending{background:#F3F4F6;color:#6B7280}
.q-row{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:0.5px solid #F3F4F6}
.q-row:last-child{border-bottom:none}
.q-num{font-size:11px;font-weight:600;color:#9CA3AF;min-width:28px;margin-top:2px}
.q-text{font-size:13px;color:#374151;flex:1;line-height:1.5}
.a-text{font-size:12.5px;color:#1F2937;background:#F8FAFF;border-radius:6px;padding:8px 12px;margin-top:6px;line-height:1.6;border:0.5px solid #E5E9F5}
.score-pills{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
.sum-card{background:#F8FAFF;border:0.5px solid #E5E9F5;border-radius:10px;padding:14px 18px;text-align:center}
.sum-val{font-size:22px;font-weight:700;color:#1a56db}
.sum-label{font-size:12px;color:#6B7280;margin-top:2px}
</style>
""", unsafe_allow_html=True)

# ── DB setup for experiment results ──────────────────────────────────────────
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)

def init_exp_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment TEXT, chunk_size INTEGER, top_k INTEGER,
            question_id TEXT, question TEXT, answer TEXT,
            accuracy INTEGER, relevance INTEGER, faithfulness INTEGER,
            avg_score REAL, created_at TEXT
        )
    """)
    conn.commit(); conn.close()

def save_result(exp, chunk, k, qid, question, answer, acc, rel, fai):
    avg = round((acc + rel + fai) / 3, 2)
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
        (exp, chunk, k, qid, question, answer, acc, rel, fai, avg,
         datetime.datetime.utcnow().isoformat())
    )
    conn.commit(); conn.close()

def load_results(exp):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT question_id, answer, accuracy, relevance, faithfulness, avg_score FROM results WHERE experiment=?",
        (exp,)
    ).fetchall()
    conn.close()
    return {r[0]: {"answer": r[1], "accuracy": r[2], "relevance": r[3],
                   "faithfulness": r[4], "avg": r[5]} for r in rows}

def exp_avg(exp):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT AVG(avg_score) FROM results WHERE experiment=?", (exp,)).fetchone()
    conn.close()
    return round(row[0], 2) if row[0] else None

def delete_exp(exp):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM results WHERE experiment=?", (exp,))
    conn.commit(); conn.close()

init_exp_db()

# ── All 20 questions ──────────────────────────────────────────────────────────
QUESTIONS = [
    ("Q1",  "Cat1-Factual",    "What is the difference between supervised and unsupervised learning?"),
    ("Q2",  "Cat1-Factual",    "What is the bias-variance tradeoff in statistical learning?"),
    ("Q3",  "Cat1-Factual",    "How does k-fold cross-validation work?"),
    ("Q4",  "Cat1-Factual",    "How does the K-Means algorithm work?"),
    ("Q5",  "Cat1-Factual",    "What is Principal Component Analysis (PCA) and what is it used for?"),
    ("Q6",  "Cat2-MultiFact",  "What is the difference between regression and classification, and when is each used?"),
    ("Q7",  "Cat2-MultiFact",  "What evaluation metrics are used for classification models in this book?"),
    ("Q8",  "Cat2-MultiFact",  "What is the difference between Bootstrap Aggregation (Bagging) and Random Forests?"),
    ("Q9",  "Cat2-MultiFact",  "How do feed-forward neural networks work and how are they trained?"),
    ("Q10", "Cat2-MultiFact",  "How does regularization help prevent overfitting and how does it relate to pruning in decision trees?"),
    ("Q11", "Cat3-Inferential","Based on the book, why would you use a Random Forest instead of a single decision tree?"),
    ("Q12", "Cat3-Inferential","What does the book suggest happens to a model when it is too complex or too simple?"),
    ("Q13", "Cat3-Inferential","Based on the EM algorithm description, what are the two main steps and what does each do?"),
    ("Q14", "Cat3-Inferential","What does the book say about adaptive gradient methods and why are they used in deep learning?"),
    ("Q15", "Cat3-Inferential","Based on the Support Vector Machine section, what is the key idea behind how SVMs find a decision boundary?"),
    ("Q16", "Cat4-OutOfScope", "What is the current version of Python released in 2025?"),
    ("Q17", "Cat4-OutOfScope", "Who won the FIFA World Cup in 2022?"),
    ("Q18", "Cat4-OutOfScope", "What is the stock market price of Nvidia today?"),
    ("Q19", "Cat4-OutOfScope", "What are the system requirements for installing TensorFlow 3.0?"),
    ("Q20", "Cat4-OutOfScope", "What is the weather forecast for Brisbane this weekend?"),
]

# ── 9 experiment configurations ───────────────────────────────────────────────
EXPERIMENTS = [
    ("E1", 300,  3),
    ("E2", 300,  5),
    ("E3", 300, 10),
    ("E4", 600,  3),
    ("E5", 600,  5),
    ("E6", 600, 10),
    ("E7",1000,  3),
    ("E8",1000,  5),
    ("E9",1000, 10),
]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-badge">Step 5 — RAG experiments</div>
<div class="page-title">Experiment runner</div>
<div class="page-sub">
    Run all 9 configurations one by one. Each experiment uses a different
    chunk size and retrieval depth. Score every answer and the results
    are saved automatically to the database.
</div>
""", unsafe_allow_html=True)

# Check index is loaded
status = rag.get_status()
if not status["loaded"]:
    st.warning("No document index loaded. Go to the main app, upload your PDF and click Build index first.")
    st.stop()

st.markdown(
    f'<div style="font-size:13px;color:#6B7280;margin-bottom:1.5rem">'
    f'Index loaded: <b style="color:#111827">{status["total_chunks"]} chunks</b> from '
    f'<b style="color:#111827">{", ".join(status["files"])}</b></div>',
    unsafe_allow_html=True,
)

# ── Experiment overview grid ──────────────────────────────────────────────────
st.markdown("### All 9 experiments")
col_grid = st.columns(3)

for i, (exp_id, chunk, k) in enumerate(EXPERIMENTS):
    avg = exp_avg(exp_id)
    with col_grid[i % 3]:
        status_badge = (
            f'<span class="badge b-done">Done — {avg}/5</span>' if avg
            else '<span class="badge b-pending">Pending</span>'
        )
        st.markdown(
            f'<div class="exp-card {"done" if avg else ""}">'
            f'<div class="exp-title">{exp_id} {status_badge}</div>'
            f'<div class="exp-meta">Chunk {chunk} chars &nbsp;·&nbsp; k = {k}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Run one experiment ────────────────────────────────────────────────────────
st.markdown("### Run an experiment")

sel_exp = st.selectbox(
    "Select experiment",
    [f"{e} — Chunk {c}, k={k}" for e, c, k in EXPERIMENTS],
    index=0,
)
exp_id, chunk_size, top_k = EXPERIMENTS[
    ["E1","E2","E3","E4","E5","E6","E7","E8","E9"].index(sel_exp.split(" ")[0])
]

existing = load_results(exp_id)
n_done = len(existing)

colA, colB = st.columns([3, 1])
with colA:
    st.markdown(
        f'<div style="font-size:13px;color:#6B7280">'
        f'<b style="color:#111827">{exp_id}</b> — '
        f'Chunk size: <b style="color:#111827">{chunk_size}</b> chars &nbsp;·&nbsp; '
        f'k: <b style="color:#111827">{top_k}</b> &nbsp;·&nbsp; '
        f'{n_done}/20 questions answered</div>',
        unsafe_allow_html=True,
    )
with colB:
    if existing:
        if st.button("Clear this experiment", use_container_width=True):
            delete_exp(exp_id)
            st.rerun()

# Run button
if st.button(f"Run all 20 questions for {exp_id}", type="primary", use_container_width=True):
    if not status["loaded"]:
        st.error("Upload and index a document first.")
    else:
        # Rebuild index with the right chunk size for this experiment
        import glob
        upload_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "uploads"
        )
        saved_paths = glob.glob(os.path.join(upload_dir, "*"))
        if not saved_paths:
            st.error("No uploaded files found. Upload your PDF in the main app first.")
            st.stop()

        prog = st.progress(0, text=f"Rebuilding index with chunk size {chunk_size}...")
        rag.build_index(saved_paths, chunk_size=chunk_size)

        prog.progress(0.05, text="Index ready. Running questions...")
        for i, (qid, cat, question) in enumerate(QUESTIONS):
            if qid in existing:
                prog.progress((i + 1) / 20, text=f"Skipping {qid} (already done)")
                continue
            chunks = rag.search(question, top_k=top_k)
            answer = ask_llama(question, chunks, [])
            st.session_state[f"ans_{exp_id}_{qid}"] = answer
            prog.progress((i + 1) / 20, text=f"Answered {qid} ({i+1}/20)")

        prog.empty()
        st.success(f"All 20 questions answered for {exp_id}. Score each answer below.")
        st.rerun()

st.divider()

# ── Score answers ─────────────────────────────────────────────────────────────
fresh = load_results(exp_id)
session_answers = {
    qid: st.session_state.get(f"ans_{exp_id}_{qid}")
    for qid, _, _ in QUESTIONS
}

# Merge DB answers + session answers
all_answers = {}
for qid, _, _ in QUESTIONS:
    if qid in fresh:
        all_answers[qid] = fresh[qid]["answer"]
    elif session_answers.get(qid):
        all_answers[qid] = session_answers[qid]

if not all_answers:
    st.markdown(
        '<div style="text-align:center;padding:2rem;color:#9CA3AF;font-size:14px">'
        'Click the button above to run this experiment.</div>',
        unsafe_allow_html=True,
    )
else:
    cats = {}
    for qid, cat, question in QUESTIONS:
        cats.setdefault(cat, []).append((qid, question))

    all_avgs = []
    for cat, qs in cats.items():
        st.markdown(
            f'<div style="font-size:12px;font-weight:600;color:#1a56db;'
            f'background:#EEF2FF;padding:5px 14px;border-radius:20px;'
            f'display:inline-block;margin:1rem 0 .6rem">{cat.replace("-"," — ")}</div>',
            unsafe_allow_html=True,
        )
        for qid, question in qs:
            answer = all_answers.get(qid)
            saved = fresh.get(qid, {})

            st.markdown(
                f'<div class="q-row">'
                f'<div class="q-num">{qid}</div>'
                f'<div class="q-text"><b>{question}</b>'
                + (f'<div class="a-text">{answer}</div>' if answer else
                   '<div style="font-size:12px;color:#9CA3AF;margin-top:4px">Not answered yet</div>')
                + '</div></div>',
                unsafe_allow_html=True,
            )

            if answer and qid not in fresh:
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    acc = st.selectbox("Accuracy", ["-",1,2,3,4,5], key=f"acc_{exp_id}_{qid}")
                with col2:
                    rel = st.selectbox("Relevance", ["-",1,2,3,4,5], key=f"rel_{exp_id}_{qid}")
                with col3:
                    fai = st.selectbox("Faithfulness", ["-",1,2,3,4,5], key=f"fai_{exp_id}_{qid}")
                with col4:
                    if all(x != "-" for x in [acc, rel, fai]):
                        avg = round((acc + rel + fai) / 3, 1)
                        st.markdown(f'<div style="margin-top:28px;font-size:13px">Avg: <b>{avg}</b></div>', unsafe_allow_html=True)
                        if st.button("Save", key=f"save_{exp_id}_{qid}"):
                            save_result(exp_id, chunk_size, top_k, qid, question, answer, acc, rel, fai)
                            st.rerun()
            elif qid in fresh:
                s = fresh[qid]
                st.markdown(
                    f'<div style="font-size:12px;color:#059669;margin:4px 0 8px">'
                    f'Saved — Accuracy: {s["accuracy"]} · Relevance: {s["relevance"]} · '
                    f'Faithfulness: {s["faithfulness"]} · Avg: {s["avg"]}</div>',
                    unsafe_allow_html=True,
                )
                all_avgs.append(s["avg"])

    # Summary for this experiment
    if all_avgs:
        overall = round(sum(all_avgs) / len(all_avgs), 2)
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="sum-card"><div class="sum-val">{exp_id}</div><div class="sum-label">Experiment</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="sum-card"><div class="sum-val">{chunk_size}</div><div class="sum-label">Chunk size</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="sum-card"><div class="sum-val">{top_k}</div><div class="sum-label">Retrieval k</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="sum-card"><div class="sum-val" style="color:#059669">{overall}/5</div><div class="sum-label">Overall avg ({len(all_avgs)} scored)</div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:#EEF2FF;border-radius:10px;padding:12px 18px;'
            f'font-size:13px;color:#1e40af;margin-top:1rem">'
            f'Record in your results sheet → <b>{exp_id}</b> row → Overall: <b>{overall} / 5</b>'
            f'</div>',
            unsafe_allow_html=True,
        )