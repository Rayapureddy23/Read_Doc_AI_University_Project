import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import ask_llama

st.set_page_config(page_title="Baseline Test", page_icon="🔵", layout="wide")

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

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments.db")

def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS baseline_scores (
        question_id INTEGER PRIMARY KEY, question TEXT, answer TEXT,
        accuracy REAL, relevance REAL, faithfulness REAL DEFAULT 0.0)""")
    conn.commit(); conn.close()

def save_score(qid, question, answer, acc, rel):
    conn = sqlite3.connect(DB)
    conn.execute("""INSERT INTO baseline_scores
        (question_id,question,answer,accuracy,relevance,faithfulness) VALUES (?,?,?,?,?,0.0)
        ON CONFLICT(question_id) DO UPDATE SET
        answer=excluded.answer,accuracy=excluded.accuracy,relevance=excluded.relevance""",
        (qid, question, answer, acc, rel))
    conn.commit(); conn.close()

def load_scores():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM baseline_scores ORDER BY question_id", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

init_db()
existing  = load_scores()
saved_ids = set(existing["question_id"].tolist()) if not existing.empty else set()

st.markdown('<h1>Baseline Test</h1>', unsafe_allow_html=True)
st.markdown('''**Control condition — no document context.**
Score each answer 0.00 / 0.25 / 0.50 / 0.75 / 1.00.
Faithfulness is always 0.00 (no document provided).
Q9-Q10: score 0.25 if it honestly declines, 0.00 if it hallucinates.''')

if st.button("Generate all 10 answers", type="primary", use_container_width=True):
    prog = st.progress(0)
    for i, q in enumerate(QUESTIONS):
        prog.progress((i+1)/10, text=f"Q{q['id']}: {q['text'][:50]}...")
        st.session_state[f"baseline_ans_{q['id']}"] = ask_llama(q["text"], [], [])
    prog.empty()
    st.success("All answers generated — score each one below.")

def _idx(val):
    try: return SCORE_OPTIONS.index(round(float(val), 2))
    except: return 2

for q in QUESTIONS:
    ex  = existing[existing["question_id"] == q["id"]].iloc[0] if q["id"] in saved_ids else None
    ans = st.session_state.get(f"baseline_ans_{q['id']}", ex["answer"] if ex is not None else None)
    st.markdown(f"**Q{q['id']} ({q['cat']}):** {q['text']}")
    if ans:
        st.markdown(f"> {ans[:300]}...")
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            acc = st.selectbox("Accuracy", [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["accuracy"]) if ex is not None else 2, key=f"b_acc_{q['id']}")
        with col2:
            rel = st.selectbox("Relevance", [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["relevance"]) if ex is not None else 2, key=f"b_rel_{q['id']}")
        with col3:
            if st.button("Save", key=f"b_save_{q['id']}"):
                save_score(q["id"], q["text"], ans, float(acc), float(rel))
                st.rerun()
    else:
        st.caption("Click Generate above.")
    st.divider()

df = load_scores()
if not df.empty:
    avg_acc = round(float(df["accuracy"].mean()), 4)
    avg_rel = round(float(df["relevance"].mean()), 4)
    overall = round((avg_acc + avg_rel + 0.0) / 3, 4)
    st.success(f"**Baseline results ({len(df)}/10):** Accuracy {avg_acc} · Relevance {avg_rel} · Faithfulness 0.0000 · Overall {overall}")
    st.dataframe(df[["question_id","question","accuracy","relevance","faithfulness"]], hide_index=True, use_container_width=True)
    st.download_button("Download baseline CSV", data=df.to_csv(index=False), file_name="baseline.csv", mime="text/csv")
