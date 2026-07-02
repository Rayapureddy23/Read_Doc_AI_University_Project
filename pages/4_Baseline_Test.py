import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import ask_llama

st.set_page_config(page_title="Baseline Test — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,0.06);max-width:900px}
h1{font-size:26px!important;font-weight:700!important;color:#111827!important;margin-bottom:4px!important}
.subtitle{font-size:14px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.rubric{background:#FFFBEB;border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;
    padding:12px 16px;font-size:13px;color:#92400E;margin-bottom:1.5rem;line-height:1.8}
.q-block{border:1px solid #E5E7EB;border-radius:12px;margin-bottom:20px;overflow:hidden}
.q-header{background:#F9FAFB;padding:12px 18px;border-bottom:1px solid #E5E7EB;
    display:flex;align-items:center;gap:10px}
.q-num{background:#1a56db;color:#fff;font-size:11px;font-weight:700;
    padding:2px 8px;border-radius:20px;letter-spacing:.04em}
.q-cat{background:#EEF2FF;color:#4338CA;font-size:11px;font-weight:600;
    padding:2px 8px;border-radius:20px}
.q-text{font-size:14px;font-weight:600;color:#111827}
.a-body{padding:16px 18px}
.a-label{font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:6px}
.a-text{font-size:13.5px;color:#374151;line-height:1.75;background:#F8FAFF;
    border-radius:8px;padding:14px 16px;border:1px solid #EEF2FF}
.score-row{display:grid;grid-template-columns:1fr 1fr 80px;gap:12px;
    padding:14px 18px;background:#FAFAFA;border-top:1px solid #E5E7EB;align-items:end}
.faith-badge{background:#ECFDF5;color:#059669;font-size:12px;font-weight:600;
    padding:8px 12px;border-radius:8px;text-align:center;border:1px solid #A7F3D0}
.summary-panel{background:#0a0a0a;border-radius:12px;padding:22px 28px;
    font-family:'DM Mono',monospace;color:#fff;margin:1.5rem 0;font-size:13px;line-height:2}
</style>
""", unsafe_allow_html=True)

st.markdown("# Baseline Test")
st.markdown('<div class="subtitle">Control condition — zero document context. The LLM answers from training data only. Faithfulness is fixed at 0.00 for all baseline answers.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="rubric">
<b>Scoring scale (0.0–1.0):</b> &nbsp;
1.00 = Perfect &nbsp;·&nbsp; 0.75 = Good &nbsp;·&nbsp; 0.50 = Moderate &nbsp;·&nbsp; 0.25 = Poor &nbsp;·&nbsp; 0.00 = Wrong<br>
<b>Q9–Q10 (out-of-scope):</b> score <b>0.25</b> if it honestly declines, <b>0.00</b> if it hallucinates a confident answer.
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
        (question_id,question,answer,accuracy,relevance,faithfulness) VALUES(?,?,?,?,?,0.0)
        ON CONFLICT(question_id) DO UPDATE SET
        answer=excluded.answer,accuracy=excluded.accuracy,relevance=excluded.relevance""",
        (qid,question,answer,acc,rel))
    conn.commit(); conn.close()

def load_scores():
    conn = sqlite3.connect(DB)
    try: df = pd.read_sql_query("SELECT * FROM baseline_scores ORDER BY question_id", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

def _idx(val):
    try: return SCORE_OPTIONS.index(round(float(val),2))
    except: return 2

init_db()
existing  = load_scores()
saved_ids = set(existing["question_id"].tolist()) if not existing.empty else set()

if st.button("Generate all 10 answers", type="primary", use_container_width=True):
    prog = st.progress(0)
    for i, q in enumerate(QUESTIONS):
        prog.progress((i+1)/10, text=f"Generating Q{q['id']} of 10...")
        st.session_state[f"b_ans_{q['id']}"] = ask_llama(q["text"], [], [])
    prog.empty()
    st.success("✓ All answers generated — score each one below and click Save.")

st.markdown("---")

for q in QUESTIONS:
    ex  = existing[existing["question_id"]==q["id"]].iloc[0] if q["id"] in saved_ids else None
    ans = st.session_state.get(f"b_ans_{q['id']}", ex["answer"] if ex is not None else None)
    saved_mark = " ✓" if ex is not None else ""

    st.markdown(f"""
    <div class="q-block">
      <div class="q-header">
        <span class="q-num">Q{q['id']}{saved_mark}</span>
        <span class="q-cat">{q['cat']}</span>
        <span class="q-text">{q['text']}</span>
      </div>
    """, unsafe_allow_html=True)

    if ans:
        st.markdown('<div class="a-label" style="font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;letter-spacing:.06em;padding:16px 0 6px 0">Answer</div>', unsafe_allow_html=True)
        st.markdown(ans)
        st.markdown("")

        st.markdown('<div class="score-row">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            acc = st.selectbox("Accuracy", [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["accuracy"]) if ex is not None else 2, key=f"b_acc_{q['id']}")
        with col2:
            rel = st.selectbox("Relevance", [f"{v:.2f}" for v in SCORE_OPTIONS],
                index=_idx(ex["relevance"]) if ex is not None else 2, key=f"b_rel_{q['id']}")
        with col3:
            st.selectbox("Faithfulness", [f"{v:.2f}" for v in SCORE_OPTIONS], disabled=True, key=f"b_faith_{q['id']}")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Save", key=f"b_save_{q['id']}", use_container_width=True):
                save_score(q["id"], q["text"], ans, float(acc), float(rel))
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="a-body" style="color:#9CA3AF;font-size:13px;">Generate answers first.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

df = load_scores()
if not df.empty:
    st.markdown("---")
    st.markdown("### Results summary")
    avg_acc = round(float(df["accuracy"].mean()),4)
    avg_rel = round(float(df["relevance"].mean()),4)
    overall = round((avg_acc + avg_rel + 0.0)/3,4)
    st.markdown(f"""
    <div class="summary-panel">
BASELINE — CONTROL CONDITION  [{len(df)}/10 scored]
══════════════════════════════════════════
Accuracy         {avg_acc:.4f}
Relevance        {avg_rel:.4f}
Faithfulness     0.0000  (no document)
──────────────────────────────────────────
Overall          {overall:.4f}
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        df[["question_id","question","accuracy","relevance","faithfulness"]].rename(
            columns={"question_id":"Q","question":"Question","accuracy":"Accuracy",
                     "relevance":"Relevance","faithfulness":"Faithfulness"}),
        hide_index=True, use_container_width=True,
        column_config={
            "Accuracy":     st.column_config.ProgressColumn("Accuracy",     min_value=0,max_value=1,format="%.2f"),
            "Relevance":    st.column_config.ProgressColumn("Relevance",    min_value=0,max_value=1,format="%.2f"),
            "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0,max_value=1,format="%.2f"),
        })
    st.download_button("Download baseline CSV", data=df.to_csv(index=False),
        file_name="baseline_results.csv", mime="text/csv")