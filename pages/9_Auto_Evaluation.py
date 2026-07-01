"""
9_Auto_Evaluation.py — Fully Automated Evaluation
===================================================

import streamlit as st
import sys, os, sqlite3, time, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag
from llm import ask_llama

st.set_page_config(
    page_title="Auto Evaluation — ReadDoc AI",
    page_icon="🔵",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{font-size:26px!important;font-weight:700!important;color:#111827!important}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.pipeline-step{display:flex;align-items:center;gap:14px;padding:12px 16px;
    background:#F8FAFF;border:1px solid #E5E7EB;border-radius:10px;margin-bottom:8px}
.step-icon{font-size:20px;width:32px;text-align:center}
.step-info{flex:1}
.step-title{font-size:13px;font-weight:600;color:#111827}
.step-desc{font-size:12px;color:#6B7280;margin-top:2px}
.token-box{background:#0a0a0a;border-radius:10px;padding:16px 22px;
    font-family:'DM Mono',monospace;color:#4ade80;font-size:12.5px;
    line-height:1.9;margin:1rem 0}
.result-panel{background:#0a0a0a;border-radius:12px;padding:22px 28px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0;
    font-size:13px;line-height:2.0}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0}
.kpi{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:12px;
    padding:16px;text-align:center}
.kpi-val{font-size:28px;font-weight:700;margin-bottom:4px}
.kpi-label{font-size:11px;color:#6B7280}
.info-box{background:#EEF2FF;border-left:4px solid #1a56db;border-radius:0 8px 8px 0;
    padding:12px 16px;font-size:13px;color:#1e40af;margin:12px 0;line-height:1.6}
.warn-box{background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;
    padding:12px 16px;font-size:13px;color:#92400E;margin:12px 0;line-height:1.6}
.coverage-pill{display:inline-block;font-size:11px;font-weight:600;
    padding:3px 10px;border-radius:20px;margin:2px}
.cov-done{background:#ECFDF5;color:#059669}
.cov-empty{background:#F3F4F6;color:#9CA3AF}
</style>
""", unsafe_allow_html=True)

st.title("Fully Automated Evaluation")
st.markdown(
    "One click — generates answers, scores all 4 RAGAS metrics automatically, "
    "saves results, shows charts. Zero manual input required."
)
st.markdown("---")

# ── Reference answers for Answer Correctness metric ───────────────────────────
GROUND_TRUTH = {
    1:  "Supervised learning uses labelled data where the correct output is known during training. Unsupervised learning finds patterns in unlabelled data without predefined correct outputs.",
    2:  "The bias-variance tradeoff is the tension between underfitting (high bias from overly simple models) and overfitting (high variance from overly complex models).",
    3:  "K-fold cross-validation splits data into k equal parts. Each part serves once as the test set while the remaining k-1 parts train the model. Performance is averaged across all k iterations.",
    4:  "K-Means assigns each point to the nearest centroid, then recomputes centroids as the mean of assigned points. This repeats until centroids converge.",
    5:  "PCA is a dimensionality reduction technique that transforms data onto axes of maximum variance called principal components, ordered by explained variance.",
    6:  "Random Forests average predictions of many decision trees trained on random subsets of data and features, reducing variance and overfitting compared to a single tree.",
    7:  "An overly complex model overfits training data and fails to generalise. An overly simple model underfits and cannot capture the underlying pattern.",
    8:  "SVMs find the hyperplane that maximises the margin between classes, defined by the support vectors — training points closest to the decision boundary.",
    9:  "I could not find this in your uploaded documents.",
    10: "I could not find this in your uploaded documents.",
}

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

EXPERIMENTS = [
    ("E1",300,3),("E2",300,5),("E3",300,10),
    ("E4",600,3),("E5",600,5),("E6",600,10),
    ("E7",1000,3),("E8",1000,5),("E9",1000,10),
]

JUDGE_MODEL = "llama-3.1-8b-instant"

# ── Judge setup ──────────────────────────────────────────────────────────────────
JUDGE_OK    = True
judge_error = None
try:
    from ragas.llms       import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.language_models.llms import LLM
    from groq import Groq as _Groq
    from typing import Optional, List, Any

    class _JudgeLLM(LLM):
        model: str = JUDGE_MODEL

        @property
        def _llm_type(self) -> str:
            return "groq-judge"

        def _call(self, prompt: str,
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any) -> str:
            key    = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            client = _Groq(api_key=key)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                    )
                    time.sleep(2)
                    return resp.choices[0].message.content
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(2 ** attempt)
            return ""

    class _LocalEmb:
        def embed_documents(self, texts):
            return rag.get_embedding_model().encode(texts).tolist()
        def embed_query(self, text):
            return rag.get_embedding_model().encode([text])[0].tolist()

    JUDGE_LLM  = LangchainLLMWrapper(_JudgeLLM())
    JUDGE_EMBS = LangchainEmbeddingsWrapper(_LocalEmb())
    _test = _JudgeLLM()._call("Reply: OK")
    if not _test.strip():
        raise ValueError("Empty response")

except Exception as e:
    JUDGE_OK    = False
    judge_error = str(e)

# ── Database ──────────────────────────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiment_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT, question_id INTEGER, question TEXT,
        answer TEXT, sources TEXT,
        accuracy REAL, relevance REAL, faithfulness REAL,
        UNIQUE(experiment, question_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS auto_eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT UNIQUE,
        chunk_size INTEGER, top_k INTEGER,
        faithfulness REAL, answer_relevancy REAL,
        context_precision REAL, answer_correctness REAL,
        overall REAL, num_questions INTEGER,
        created_at TEXT)""")
    conn.commit(); conn.close()

def save_answer(exp, qid, question, answer, src):
    conn = sqlite3.connect(DB)
    conn.execute("""INSERT INTO experiment_scores
        (experiment,question_id,question,answer,sources,accuracy,relevance,faithfulness)
        VALUES(?,?,?,?,?,0.5,0.5,0.5)
        ON CONFLICT(experiment,question_id) DO UPDATE SET
        answer=excluded.answer,sources=excluded.sources""",
        (exp, qid, question, answer, src))
    conn.commit(); conn.close()

def save_result(exp, cs, k, f, ar, cp, ac, overall, n):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM auto_eval_results WHERE experiment=?", (exp,))
    conn.execute("""INSERT INTO auto_eval_results
        (experiment,chunk_size,top_k,faithfulness,answer_relevancy,
         context_precision,answer_correctness,overall,num_questions,created_at)
        VALUES(?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (exp, cs, k, f, ar, cp, ac, overall, n))
    conn.commit(); conn.close()

def load_results():
    if not os.path.exists(DB): return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM auto_eval_results ORDER BY experiment", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

init_db()

# ── Pipeline explanation ──────────────────────────────────────────────────────
st.markdown('<div class="sec">What happens when you click Run</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pipeline-step">
  <div class="step-icon">📄</div>
  <div class="step-info">
    <div class="step-title">Step 1 — Switch FAISS index to the selected chunk size</div>
    <div class="step-desc">Loads the pre-built index instantly from cache</div>
  </div>
</div>
<div class="pipeline-step">
  <div class="step-icon">🤖</div>
  <div class="step-info">
    <div class="step-title">Step 2 — Generate 10 answers using the LLM</div>
    <div class="step-desc">Retrieves top-k chunks per question → LLM generates grounded answer → saves to DB</div>
  </div>
</div>
<div class="pipeline-step">
  <div class="step-icon">⚖️</div>
  <div class="step-info">
    <div class="step-title">Step 3 — RAGAS judge scores all 4 metrics</div>
    <div class="step-desc">Faithfulness · Answer Relevancy · Context Precision · Answer Correctness</div>
  </div>
</div>
<div class="pipeline-step">
  <div class="step-icon">📊</div>
  <div class="step-info">
    <div class="step-title">Step 4 — Save results and show charts</div>
    <div class="step-desc">Terminal panel · KPI cards · Radar chart · Grouped bar chart · CSV download</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="token-box">
TOKEN BUDGET  (llama-3.1-8b-instant — 500,000 tokens/day free)
═══════════════════════════════════════════════════════════════
Answer generation   10 questions × ~600 tokens  =   6,000
RAGAS judging       ~200 calls   × ~250 tokens  =  50,000
Per experiment                                  =  56,000
All 9 experiments   9 × 56,000                  = 504,000
→ Plan: 5 experiments today, 4 tomorrow
</div>
""", unsafe_allow_html=True)

if JUDGE_OK:
    st.success(f"✓ Judge ready — {JUDGE_MODEL} via Groq · Local all-MiniLM-L6-v2 embeddings")
else:
    st.error(f"Judge setup failed: {judge_error}")
    st.markdown('<div class="warn-box">Check GROQ_API_KEY in .streamlit/secrets.toml</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Coverage ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Coverage</div>', unsafe_allow_html=True)
existing = load_results()
done_set = set(existing["experiment"].tolist()) if not existing.empty else set()
pills    = "".join(
    f'<span class="coverage-pill {"cov-done" if e in done_set else "cov-empty"}">'
    f'{e} {"✓" if e in done_set else "—"}</span>'
    for e,_,_ in EXPERIMENTS
)
st.markdown(pills, unsafe_allow_html=True)

# ── Selector and run ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">Run automated evaluation</div>', unsafe_allow_html=True)

status        = rag.get_status()
indexed_paths = st.session_state.get("indexed_file_paths")

if not indexed_paths and not status["loaded"]:
    st.warning("Upload and index your document on the main page first.")
    st.stop()

exp_options = [f"{e}  —  Chunk {c} chars, k={k}" for e,c,k in EXPERIMENTS]
exp_choice  = st.selectbox("Select experiment", exp_options, index=4)
exp_idx     = exp_options.index(exp_choice)
exp_id, chunk_size, top_k = EXPERIMENTS[exp_idx]

if exp_id in done_set:
    st.info(f"{exp_id} already evaluated. Click Run to re-evaluate and overwrite.")

if st.button(
    f"▶  Auto-evaluate {exp_id}  (chunk {chunk_size}, k={top_k})",
    type="primary", use_container_width=True, disabled=not JUDGE_OK
):
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import (faithfulness, answer_relevancy,
                                context_precision, answer_correctness)

    # Step 1 — switch index
    if indexed_paths:
        with st.spinner("Switching index..."):
            rag.build_index(indexed_paths, chunk_size=chunk_size)

    # Step 2 — generate answers
    st.markdown("**Step 2 — Generating 10 answers...**")
    prog = st.progress(0)
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, q in enumerate(QUESTIONS):
        prog.progress((i+1)/10, text=f"Q{q['id']}/10 — {q['text'][:55]}...")
        chunks = rag.search(q["text"], top_k=top_k)
        answer = ask_llama(q["text"], chunks, [])
        src    = " | ".join({f"p.{c['page_number']}" for c in chunks})
        ctx    = [c["text"] for c in chunks] if chunks else ["No context retrieved."]
        save_answer(exp_id, q["id"], q["text"], answer, src)
        questions.append(q["text"])
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(GROUND_TRUTH.get(q["id"], ""))

    prog.empty()
    st.success(f"✓ 10 answers generated and saved for {exp_id}")

    # Step 3 — RAGAS scoring
    st.markdown("**Step 3 — Running RAGAS (4 metrics)...**")
    st.caption("This takes a few minutes — each question triggers ~20 judge calls")

    try:
        dataset = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_precision, answer_correctness],
            llm=JUDGE_LLM,
            embeddings=JUDGE_EMBS,
            run_config=RunConfig(max_workers=1, timeout=300),
        )
        df_r = result.to_pandas()

        def safe(col):
            if col not in df_r.columns: return 0.0
            v = float(df_r[col].mean())
            return 0.0 if math.isnan(v) else round(v, 4)

        f  = safe("faithfulness")
        ar = safe("answer_relevancy")
        cp = safe("context_precision")
        ac = safe("answer_correctness")
        ov = round((f + ar + cp + ac) / 4, 4)

        save_result(exp_id, chunk_size, top_k, f, ar, cp, ac, ov, len(questions))
        st.session_state["auto_latest"] = {
            "experiment": exp_id, "chunk_size": chunk_size, "top_k": top_k,
            "faithfulness": f, "answer_relevancy": ar,
            "context_precision": cp, "answer_correctness": ac,
            "overall": ov, "n": len(questions),
        }
        st.rerun()

    except Exception as e:
        err = str(e)
        st.error(f"RAGAS scoring failed: {err}")
        if "rate_limit" in err.lower() or "429" in err:
            st.info("Rate limit hit — wait 1 minute and try again.")
        elif "openai" in err.lower():
            st.info("Check GROQ_API_KEY is set correctly.")

# ── Latest result ─────────────────────────────────────────────────────────────
if "auto_latest" in st.session_state:
    r = st.session_state["auto_latest"]
    st.markdown('<div class="sec">Latest result</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-panel">
AUTO EVAL — {r['experiment']}  chunk={r['chunk_size']}  k={r['top_k']}  n={r['n']}
Judge: {JUDGE_MODEL} · Embeddings: all-MiniLM-L6-v2 (local)
══════════════════════════════════════════════════════════════
Faithfulness        {r['faithfulness']:.4f}   claim-level grounding in retrieved context
Answer Relevancy    {r['answer_relevancy']:.4f}   does the answer address the question?
Context Precision   {r['context_precision']:.4f}   are retrieved chunks actually relevant?
Answer Correctness  {r['answer_correctness']:.4f}   semantic match vs reference answer
══════════════════════════════════════════════════════════════
Overall             {r['overall']:.4f}   mean of 4 metrics
    </div>
    """, unsafe_allow_html=True)

    colours = ["#1a56db","#7C3AED","#059669","#DC2626"]
    labels  = ["Faithfulness","Answer Relevancy","Context Precision","Answer Correctness"]
    vals    = [r["faithfulness"],r["answer_relevancy"],r["context_precision"],r["answer_correctness"]]
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    for label, val, colour in zip(labels, vals, colours):
        st.markdown(
            f'<div class="kpi"><div class="kpi-val" style="color:{colour}">'
            f'{val:.4f}</div><div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar
    cats   = labels + [labels[0]]
    vvals  = vals   + [vals[0]]
    fig_r  = go.Figure(go.Scatterpolar(
        r=vvals, theta=cats, fill="toself", name=r["experiment"],
        line_color="#1a56db", fillcolor="rgba(26,86,219,0.15)",
    ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=360, margin=dict(t=50,b=20),
        paper_bgcolor="#FFFFFF", font=dict(family="DM Sans"),
        title=f"Metric profile — {r['experiment']}",
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ── All results ───────────────────────────────────────────────────────────────
df_all = load_results()
if not df_all.empty:
    st.markdown('<div class="sec">All automated evaluation results</div>',
                unsafe_allow_html=True)

    st.dataframe(
        df_all[["experiment","chunk_size","top_k","faithfulness",
                "answer_relevancy","context_precision","answer_correctness","overall"
               ]].rename(columns={
                   "experiment":"Experiment","chunk_size":"Chunk","top_k":"k",
                   "faithfulness":"Faithfulness","answer_relevancy":"Ans.Relevancy",
                   "context_precision":"Ctx.Precision","answer_correctness":"Ans.Correctness",
                   "overall":"Overall",
               }),
        hide_index=True, use_container_width=True,
        column_config={
            "Faithfulness":    st.column_config.ProgressColumn("Faithfulness",    min_value=0,max_value=1,format="%.4f"),
            "Ans.Relevancy":   st.column_config.ProgressColumn("Ans.Relevancy",   min_value=0,max_value=1,format="%.4f"),
            "Ctx.Precision":   st.column_config.ProgressColumn("Ctx.Precision",   min_value=0,max_value=1,format="%.4f"),
            "Ans.Correctness": st.column_config.ProgressColumn("Ans.Correctness", min_value=0,max_value=1,format="%.4f"),
            "Overall":         st.column_config.ProgressColumn("Overall",         min_value=0,max_value=1,format="%.4f"),
        })

    if len(df_all) >= 2:
        fig_all = go.Figure()
        for metric, colour, label in [
            ("faithfulness","#1a56db","Faithfulness"),
            ("answer_relevancy","#7C3AED","Answer Relevancy"),
            ("context_precision","#059669","Context Precision"),
            ("answer_correctness","#DC2626","Answer Correctness"),
        ]:
            fig_all.add_trace(go.Bar(
                name=label, x=df_all["experiment"],
                y=df_all[metric].round(4), marker_color=colour,
                text=df_all[metric].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
            ))
        fig_all.update_layout(
            barmode="group",
            yaxis=dict(title="Score (0.0–1.0)", range=[0,1.2]),
            xaxis_title="Experiment", height=420,
            margin=dict(t=50,b=40),
            plot_bgcolor="#F8FAFF", paper_bgcolor="#FFFFFF",
            font=dict(family="DM Sans"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            title="All 4 RAGAS metrics across experiment configurations",
        )
        st.plotly_chart(fig_all, use_container_width=True)

        best = df_all.loc[df_all["overall"].idxmax()]
        st.markdown(f"""
        <div class="info-box">
            <b>Best configuration:</b> {best['experiment']}
            (chunk {int(best['chunk_size'])} chars, k={int(best['top_k'])}) —
            Overall {best['overall']:.4f} ·
            Faithfulness {best['faithfulness']:.4f} ·
            Answer Relevancy {best['answer_relevancy']:.4f} ·
            Context Precision {best['context_precision']:.4f} ·
            Answer Correctness {best['answer_correctness']:.4f}
        </div>
        """, unsafe_allow_html=True)

    st.download_button(
        "Download automated evaluation CSV",
        data=df_all.to_csv(index=False),
        file_name="auto_evaluation_results.csv",
        mime="text/csv",
    )

# ── Methodology note ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">How each metric is computed</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>Faithfulness</b> — judge extracts individual claims from the answer,
    verifies each against retrieved chunks. Score = verified / total claims.<br><br>
    <b>Answer Relevancy</b> — judge generates reverse questions from the answer,
    embeds them, computes cosine similarity with the original question.<br><br>
    <b>Context Precision</b> — judge rates each retrieved chunk relevant or not,
    weighted by rank position (earlier ranks matter more).<br><br>
    <b>Answer Correctness</b> — semantic similarity between generated answer and
    the ground-truth reference answer, combined with factual overlap. This is the
    fourth metric that most directly measures whether the answer is correct.<br><br>
    <b>Overall</b> — unweighted mean of all four metrics.<br><br>
    <b>Judge:</b> llama-3.1-8b-instant via Groq (500K tokens/day free).
    <b>Embeddings:</b> all-MiniLM-L6-v2 (fully local, zero API cost).
</div>
""", unsafe_allow_html=True)