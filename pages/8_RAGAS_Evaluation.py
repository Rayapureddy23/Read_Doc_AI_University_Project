"""
8_RAGAS_Evaluation.py — Automated RAG Evaluation using RAGAS
==============================================================
Runs RAGAS scoring for a chosen experiment configuration (E1-E9), saves
the result to the shared experiments database, displays a terminal-style
score panel for the latest run, and plots comparison charts across every
experiment evaluated so far.
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag
from llm import ask_llama

st.set_page_config(page_title="RAGAS Evaluation — ReadDoc AI", page_icon="🔵", layout="wide")

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
.score-box{background:#0a0a0a;border-radius:12px;padding:28px 32px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0}
.score-title{font-size:13px;font-weight:600;color:#fff;margin-bottom:16px;
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
.method-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:12px;
    padding:18px 22px;margin-bottom:12px}
.method-name{font-size:14px;font-weight:600;color:#111827;margin-bottom:8px}
.method-step{font-size:13px;color:#374151;line-height:1.8;margin-left:18px}
.method-formula{background:#0a0a0a;color:#4ade80;font-family:'DM Mono',monospace;
    font-size:12.5px;padding:8px 14px;border-radius:6px;margin-top:8px;display:inline-block}
.empty-box{text-align:center;padding:2.5rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Automated Evaluation</div>
<div class="page-title">RAGAS evaluation</div>
<div class="page-sub">
    RAGAS (Retrieval Augmented Generation Assessment) automatically scores your
    RAG system using an LLM as judge. Run it per experiment configuration, save
    results, and compare scores across all 9 configurations.
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

# ── RAGAS judge setup — uses Groq (already in this project) instead of
# OpenAI, which RAGAS defaults to and which this project has no key for.
GROQ_JUDGE_AVAILABLE = True
groq_judge_error = None
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq

    class _LocalEmbeddings:
        """Minimal LangChain-compatible embeddings wrapper around the
        same sentence-transformers model already used for retrieval —
        avoids needing a second embeddings provider or API key."""
        def embed_documents(self, texts):
            return rag.get_embedding_model().encode(texts).tolist()
        def embed_query(self, text):
            return rag.get_embedding_model().encode([text])[0].tolist()

    def get_ragas_judge():
        groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        judge_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)
        )
        judge_embeddings = LangchainEmbeddingsWrapper(_LocalEmbeddings())
        return judge_llm, judge_embeddings

except ImportError as e:
    GROQ_JUDGE_AVAILABLE = False
    groq_judge_error = str(e)

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

# ── Methodology — how each metric is actually calculated ──────────────────────
st.markdown('<div class="sec">How each score is calculated</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="method-card">
  <div class="method-name">Faithfulness — claim-level verification</div>
  <div class="method-step">
    1. The LLM breaks the generated answer into individual factual claims<br>
    2. Each claim is checked: can it be directly inferred from the retrieved context?<br>
    3. The fraction of claims that pass this check becomes the score
  </div>
  <div class="method-formula">Faithfulness = verified claims / total claims</div>
</div>

<div class="method-card">
  <div class="method-name">Answer relevancy — reverse-question similarity</div>
  <div class="method-step">
    1. The LLM reads the generated answer and invents several plausible questions it could be answering<br>
    2. Each reverse-engineered question is embedded, same as the original question<br>
    3. Cosine similarity is measured between each one and the original question
  </div>
  <div class="method-formula">Answer relevancy = average cosine similarity across N reverse questions</div>
</div>

<div class="method-card">
  <div class="method-name">Context precision — rank-weighted relevance</div>
  <div class="method-step">
    1. The LLM judges each retrieved chunk individually: relevant or not relevant?<br>
    2. Relevant chunks appearing earlier in the ranking are weighted more heavily<br>
    3. The weighted average becomes the score — rewarding good chunks at the top of the list
  </div>
  <div class="method-formula">Context precision = rank-weighted average of chunk relevance judgments</div>
</div>

<div class="method-card" style="background:#EEF2FF;border-color:#C7D3F5">
  <div class="method-name">Overall score</div>
  <div class="method-step">The simple average of the three metrics above.</div>
  <div class="method-formula">Overall = (Faithfulness + Answer Relevancy + Context Precision) / 3</div>
</div>

<div class="info-box" style="margin-top:4px">
  This methodology follows the RAGAS framework described in Es et al. (2023),
  <i>"RAGAS: Automated Evaluation of Retrieval Augmented Generation"</i>.
  All three metrics use an LLM as judge rather than exact-match comparison,
  because document Q&A answers can be correct while using entirely different
  wording — making traditional metrics like BLEU or ROUGE unsuitable for RAG evaluation.
</div>
""", unsafe_allow_html=True)

# ── Database — ragas_results table ─────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

def init_ragas_table():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ragas_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment         TEXT NOT NULL,
            chunk_size         INTEGER,
            top_k              INTEGER,
            faithfulness       REAL,
            answer_relevancy   REAL,
            context_precision  REAL,
            overall            REAL,
            num_questions      INTEGER,
            created_at         TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_ragas_result(experiment, chunk_size, top_k, faith, relev, prec, overall, num_q):
    conn = sqlite3.connect(DB)
    # Re-running an experiment replaces its previous RAGAS result rather than duplicating it
    conn.execute("DELETE FROM ragas_results WHERE experiment = ?", (experiment,))
    conn.execute("""
        INSERT INTO ragas_results
            (experiment, chunk_size, top_k, faithfulness, answer_relevancy,
             context_precision, overall, num_questions, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (experiment, chunk_size, top_k, faith, relev, prec, overall, num_q))
    conn.commit()
    conn.close()

def load_ragas_results():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM ragas_results ORDER BY experiment", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

init_ragas_table()

# ── Check index status ──────────────────────────────────────────────────────────
status        = rag.get_status()
indexed_paths = st.session_state.get("indexed_file_paths")

if not indexed_paths and not status["loaded"]:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No document indexed yet</b><br><br>
        Go to the main ReadDoc AI page, upload your document, and click
        "Build index — all chunk sizes" first. Then come back here.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Test questions (same 20 used across baseline + experiments) ───────────────
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

EXPERIMENTS = [
    ("E1", 300, 3),  ("E2", 300, 5),  ("E3", 300, 10),
    ("E4", 600, 3),  ("E5", 600, 5),  ("E6", 600, 10),
    ("E7", 1000, 3), ("E8", 1000, 5), ("E9", 1000, 10),
]

# ── Section 1: select experiment ───────────────────────────────────────────────
st.markdown('<div class="sec">1. Select experiment to evaluate</div>', unsafe_allow_html=True)

exp_options = [f"{eid} — Chunk {cs}, k={k}" for eid, cs, k in EXPERIMENTS]
exp_choice  = st.selectbox("Experiment configuration", exp_options, index=4)
exp_idx     = exp_options.index(exp_choice)
exp_id, chunk_size, top_k = EXPERIMENTS[exp_idx]

num_q = st.slider("Number of questions to evaluate", min_value=5, max_value=20, value=20, step=5)
questions_to_run = TEST_QUESTIONS[:num_q]

st.info(f"Will evaluate {exp_id} — chunk size {chunk_size} chars, k={top_k} — on {len(questions_to_run)} questions")

# ── Section 2: run evaluation ──────────────────────────────────────────────────
st.markdown('<div class="sec">2. Run evaluation</div>', unsafe_allow_html=True)

if st.button(f"Run RAGAS for {exp_id}", type="primary", use_container_width=True):

    if indexed_paths:
        with st.spinner(f"Preparing index for chunk size {chunk_size}..."):
            rag.build_index(indexed_paths, chunk_size=chunk_size)

    prog = st.progress(0, text="Generating answers...")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, question in enumerate(questions_to_run):
        prog.progress(
            (i + 1) / len(questions_to_run),
            text=f"Question {i+1}/{len(questions_to_run)}: {question[:50]}..."
        )
        chunks   = rag.search(question, top_k=top_k)
        answer   = ask_llama(question, chunks, [])
        contexts = [c["text"] for c in chunks] if chunks else ["No relevant context found."]

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append("")

    prog.empty()
    st.success("Answers generated. Running RAGAS scoring...")

    if not GROQ_JUDGE_AVAILABLE:
        st.warning(
            "langchain_groq is not installed, so RAGAS will fall back to its "
            "OpenAI default judge and fail without an OpenAI key. "
            "Run: pip install langchain_groq"
        )

    try:
        dataset = Dataset.from_dict(data)

        if GROQ_JUDGE_AVAILABLE:
            judge_llm, judge_embeddings = get_ragas_judge()
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
        else:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
            )

        # RAGAS returns raw scores on a 0-1 scale — converted here to a
        # 0-100% scale so they read the same way as accuracy, precision,
        # and recall are conventionally reported.
        faith   = round(float(result["faithfulness"]) * 100, 1)
        relev   = round(float(result["answer_relevancy"]) * 100, 1)
        prec    = round(float(result["context_precision"]) * 100, 1)
        overall = round((faith + relev + prec) / 3, 1)

        save_ragas_result(exp_id, chunk_size, top_k, faith, relev, prec, overall, len(questions_to_run))

        st.session_state.ragas_latest = {
            "experiment":        exp_id,
            "chunk_size":        chunk_size,
            "top_k":             top_k,
            "faithfulness":      faith,
            "answer_relevancy":  relev,
            "context_precision": prec,
            "overall":           overall,
            "num_questions":     len(questions_to_run),
        }
        st.rerun()

    except Exception as e:
        st.error(f"RAGAS scoring failed: {str(e)}")
        if "openai" in str(e).lower() or "api_key" in str(e).lower():
            st.info(
                "This looks like RAGAS tried to use OpenAI instead of Groq. "
                "Make sure langchain_groq is installed and your GROQ_API_KEY "
                "is set in .streamlit/secrets.toml."
            )

# ── Section 3: latest run — terminal panel ─────────────────────────────────────
if "ragas_latest" in st.session_state:
    r = st.session_state.ragas_latest

    st.markdown('<div class="sec">3. Latest result</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="score-box">
        <div class="score-title">READDOC AI — RAGAS EVALUATION RESULTS [{r['experiment']}]</div>
        <div class="score-row">
            <span class="score-label">Faithfulness</span>
            <span class="score-val">{r['faithfulness']}%</span>
        </div>
        <div class="score-row">
            <span class="score-label">Answer Relevancy</span>
            <span class="score-val">{r['answer_relevancy']}%</span>
        </div>
        <div class="score-row">
            <span class="score-label">Context Precision</span>
            <span class="score-val">{r['context_precision']}%</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">Overall</span>
            <span class="score-overall">{r['overall']}%</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">Chunk {r['chunk_size']} chars · k={r['top_k']} · {r['num_questions']} questions scored</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="metric-compare">', unsafe_allow_html=True)
    for label, val in [
        ("Faithfulness", r["faithfulness"]),
        ("Answer Relevancy", r["answer_relevancy"]),
        ("Context Precision", r["context_precision"]),
    ]:
        colour = "#059669" if val >= 80 else "#C2410C" if val < 60 else "#1a56db"
        st.markdown(
            f'<div class="mc"><div class="mc-val" style="color:{colour}">{val}%</div>'
            f'<div class="mc-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Section 4: results across all experiments ──────────────────────────────────
st.markdown('<div class="sec">4. RAGAS results across all experiments</div>',
            unsafe_allow_html=True)

df_ragas = load_ragas_results()

if df_ragas.empty:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No saved RAGAS results yet</b><br><br>
        Run RAGAS for at least one experiment above — every run is saved automatically,
        and this section fills in with comparison charts once you have two or more.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("**Full results table**")
    display_df = df_ragas[[
        "experiment", "chunk_size", "top_k",
        "faithfulness", "answer_relevancy", "context_precision", "overall", "num_questions"
    ]].rename(columns={
        "experiment":        "Experiment",
        "chunk_size":        "Chunk size",
        "top_k":             "k",
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "overall":           "Overall",
        "num_questions":     "Questions",
    })
    st.dataframe(
        display_df, hide_index=True, use_container_width=True,
        column_config={
            "Faithfulness":      st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=100, format="%.1f%%"),
            "Answer Relevancy":  st.column_config.ProgressColumn("Answer Relevancy", min_value=0, max_value=100, format="%.1f%%"),
            "Context Precision": st.column_config.ProgressColumn("Context Precision", min_value=0, max_value=100, format="%.1f%%"),
            "Overall":           st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f%%"),
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall RAGAS score per experiment**")
        chart_df = df_ragas.set_index("experiment")["overall"].rename("Overall score")
        st.bar_chart(chart_df, color="#1a56db", height=280)
        st.caption("Higher is better — combined average of all three metrics")

    with col2:
        st.markdown("**All three metrics, side by side**")
        metrics_df = df_ragas.set_index("experiment")[
            ["faithfulness", "answer_relevancy", "context_precision"]
        ]
        metrics_df.columns = ["Faithfulness", "Answer Relevancy", "Context Precision"]
        st.bar_chart(metrics_df, height=280)
        st.caption("Shows which metric drives the overall score for each configuration")

    if len(df_ragas) >= 2:
        st.markdown("**Chunk size vs average RAGAS score**")
        chunk_avg = df_ragas.groupby("chunk_size")["overall"].mean().reset_index()
        chunk_avg["chunk_size"] = chunk_avg["chunk_size"].astype(str) + " chars"
        st.bar_chart(chunk_avg.set_index("chunk_size")["overall"], color="#1D9E75", height=240)

    best_row = df_ragas.loc[df_ragas["overall"].idxmax()]
    st.markdown(f"""
    <div class="info-box">
        Best configuration so far: <b>{best_row['experiment']}</b>
        (chunk {int(best_row['chunk_size'])}, k={int(best_row['top_k'])})
        with an overall RAGAS score of <b>{best_row['overall']}%</b>.
        Evaluated {len(df_ragas)} of 9 experiment configurations.
    </div>
    """, unsafe_allow_html=True)

    csv = df_ragas.to_csv(index=False)
    st.download_button(
        "Download all RAGAS results (CSV)",
        data=csv,
        file_name="ragas_all_experiments.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption("Include this alongside your manual evaluation scores in the dissertation appendix.")

# ── Judge configuration note ───────────────────────────────────────────────────
st.markdown('<div class="sec">LLM judge configuration</div>',
            unsafe_allow_html=True)

if GROQ_JUDGE_AVAILABLE:
    st.markdown("""
    <div class="info-box">
        This page scores answers using <b>Llama 3.3 (via Groq)</b> as the RAGAS
        judge, and the same local <b>all-MiniLM-L6-v2</b> embedding model used
        for retrieval — no OpenAI key required. This is wired in automatically
        for every run above.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box" style="background:#FFF7ED;border-color:#FED7AA;color:#92400E">
        <b>langchain_groq is not installed</b> ({groq_judge_error}), so RAGAS
        will fall back to its OpenAI default and fail without an OpenAI key.
        Install it to use Groq as the judge instead:
    </div>
    """, unsafe_allow_html=True)
    st.code("pip install langchain_groq", language="bash")