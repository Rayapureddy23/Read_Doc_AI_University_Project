"""
8_RAGAS_Evaluation.py — RAGAS Automated Evaluation
=====================================================
ReadDoc AI | MSc Data Science and Analytics

Runs RAGAS scoring on answers already saved in the database
by the Experiment Runner. Does NOT regenerate answers — reads
them directly from the SQLite database, which means:

  1. Zero answer-generation tokens consumed here
  2. llama-3.1-8b-instant used as judge — 500K tokens/day
     completely separate from the 100K/day 70B model quota
  3. ~20,000–30,000 tokens per experiment for judging only
  4. All 9 experiments feasible within one day's free quota

RAGAS metrics computed:
  - Faithfulness      : claim-level grounding in retrieved context
  - Answer Relevancy  : does the answer address the question?
  - Context Precision : are retrieved chunks actually relevant?
"""

import streamlit as st
import sys, os, sqlite3, time, math
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag

st.set_page_config(
    page_title="RAGAS Evaluation — ReadDoc AI",
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
.info-box{background:#EEF2FF;border-left:4px solid #1a56db;border-radius:0 8px 8px 0;
    padding:14px 18px;font-size:13px;color:#1e40af;margin-bottom:1.5rem;line-height:1.6}
.warn-box{background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;
    padding:14px 18px;font-size:13px;color:#92400E;margin-bottom:1.5rem;line-height:1.6}
.score-panel{background:#0a0a0a;border-radius:12px;padding:22px 28px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0;font-size:13px;line-height:2}
.coverage-pill{display:inline-block;font-size:11px;font-weight:600;
    padding:3px 10px;border-radius:20px;margin:2px}
.cov-full{background:#ECFDF5;color:#059669}
.cov-none{background:#F3F4F6;color:#9CA3AF}
</style>
""", unsafe_allow_html=True)

st.title("RAGAS Evaluation")
st.markdown(
    "Computes RAGAS automated scores — Faithfulness, Answer Relevancy, and "
    "Context Precision — using answers already saved by the Experiment Runner. "
    "No answer regeneration. Zero additional tokens for answer generation."
)

# ── Judge setup ─────────────────────────────────────────────────────────────────
JUDGE_MODEL = "llama-3.1-8b-instant"   # 500K tokens/day — separate quota from 70B

RAGAS_AVAILABLE = True
ragas_error     = None
try:
    from ragas.llms       import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.language_models.llms import LLM
    from groq import Groq as _GroqClient
    from typing import Optional, List, Any
    import os as _os

    class _GroqJudgeLLM(LLM):
        """Minimal LangChain-compatible wrapper around Groq SDK.
        Uses llama-3.1-8b-instant which has a separate 500K token/day
        quota — completely independent of the 70B model the app uses for
        answer generation. Includes run_manager parameter (3rd positional
        argument after self) to avoid the 'takes from 2 to 3 positional
        arguments but 4 were given' crash from LangChain's base class."""
        model: str = JUDGE_MODEL

        @property
        def _llm_type(self) -> str:
            return "groq-judge"

        def _call(self, prompt: str,
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None,
                  **kwargs: Any) -> str:
            api_key = (st.secrets.get("GROQ_API_KEY") or
                       _os.environ.get("GROQ_API_KEY"))
            client = _GroqClient(api_key=api_key)
            last_err = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                    )
                    time.sleep(2)  # 30 RPM safety buffer
                    return resp.choices[0].message.content
                except Exception as e:
                    last_err = e
                    time.sleep(2 ** attempt)
            raise last_err

    class _LocalEmbeddings:
        """LangChain-compatible embeddings wrapper using the local
        all-MiniLM-L6-v2 model — no API call, no cost."""
        def embed_documents(self, texts):
            return rag.get_embedding_model().encode(texts).tolist()
        def embed_query(self, text):
            return rag.get_embedding_model().encode([text])[0].tolist()

    JUDGE_LLM  = LangchainLLMWrapper(_GroqJudgeLLM())
    JUDGE_EMBS = LangchainEmbeddingsWrapper(_LocalEmbeddings())

    # Pre-flight: one quick call to confirm the judge is reachable
    try:
        test = _GroqJudgeLLM()._call("Reply OK")
        if not test.strip():
            raise ValueError("empty response")
    except Exception as e:
        RAGAS_AVAILABLE = False
        ragas_error = str(e)

except Exception as e:
    RAGAS_AVAILABLE = False
    ragas_error = str(e)

# ── Status banner ───────────────────────────────────────────────────────────────
if RAGAS_AVAILABLE:
    st.markdown(f"""
    <div class="info-box">
        <b>Judge ready</b> — using <code>{JUDGE_MODEL}</code> via Groq
        (500,000 tokens/day, separate quota from the answer-generation model).
        Answers are loaded from the database — no regeneration needed.
        All embeddings use the local <b>all-MiniLM-L6-v2</b> model.
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(f"Judge setup failed: {ragas_error}")
    st.markdown("""
    <div class="warn-box">
        Check that <code>GROQ_API_KEY</code> is set correctly in
        <code>.streamlit/secrets.toml</code> and that the key is valid
        at console.groq.com.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Database ─────────────────────────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)
QUESTIONS = [
    {"id": 1,  "text": "What is the difference between supervised and unsupervised learning?"},
    {"id": 2,  "text": "What is the bias-variance tradeoff in statistical learning?"},
    {"id": 3,  "text": "How does k-fold cross-validation work?"},
    {"id": 4,  "text": "How does the K-Means algorithm work?"},
    {"id": 5,  "text": "What is Principal Component Analysis (PCA) and what is it used for?"},
    {"id": 6,  "text": "Why would you use a Random Forest instead of a single decision tree?"},
    {"id": 7,  "text": "What happens to a model when it is too complex or too simple?"},
    {"id": 8,  "text": "What is the key idea behind how SVMs find a decision boundary?"},
    {"id": 9,  "text": "What is the current stock price of Nvidia today?"},
    {"id": 10, "text": "What is the weather forecast for London this weekend?"},
]
EXPERIMENTS = [
    ("E1",300,3),("E2",300,5),("E3",300,10),
    ("E4",600,3),("E5",600,5),("E6",600,10),
    ("E7",1000,3),("E8",1000,5),("E9",1000,10),
]

def load_exp_answers(exp):
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM experiment_scores WHERE experiment=? ORDER BY question_id",
            conn, params=(exp,))
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def init_ragas_table():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ragas_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment       TEXT UNIQUE,
            chunk_size       INTEGER,
            top_k            INTEGER,
            faithfulness     REAL,
            answer_relevancy REAL,
            context_precision REAL,
            overall          REAL,
            num_questions    INTEGER,
            created_at       TEXT
        )""")
    conn.commit()
    conn.close()

def save_ragas_result(exp, cs, k, faith, relev, prec, overall, n):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM ragas_results WHERE experiment=?", (exp,))
    conn.execute("""
        INSERT INTO ragas_results
            (experiment,chunk_size,top_k,faithfulness,answer_relevancy,
             context_precision,overall,num_questions,created_at)
        VALUES (?,?,?,?,?,?,?,?,datetime('now'))
    """, (exp, cs, k, faith, relev, prec, overall, n))
    conn.commit()
    conn.close()

def load_ragas_results():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM ragas_results ORDER BY experiment", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

init_ragas_table()

# ── Coverage pills ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">1. Data coverage</div>', unsafe_allow_html=True)
existing_ragas = set(load_ragas_results()["experiment"].tolist()) if not load_ragas_results().empty else set()
pills = ""
for eid, cs, k in EXPERIMENTS:
    has_answers = not load_exp_answers(eid).empty
    has_ragas   = eid in existing_ragas
    if has_ragas:
        cls, label = "cov-full", f"{eid} ✓ scored"
    elif has_answers:
        cls, label = "cov-full", f"{eid} ✓ answers ready"
    else:
        cls, label = "cov-none", f"{eid} — no answers"
    pills += f'<span class="coverage-pill {cls}">{label}</span>'
st.markdown(pills, unsafe_allow_html=True)
st.caption("Only experiments with answers in the database can be scored. "
           "Run Experiment Runner first if an experiment shows 'no answers'.")

# ── Experiment selector & run ────────────────────────────────────────────────
st.markdown('<div class="sec">2. Run RAGAS scoring</div>', unsafe_allow_html=True)

exp_options = [f"{e}  —  Chunk {c} chars, k={k}" for e,c,k in EXPERIMENTS]
exp_choice  = st.selectbox("Select experiment", exp_options, index=4)
exp_idx     = exp_options.index(exp_choice)
exp_id, chunk_size, top_k = EXPERIMENTS[exp_idx]

df_answers = load_exp_answers(exp_id)
if df_answers.empty:
    st.warning(
        f"No saved answers for {exp_id}. Run the Experiment Runner for "
        f"{exp_id} first, then come back here."
    )
else:
    st.info(
        f"Found {len(df_answers)} saved answers for {exp_id}. "
        f"RAGAS will load these directly — no regeneration. "
        f"Estimated judge calls: ~{len(df_answers) * 12} "
        f"(~{len(df_answers) * 12 * 250:,} tokens on the 8B model)."
    )

    indexed_paths = st.session_state.get("indexed_file_paths")
    if st.button(
        f"Run RAGAS for {exp_id}",
        type="primary",
        use_container_width=True,
        disabled=not RAGAS_AVAILABLE
    ):
        # Switch index to correct chunk size
        if indexed_paths and chunk_size not in st.session_state.get("prebuilt_sizes", set()):
            with st.spinner(f"Switching index to chunk size {chunk_size}..."):
                rag.build_index(indexed_paths, chunk_size=chunk_size)

        # Build RAGAS dataset from saved answers + re-retrieved context
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        questions, answers, contexts, ground_truths = [], [], [], []

        prog = st.progress(0, text="Loading saved answers and re-retrieving context...")
        for i, q in enumerate(QUESTIONS):
            prog.progress((i+1)/10, text=f"Preparing Q{q['id']}/10...")
            row = df_answers[df_answers["question_id"] == q["id"]]
            if row.empty:
                continue
            answer = row.iloc[0]["answer"]
            chunks = rag.search(q["text"], top_k=top_k)
            ctx    = [c["text"] for c in chunks] if chunks else ["No context retrieved."]
            questions.append(q["text"])
            answers.append(answer)
            contexts.append(ctx)
            ground_truths.append("")  # not required by these three metrics
        prog.empty()

        if not questions:
            st.error("No valid answers found. Ensure questions are saved for this experiment.")
            st.stop()

        n = len(questions)
        st.info(
            f"Prepared {n} question-answer-context triples. "
            f"Running RAGAS judge — this takes a few minutes as each question "
            f"triggers multiple small judge calls..."
        )

        try:
            from ragas import RunConfig
            dataset = Dataset.from_dict({
                "question":     questions,
                "answer":       answers,
                "contexts":     contexts,
                "ground_truth": ground_truths,
            })
            run_cfg = RunConfig(max_workers=1, timeout=180)
            result  = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=JUDGE_LLM,
                embeddings=JUDGE_EMBS,
                run_config=run_cfg,
            )
            df_result = result.to_pandas()

            raw_faith = float(df_result["faithfulness"].mean())
            raw_relev = float(df_result["answer_relevancy"].mean())
            raw_prec  = float(df_result["context_precision"].mean())

            if all(math.isnan(v) for v in [raw_faith, raw_relev, raw_prec]):
                st.error(
                    "All metrics returned NaN — every judge call failed. "
                    "This usually means the Groq key hit a rate limit during "
                    "the burst of judge calls. Wait a minute and try again."
                )
                st.stop()

            faith   = round(raw_faith if not math.isnan(raw_faith) else 0.0, 4)
            relev   = round(raw_relev if not math.isnan(raw_relev) else 0.0, 4)
            prec    = round(raw_prec  if not math.isnan(raw_prec)  else 0.0, 4)
            overall = round((faith + relev + prec) / 3, 4)

            save_ragas_result(exp_id, chunk_size, top_k, faith, relev, prec, overall, n)
            st.session_state["ragas_latest"] = {
                "experiment": exp_id, "chunk_size": chunk_size, "top_k": top_k,
                "faithfulness": faith, "answer_relevancy": relev,
                "context_precision": prec, "overall": overall, "n": n,
            }
            st.rerun()

        except Exception as e:
            err = str(e)
            st.error(f"RAGAS scoring failed: {err}")
            if "rate_limit" in err.lower() or "429" in err or "quota" in err.lower():
                st.info(
                    "This is Groq's rate limit on the 8B judge model. "
                    "The 2-second delay between calls usually prevents this. "
                    "Wait 1 minute and try again."
                )
            elif "openai" in err.lower() or "api_key" in err.lower():
                st.info(
                    "RAGAS tried to fall back to OpenAI — the judge setup "
                    "failed. Check GROQ_API_KEY is set correctly."
                )

# ── Latest result panel ────────────────────────────────────────────────────────
if "ragas_latest" in st.session_state:
    r = st.session_state["ragas_latest"]
    st.markdown('<div class="sec">3. Latest result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="score-panel">
RAGAS SCORES — {r['experiment']}  (chunk={r['chunk_size']}, k={r['top_k']})  [n={r['n']}]
══════════════════════════════════════════════════
Judge: llama-3.1-8b-instant via Groq (local embeddings)
──────────────────────────────────────────────────
Faithfulness        {r['faithfulness']:.4f}
Answer Relevancy    {r['answer_relevancy']:.4f}
Context Precision   {r['context_precision']:.4f}
──────────────────────────────────────────────────
Overall             {r['overall']:.4f}
══════════════════════════════════════════════════
    </div>
    """, unsafe_allow_html=True)

# ── All results table ──────────────────────────────────────────────────────────
df_ragas = load_ragas_results()
if not df_ragas.empty:
    st.markdown('<div class="sec">4. RAGAS results across all experiments</div>',
                unsafe_allow_html=True)

    display = df_ragas[[
        "experiment","chunk_size","top_k",
        "faithfulness","answer_relevancy","context_precision","overall","num_questions"
    ]].rename(columns={
        "experiment":"Experiment","chunk_size":"Chunk","top_k":"k",
        "faithfulness":"Faithfulness","answer_relevancy":"Answer Relevancy",
        "context_precision":"Context Precision","overall":"Overall",
        "num_questions":"n",
    })
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        column_config={
            "Faithfulness":       st.column_config.ProgressColumn("Faithfulness",       min_value=0,max_value=1,format="%.4f"),
            "Answer Relevancy":   st.column_config.ProgressColumn("Answer Relevancy",   min_value=0,max_value=1,format="%.4f"),
            "Context Precision":  st.column_config.ProgressColumn("Context Precision",  min_value=0,max_value=1,format="%.4f"),
            "Overall":            st.column_config.ProgressColumn("Overall",            min_value=0,max_value=1,format="%.4f"),
        }
    )

    valid = df_ragas.dropna(subset=["overall"])
    if len(valid) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RAGAS Overall per experiment**")
            st.bar_chart(valid.set_index("experiment")["overall"], color="#1a56db", height=260)
        with col2:
            st.markdown("**Three RAGAS metrics**")
            st.bar_chart(
                valid.set_index("experiment")[["faithfulness","answer_relevancy","context_precision"]],
                height=260)

    csv = df_ragas.to_csv(index=False)
    st.download_button("Download RAGAS results CSV",
        data=csv, file_name="ragas_results.csv", mime="text/csv")

# ── What the metrics mean ──────────────────────────────────────────────────────
st.markdown('<div class="sec">What each RAGAS metric measures</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>Faithfulness</b> — the judge LLM decomposes the answer into individual factual
    claims, then checks each one against the retrieved context chunks.
    Score = verified claims ÷ total claims. A score of 1.0 means every claim
    in the answer is grounded in the retrieved text with no hallucination.<br><br>
    <b>Answer Relevancy</b> — the judge generates several reverse questions from the
    answer (questions the answer could be responding to), then embeds them and
    computes cosine similarity with the original question. High score = the answer
    addresses the question asked, not a related but different one.<br><br>
    <b>Context Precision</b> — the judge rates each retrieved chunk as relevant or not
    to the question, with chunks ranked earlier in the list weighted more heavily.
    Measures whether FAISS retrieved the right chunks, in the right order.<br><br>
    <b>Judge model:</b> llama-3.1-8b-instant via Groq — 500,000 tokens/day free tier,
    completely separate quota from the answer-generation model. No answers are
    regenerated on this page; they are loaded from the experiment database.
</div>
""", unsafe_allow_html=True)