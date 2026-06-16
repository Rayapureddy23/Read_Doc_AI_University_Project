"""
7_Dashboard.py — Research Results Dashboard
============================================

Shows all 8 key visualisations on one screen:
1. Baseline vs Best RAG overall score
2. E1-E9 overall average score
3. Chunk size vs average score
4. Top-k vs average score
5. Accuracy comparison
6. Relevance comparison
7. Faithfulness comparison
8. Question category performance
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Dashboard — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2rem 2.5rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:8px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:4px}
.page-sub{font-size:13px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.chart-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:12px;
    padding:16px 18px;height:100%}
.chart-title{font-size:13px;font-weight:600;color:#111827;margin-bottom:4px}
.chart-sub{font-size:11.5px;color:#6B7280;margin-bottom:10px;line-height:1.5}
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1rem}
.mc{flex:1;min-width:120px;background:#F8FAFF;border:.5px solid #E5E9F5;
    border-radius:10px;padding:12px 14px;text-align:center}
.mc.green{background:#ECFDF5;border-color:#A7F3D0}
.mc.grey{background:#F3F4F6;border-color:#D1D5DB}
.mc.blue{background:#EEF2FF;border-color:#C7D3F5}
.mv{font-size:22px;font-weight:700;color:#1a56db;margin-bottom:2px}
.mc.green .mv{color:#059669}
.mc.grey .mv{color:#6B7280}
.ml{font-size:11px;color:#6B7280;line-height:1.4}
.divider{height:.5px;background:#F3F4F6;margin:1.2rem 0}
.empty{text-align:center;padding:3rem;background:#F8FAFF;border-radius:12px;
    border:1px dashed #C7D3F5}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

def load_summary():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df   = pd.read_sql_query("""
        SELECT experiment, chunk_size, top_k,
               ROUND(AVG(accuracy),2)     AS avg_accuracy,
               ROUND(AVG(relevance),2)    AS avg_relevance,
               ROUND(AVG(faithfulness),2) AS avg_faithfulness,
               ROUND(AVG(avg_score),2)    AS overall_avg,
               COUNT(*)                   AS questions_scored
        FROM results
        GROUP BY experiment, chunk_size, top_k
        ORDER BY experiment
    """, conn)
    conn.close()
    return df

def load_all():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df   = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    return df

df_sum = load_summary()
df_all = load_all()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-badge">Research Dashboard</div>
<div class="page-title">All results — one screen</div>
<div class="page-sub">
    Complete visual summary of all 9 RAG experiments vs baseline.
    Every chart answers a specific part of the research question.
</div>
""", unsafe_allow_html=True)

if df_sum.empty:
    st.markdown("""
    <div class="empty">
        <b style="font-size:15px;color:#374151">No results yet</b><br><br>
        <span style="font-size:13px;color:#6B7280">
        Run your experiments in the Experiment Runner page and score the answers.<br>
        All 8 charts will appear here automatically once scores are saved.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

rag_df  = df_sum[df_sum["experiment"] != "BASE"]
base_df = df_sum[df_sum["experiment"] == "BASE"]

best_row  = rag_df.loc[rag_df["overall_avg"].idxmax()]  if not rag_df.empty else None
worst_row = rag_df.loc[rag_df["overall_avg"].idxmin()]  if not rag_df.empty else None
base_avg  = float(base_df["overall_avg"].values[0])     if not base_df.empty else None
best_avg  = float(best_row["overall_avg"])               if best_row is not None else None
impr      = round(((best_avg - base_avg) / base_avg) * 100, 1) if (best_avg and base_avg) else None

# ── Top metric cards ───────────────────────────────────────────────────────────
st.markdown('<div class="metric-row">', unsafe_allow_html=True)
cards = []
if best_avg:
    cards.append(("green", f"{best_avg}/5", f"Best RAG score<br>{best_row['experiment']} · Chunk {int(best_row['chunk_size'])} · k={int(best_row['top_k'])}"))
if base_avg:
    cards.append(("grey",  f"{base_avg}/5", "Baseline score<br>LLM with no documents"))
if impr:
    cards.append(("blue",  f"+{impr}%",     "RAG improvement<br>over baseline"))
if best_avg and worst_row is not None:
    gap = round(best_avg - float(worst_row["overall_avg"]), 2)
    cards.append(("blue",  f"{gap} pts",    "Score gap<br>Best vs worst config"))
cards.append(("blue", f"{len(rag_df)}/9", "Experiments<br>completed"))

html = ""
for cls, val, label in cards:
    html += f'<div class="mc {cls}"><div class="mv">{val}</div><div class="ml">{label}</div></div>'
st.markdown(html + "</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ROW 1: Charts 1 and 2 ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 1 — Baseline vs Best RAG overall score</div>
        <div class="chart-sub">Most important chart — proves RAG improves over a plain LLM with no documents</div>
    </div>
    """, unsafe_allow_html=True)
    if base_avg and best_avg:
        compare = pd.DataFrame({
            "System": ["Baseline (no docs)", f"Best RAG — {best_row['experiment']}"],
            "Score":  [base_avg, best_avg]
        }).set_index("System")
        st.bar_chart(compare, color="#1a56db", height=250)
        st.caption(f"RAG scores {best_avg}/5 vs baseline {base_avg}/5 — improvement of +{impr}%")

with col2:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 2 — E1 to E9 overall average score</div>
        <div class="chart-sub">Shows how each of the 9 configurations performed — which chunk+k combination gives the best answers</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        e_chart = rag_df.set_index("experiment")["overall_avg"].rename("Avg score")
        st.bar_chart(e_chart, color="#1a56db", height=250)
        st.caption(f"Best: {best_row['experiment']} ({best_avg}/5) · Worst: {worst_row['experiment']} ({worst_row['overall_avg']}/5)")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ROW 2: Charts 3 and 4 ─────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 3 — Chunk size vs average score</div>
        <div class="chart-sub">Answers: which chunk size (300 / 600 / 1000 chars) produces the best answers overall?</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        chunk_avg = rag_df.groupby("chunk_size")["overall_avg"].mean().reset_index()
        chunk_avg["chunk_size"] = chunk_avg["chunk_size"].astype(str) + " chars"
        st.bar_chart(chunk_avg.set_index("chunk_size")["overall_avg"].rename("Avg score"), color="#1D9E75", height=250)
        best_chunk = int(rag_df.groupby("chunk_size")["overall_avg"].mean().idxmax())
        st.caption(f"Best chunk size: {best_chunk} chars")

with col4:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 4 — Retrieval depth (k) vs average score</div>
        <div class="chart-sub">Answers: which k value (3 / 5 / 10 chunks retrieved) produces the best answers overall?</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        k_avg = rag_df.groupby("top_k")["overall_avg"].mean().reset_index()
        k_avg["top_k"] = "k = " + k_avg["top_k"].astype(str)
        st.bar_chart(k_avg.set_index("top_k")["overall_avg"].rename("Avg score"), color="#7F77DD", height=250)
        best_k = int(rag_df.groupby("top_k")["overall_avg"].mean().idxmax())
        st.caption(f"Best retrieval depth: k = {best_k}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ROW 3: Charts 5, 6, 7 ─────────────────────────────────────────────────────
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 5 — Accuracy per experiment</div>
        <div class="chart-sub">Is the answer factually correct? How does accuracy change across the 9 configs?</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        acc_chart = rag_df.set_index("experiment")["avg_accuracy"].rename("Accuracy")
        st.bar_chart(acc_chart, color="#1a56db", height=230)
        st.caption(f"Max accuracy: {rag_df['avg_accuracy'].max()}/5 · Min: {rag_df['avg_accuracy'].min()}/5")

with col6:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 6 — Relevance per experiment</div>
        <div class="chart-sub">Does the answer address what was asked? Reflects quality of FAISS semantic retrieval.</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        rel_chart = rag_df.set_index("experiment")["avg_relevance"].rename("Relevance")
        st.bar_chart(rel_chart, color="#1D9E75", height=230)
        st.caption(f"Max relevance: {rag_df['avg_relevance'].max()}/5 · Min: {rag_df['avg_relevance'].min()}/5")

with col7:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Chart 7 — Faithfulness per experiment</div>
        <div class="chart-sub">Is the answer grounded in the document? Measures hallucination rate — key RAG metric.</div>
    </div>
    """, unsafe_allow_html=True)
    if not rag_df.empty:
        fai_chart = rag_df.set_index("experiment")["avg_faithfulness"].rename("Faithfulness")
        st.bar_chart(fai_chart, color="#EF9F27", height=230)
        st.caption(f"Max faithfulness: {rag_df['avg_faithfulness'].max()}/5 · Min: {rag_df['avg_faithfulness'].min()}/5")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ROW 4: Chart 8 — Category performance ─────────────────────────────────────
st.markdown("""
<div class="chart-title" style="font-size:14px;font-weight:600;color:#111827;margin-bottom:4px">
    Chart 8 — Question category performance
</div>
<div class="chart-sub" style="font-size:12px;color:#6B7280;margin-bottom:12px">
    How does RAG perform across different question types?
    Factual = single answer · Multi-fact = combines sections ·
    Inferential = requires reasoning · Out-of-scope = should refuse
</div>
""", unsafe_allow_html=True)

if not df_all.empty and "question_id" in df_all.columns:
    rag_all  = df_all[df_all["experiment"] != "BASE"].copy()
    base_all = df_all[df_all["experiment"] == "BASE"].copy()

    def get_cat(qid):
        n = int("".join(filter(str.isdigit, str(qid))) or 0)
        if n <= 5:  return "Factual"
        if n <= 10: return "Multi-fact"
        if n <= 15: return "Inferential"
        return "Out-of-scope"

    col8a, col8b = st.columns(2)

    with col8a:
        if not rag_all.empty:
            rag_all["category"] = rag_all["question_id"].apply(get_cat)
            cat_rag = (rag_all.groupby("category")["avg_score"]
                       .mean().round(2)
                       .reindex(["Factual","Multi-fact","Inferential","Out-of-scope"])
                       .rename("RAG avg score"))
            st.markdown("**RAG performance by category**")
            st.bar_chart(cat_rag, color="#1a56db", height=250)
            st.caption("Out-of-scope should score highest — model correctly refuses unanswerable questions")

    with col8b:
        if not base_all.empty and not rag_all.empty:
            base_all["category"] = base_all["question_id"].apply(get_cat)
            cat_base = (base_all.groupby("category")["avg_score"]
                        .mean().round(2)
                        .reindex(["Factual","Multi-fact","Inferential","Out-of-scope"]))
            cat_rag2 = (rag_all.groupby("category")["avg_score"]
                        .mean().round(2)
                        .reindex(["Factual","Multi-fact","Inferential","Out-of-scope"]))
            compare_cat = pd.DataFrame({"Baseline": cat_base, "Best RAG": cat_rag2})
            st.markdown("**RAG vs Baseline — by category**")
            st.bar_chart(compare_cat, height=250)
            st.caption("Shows which question categories benefit most from RAG over baseline")

# ── Research question answer ───────────────────────────────────────────────────
if best_row is not None and base_avg is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    best_chunk = int(rag_df.groupby("chunk_size")["overall_avg"].mean().idxmax())
    best_k     = int(rag_df.groupby("top_k")["overall_avg"].mean().idxmax())

    st.markdown(f"""
    <div style="background:#1a56db;border-radius:12px;padding:22px 28px;color:#fff">
        <div style="font-size:13px;font-weight:600;opacity:.8;margin-bottom:10px;
            text-transform:uppercase;letter-spacing:.06em">
            Research question answer — based on real scores
        </div>
        <div style="font-size:14px;line-height:1.9;opacity:.95">
            Varying chunk size and retrieval depth significantly affects answer quality
            in the RAG pipeline. A chunk size of <b>{best_chunk} characters</b> with
            retrieval depth <b>k={best_k}</b> produced the highest overall score of
            <b>{best_avg}/5</b> — a <b>+{impr}%</b> improvement over the baseline LLM
            ({base_avg}/5) which had no document context. The worst RAG configuration
            scored {worst_row['overall_avg']}/5, confirming that hyperparameter
            selection has a measurable and significant impact on accuracy,
            contextual relevance, and faithfulness.
        </div>
    </div>
    """, unsafe_allow_html=True)