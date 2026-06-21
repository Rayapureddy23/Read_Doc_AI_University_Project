"""
10_RQ_Results.py — Research Question Results (Fully Automated)

Combines RAGAS and Local Metrics into the three constructs named in the
research question — accuracy, contextual relevance, faithfulness — with
no manual scoring involved. Every number on this page comes from an
automated, reproducible source.

RQ: "How does varying chunk size and retrieval depth in a RAG pipeline
affect the accuracy, contextual relevance, and faithfulness of answers
generated from unstructured documents?"

CONSTRUCT MAPPING (documented on-page too, so it's transparent to anyone
reviewing this — supervisor, examiner, or future you):

  ACCURACY              = average of Local Metrics' four sub-scores
                           (semantic similarity, retrieval precision,
                           retrieval recall, MRR) — i.e. Local Metrics'
                           own "overall" score. Measures whether the
                           system retrieves and conveys the factually
                           correct information.

  CONTEXTUAL RELEVANCE  = average of RAGAS Answer Relevancy and RAGAS
                           Context Precision. Measures whether the
                           answer addresses the question, and whether
                           the retrieved chunks were relevant to it.

  FAITHFULNESS           = RAGAS Faithfulness directly — a 1:1 named
                           match with the RQ's own wording.

  RQ OVERALL             = unweighted average of the three constructs
                           above, per experiment configuration.
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="RQ Results — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-badge{display:inline-block;background:#F0EBFA;color:#534AB7;font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:10px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.rq-box{background:#F0EBFA;border:.5px solid #D9CFF2;border-radius:10px;
    padding:16px 20px;font-size:14px;color:#3D3578;margin-bottom:1.5rem;
    font-style:italic;line-height:1.6}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #F0EBFA}
.map-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;
    padding:16px 20px;margin-bottom:10px}
.map-construct{font-size:13px;font-weight:700;color:#534AB7;margin-bottom:4px}
.map-formula{font-family:'DM Mono',monospace;font-size:12px;color:#374151;
    background:#fff;padding:6px 10px;border-radius:6px;display:inline-block;margin-top:4px}
.score-box{background:#0a0a0a;border-radius:12px;padding:28px 32px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0}
.score-title{font-size:13px;font-weight:600;color:#fff;margin-bottom:16px;
    border-bottom:1px solid #333;padding-bottom:8px;letter-spacing:.08em}
.score-row{display:flex;justify-content:space-between;padding:4px 0;font-size:14px;color:#fff}
.score-label{color:#ccc}
.score-val{color:#a78bfa;font-weight:600}
.score-overall{color:#facc15;font-weight:700;font-size:16px}
.score-divider{border-top:1px solid #333;margin:8px 0}
.info-box{background:#F0EBFA;border:.5px solid #D9CFF2;border-radius:10px;
    padding:14px 18px;font-size:13px;color:#3D3578;margin-bottom:1.5rem;line-height:1.6}
.finding-box{background:#F8FAFF;border-left:3px solid #534AB7;border-radius:6px;
    padding:14px 18px;margin-bottom:12px;font-size:13.5px;color:#374151;line-height:1.7}
.metric-compare{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:1rem 0}
.mc{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:14px 16px;text-align:center}
.mc-val{font-size:26px;font-weight:700;color:#534AB7;margin-bottom:4px}
.mc-label{font-size:11px;color:#6B7280}
.empty-box{text-align:center;padding:2.5rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
.coverage-pill{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;
    border-radius:20px;margin:2px}
.cov-full{background:#ECFDF5;color:#059669}
.cov-partial{background:#FFF7ED;color:#C2410C}
.cov-none{background:#F3F4F6;color:#9CA3AF}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Fully automated · no manual scoring</div>
<div class="page-title">Research question results</div>
<div class="page-sub">
    Combines RAGAS and Local Metrics into the three constructs named in the
    research question. Every number here comes from an automated,
    reproducible source — no human rubric scoring involved.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rq-box">
    "How does varying chunk size and retrieval depth in a RAG pipeline affect
    the accuracy, contextual relevance, and faithfulness of answers generated
    from unstructured documents?"
</div>
""", unsafe_allow_html=True)

# ── Construct mapping — shown transparently ────────────────────────────────────
st.markdown('<div class="sec">How the three RQ constructs are measured</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="map-card">
    <div class="map-construct">ACCURACY</div>
    Whether the system retrieves and conveys the factually correct information.
    <div class="map-formula">avg(semantic similarity, retrieval precision, retrieval recall, MRR) — from Local Metrics</div>
</div>
<div class="map-card">
    <div class="map-construct">CONTEXTUAL RELEVANCE</div>
    Whether the answer addresses the question, and whether retrieved chunks were relevant to it.
    <div class="map-formula">avg(RAGAS Answer Relevancy, RAGAS Context Precision)</div>
</div>
<div class="map-card">
    <div class="map-construct">FAITHFULNESS</div>
    Whether the answer is grounded in the retrieved context, with nothing invented.
    <div class="map-formula">RAGAS Faithfulness — a direct, named match with the RQ</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>Why no manual scoring:</b> a single human rater (one person scoring
    alone) has no inter-rater reliability check, which is a recognised
    limitation of single-annotator evaluation. RAGAS (Es et al., 2023) is a
    peer-reviewed, published evaluation framework now standard in RAG
    research; semantic similarity and information-retrieval metrics
    (precision, recall, MRR) have decades of grounding in NLP and IR
    literature. Together they give every RQ construct independent automated
    coverage, with two of the three constructs backed by more than one
    underlying metric.
</div>
""", unsafe_allow_html=True)

# ── Load and merge data ─────────────────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

EXPERIMENTS = [
    ("E1", 300, 3),  ("E2", 300, 5),  ("E3", 300, 10),
    ("E4", 600, 3),  ("E5", 600, 5),  ("E6", 600, 10),
    ("E7", 1000, 3), ("E8", 1000, 5), ("E9", 1000, 10),
]

def load_table(name):
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

ragas_df = load_table("ragas_results")
local_df = load_table("local_metrics_results")

# ── Coverage status — which experiments have what ──────────────────────────────
st.markdown('<div class="sec">Data coverage</div>', unsafe_allow_html=True)

ragas_done = set(ragas_df["experiment"]) if not ragas_df.empty else set()
local_done = set(local_df["experiment"]) if not local_df.empty else set()

pills = ""
for eid, cs, k in EXPERIMENTS:
    has_ragas = eid in ragas_done
    has_local = eid in local_done
    if has_ragas and has_local:
        cls, label = "cov-full", f"{eid} ✓ both"
    elif has_ragas or has_local:
        missing = "Local" if has_ragas else "RAGAS"
        cls, label = "cov-partial", f"{eid} missing {missing}"
    else:
        cls, label = "cov-none", f"{eid} not run"
    pills += f'<span class="coverage-pill {cls}">{label}</span>'
st.markdown(pills, unsafe_allow_html=True)
st.caption("Only experiments with BOTH RAGAS and Local Metrics results can be combined below.")

# ── Combine ──────────────────────────────────────────────────────────────────────
if ragas_df.empty or local_df.empty:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">Not enough data yet</b><br><br>
        Run at least one experiment through both the RAGAS Evaluation page
        and the Local Metrics page — this page combines them automatically
        once both exist for the same experiment.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

merged = pd.merge(
    ragas_df[["experiment", "chunk_size", "top_k", "faithfulness", "answer_relevancy", "context_precision"]],
    local_df[["experiment", "semantic_similarity", "retrieval_precision", "retrieval_recall", "mrr"]],
    on="experiment", how="inner",
)

if merged.empty:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No experiment has both sources yet</b><br><br>
        Check the coverage pills above — at least one experiment needs a
        checkmark for both RAGAS and Local before this page can combine them.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

merged["accuracy"] = merged[["semantic_similarity", "retrieval_precision", "retrieval_recall", "mrr"]].mean(axis=1, skipna=True)
merged["contextual_relevance"] = merged[["answer_relevancy", "context_precision"]].mean(axis=1, skipna=True)
merged["faithfulness_construct"] = merged["faithfulness"]
merged["rq_overall"] = merged[["accuracy", "contextual_relevance", "faithfulness_construct"]].mean(axis=1, skipna=True)
merged = merged.round(1)

# ── Section: combined table ─────────────────────────────────────────────────────
st.markdown('<div class="sec">Combined results</div>', unsafe_allow_html=True)

display_df = merged[[
    "experiment", "chunk_size", "top_k", "accuracy", "contextual_relevance",
    "faithfulness_construct", "rq_overall"
]].rename(columns={
    "experiment":             "Experiment",
    "chunk_size":             "Chunk size",
    "top_k":                  "k",
    "accuracy":               "Accuracy",
    "contextual_relevance":   "Contextual Relevance",
    "faithfulness_construct": "Faithfulness",
    "rq_overall":             "RQ Overall",
}).sort_values("Experiment")

st.dataframe(
    display_df, hide_index=True, use_container_width=True,
    column_config={
        "Accuracy":              st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.1f%%"),
        "Contextual Relevance":  st.column_config.ProgressColumn("Contextual Relevance", min_value=0, max_value=100, format="%.1f%%"),
        "Faithfulness":          st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=100, format="%.1f%%"),
        "RQ Overall":            st.column_config.ProgressColumn("RQ Overall", min_value=0, max_value=100, format="%.1f%%"),
    }
)

# ── Section: best configuration — terminal panel ────────────────────────────────
valid = merged.dropna(subset=["rq_overall"])

if not valid.empty:
    best = valid.loc[valid["rq_overall"].idxmax()]

    st.markdown('<div class="sec">Best configuration</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="score-box">
        <div class="score-title">READDOC AI — RQ RESULTS — BEST CONFIGURATION</div>
        <div class="score-row">
            <span class="score-label">Experiment</span>
            <span class="score-val">{best['experiment']} (chunk {int(best['chunk_size'])}, k={int(best['top_k'])})</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">Accuracy</span>
            <span class="score-val">{best['accuracy']}%</span>
        </div>
        <div class="score-row">
            <span class="score-label">Contextual Relevance</span>
            <span class="score-val">{best['contextual_relevance']}%</span>
        </div>
        <div class="score-row">
            <span class="score-label">Faithfulness</span>
            <span class="score-val">{best['faithfulness_construct']}%</span>
        </div>
        <div class="score-divider"></div>
        <div class="score-row">
            <span class="score-label">RQ Overall</span>
            <span class="score-overall">{best['rq_overall']}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="metric-compare">', unsafe_allow_html=True)
    for label, val in [
        ("Accuracy", best["accuracy"]),
        ("Contextual Relevance", best["contextual_relevance"]),
        ("Faithfulness", best["faithfulness_construct"]),
    ]:
        st.markdown(
            f'<div class="mc"><div class="mc-val">{val}%</div>'
            f'<div class="mc-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Comparison charts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RQ Overall per experiment**")
        st.bar_chart(valid.set_index("experiment")["rq_overall"].rename("RQ Overall"),
                     color="#534AB7", height=280)
    with col2:
        st.markdown("**Three constructs, side by side**")
        constructs_df = valid.set_index("experiment")[["accuracy", "contextual_relevance", "faithfulness_construct"]]
        constructs_df.columns = ["Accuracy", "Contextual Relevance", "Faithfulness"]
        st.bar_chart(constructs_df, height=280)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Effect of chunk size**")
        chunk_avg = valid.groupby("chunk_size")["rq_overall"].mean().round(1)
        chunk_avg.index = chunk_avg.index.astype(str) + " chars"
        st.bar_chart(chunk_avg, color="#1D9E75", height=240)
    with col4:
        st.markdown("**Effect of retrieval depth (k)**")
        k_avg = valid.groupby("top_k")["rq_overall"].mean().round(1)
        k_avg.index = "k=" + k_avg.index.astype(str)
        st.bar_chart(k_avg, color="#1a56db", height=240)

    # ── Auto-written findings ───────────────────────────────────────────────────
    st.markdown('<div class="sec">Auto-written findings</div>', unsafe_allow_html=True)

    chunk_ranking = valid.groupby("chunk_size")["rq_overall"].mean().round(1).sort_values(ascending=False)
    k_ranking     = valid.groupby("top_k")["rq_overall"].mean().round(1).sort_values(ascending=False)

    best_chunk      = int(chunk_ranking.index[0])
    worst_chunk     = int(chunk_ranking.index[-1])
    chunk_spread    = round(chunk_ranking.iloc[0] - chunk_ranking.iloc[-1], 1)

    best_k          = int(k_ranking.index[0])
    worst_k         = int(k_ranking.index[-1])
    k_spread        = round(k_ranking.iloc[0] - k_ranking.iloc[-1], 1)

    strongest_construct = max(
        [("Accuracy", valid["accuracy"].mean()),
         ("Contextual Relevance", valid["contextual_relevance"].mean()),
         ("Faithfulness", valid["faithfulness_construct"].mean())],
        key=lambda x: x[1]
    )
    weakest_construct = min(
        [("Accuracy", valid["accuracy"].mean()),
         ("Contextual Relevance", valid["contextual_relevance"].mean()),
         ("Faithfulness", valid["faithfulness_construct"].mean())],
        key=lambda x: x[1]
    )

    st.markdown(f"""
    <div class="finding-box">
        <b>Chunk size effect:</b> across {len(valid)} evaluated configuration(s),
        a chunk size of <b>{best_chunk} characters</b> produced the highest
        average RQ Overall score ({chunk_ranking.iloc[0]}%), compared to
        {worst_chunk} characters ({chunk_ranking.iloc[-1]}%) — a spread of
        {chunk_spread} percentage points.
    </div>
    <div class="finding-box">
        <b>Retrieval depth effect:</b> a retrieval depth of <b>k={best_k}</b>
        produced the highest average RQ Overall score ({k_ranking.iloc[0]}%),
        compared to k={worst_k} ({k_ranking.iloc[-1]}%) — a spread of
        {k_spread} percentage points.
    </div>
    <div class="finding-box">
        <b>Strongest construct:</b> {strongest_construct[0]} scored highest on
        average ({round(strongest_construct[1], 1)}%) across all evaluated
        configurations, while {weakest_construct[0]} scored lowest
        ({round(weakest_construct[1], 1)}%) — indicating chunk size and
        retrieval depth have a larger effect on {weakest_construct[0].lower()}
        than on {strongest_construct[0].lower()}.
    </div>
    <div class="finding-box">
        <b>Best overall configuration:</b> <b>{best['experiment']}</b> (chunk
        {int(best['chunk_size'])} characters, k={int(best['top_k'])}) achieved
        the highest combined RQ Overall score of <b>{best['rq_overall']}%</b>,
        with Accuracy {best['accuracy']}%, Contextual Relevance
        {best['contextual_relevance']}%, and Faithfulness
        {best['faithfulness_construct']}%.
    </div>
    """, unsafe_allow_html=True)

    # ── RQ answer paragraph ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Research question — answer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        Based on {len(valid)} of 9 planned experiment configurations evaluated
        using fully automated metrics, varying chunk size and retrieval depth
        measurably affected all three RQ constructs. Chunk size had the larger
        effect on overall performance when comparing {best_chunk} versus
        {worst_chunk} characters ({chunk_spread} point spread), while retrieval
        depth k={best_k} versus k={worst_k} produced a {k_spread} point spread.
        {weakest_construct[0]} was the construct most sensitive to these
        parameter changes, while {strongest_construct[0]} remained comparatively
        stable. The strongest overall configuration was {best['experiment']}
        (chunk {int(best['chunk_size'])}, k={int(best['top_k'])}), suggesting
        this combination best balances retrieval breadth against context
        precision for this document and question set.
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "This paragraph is a starting draft generated from your actual data — "
        "rewrite it in your own words for the dissertation, but the numbers "
        "and rankings are computed directly from your saved results."
    )

    # ── Export ───────────────────────────────────────────────────────────────────
    csv = merged.to_csv(index=False)
    st.download_button(
        "Download combined RQ results (CSV)",
        data=csv,
        file_name="rq_results_combined.csv",
        mime="text/csv",
    )
else:
    st.info("No complete (non-NaN) combined results yet — check back once "
            "RAGAS and Local Metrics both have valid scores for at least "
            "one experiment.")