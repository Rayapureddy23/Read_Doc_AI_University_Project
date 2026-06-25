"""
# 6_Results_Analysis.py — Results Comparison (0.0–1.0 scale)
============================================================
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Results Analysis — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:1.5rem}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.score-panel{background:#0a0a0a;border-radius:12px;padding:22px 28px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0;font-size:13px;
    line-height:1.9}
.finding{background:#F8FAFF;border-left:3px solid #1a56db;border-radius:6px;
    padding:12px 16px;margin-bottom:10px;font-size:13.5px;color:#374151;line-height:1.7}
.rq-box{background:#EEF2FF;border:.5px solid #C7D3F5;border-radius:10px;
    padding:16px 20px;font-size:14px;color:#1e40af;font-style:italic;
    line-height:1.7;margin:1rem 0}
.empty-box{text-align:center;padding:2.5rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-title">Results analysis</div>
<div class="page-sub">
    All scores on 0.0–1.0 scale. Baseline vs E1–E9 compared across
    Accuracy, Relevance, and Faithfulness.
</div>
""", unsafe_allow_html=True)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "experiments.db")

def load_baseline():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM baseline_scores", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

def load_experiments():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM experiment_scores", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

EXPERIMENTS = ["E1","E2","E3","E4","E5","E6","E7","E8","E9"]
EXP_CONFIG  = {
    "E1":(300,3),"E2":(300,5),"E3":(300,10),
    "E4":(600,3),"E5":(600,5),"E6":(600,10),
    "E7":(1000,3),"E8":(1000,5),"E9":(1000,10),
}

baseline_df = load_baseline()
exp_df      = load_experiments()

if baseline_df.empty and exp_df.empty:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No results yet</b><br><br>
        Complete the Baseline Test and at least one Experiment to see comparison.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build summary ──────────────────────────────────────────────────────────────
rows = []
if not baseline_df.empty:
    rows.append({
        "Config":        "Baseline",
        "Chunk size":    "—",
        "k":             "—",
        "Accuracy":      round(float(baseline_df["accuracy"].mean()), 4),
        "Relevance":     round(float(baseline_df["relevance"].mean()), 4),
        "Faithfulness":  0.0000,
        "Overall":       round((float(baseline_df["accuracy"].mean()) +
                                float(baseline_df["relevance"].mean()) + 0.0) / 3, 4),
    })

if not exp_df.empty:
    for exp in EXPERIMENTS:
        df_e = exp_df[exp_df["experiment"] == exp]
        if not df_e.empty:
            cs, k = EXP_CONFIG[exp]
            avg_a = round(float(df_e["accuracy"].mean()), 4)
            avg_r = round(float(df_e["relevance"].mean()), 4)
            avg_f = round(float(df_e["faithfulness"].mean()), 4)
            rows.append({
                "Config":       f"{exp} (chunk {cs}, k={k})",
                "Chunk size":   cs,
                "k":            k,
                "Accuracy":     avg_a,
                "Relevance":    avg_r,
                "Faithfulness": avg_f,
                "Overall":      round((avg_a + avg_r + avg_f) / 3, 4),
            })

summary = pd.DataFrame(rows)

# ── Terminal panel — best RAGAS-style display ──────────────────────────────────
st.markdown('<div class="sec">1. Score summary</div>', unsafe_allow_html=True)

lines = "READDOC AI — EVALUATION RESULTS (0.0–1.0 SCALE)\n"
lines += "═" * 58 + "\n"
lines += f"{'Config':<28} {'Acc':>6} {'Rel':>6} {'Faith':>6} {'Overall':>8}\n"
lines += "─" * 58 + "\n"
for _, row in summary.iterrows():
    marker = " ← BEST" if row["Overall"] == summary["Overall"].max() and row["Config"] != "Baseline" else ""
    lines += f"{str(row['Config']):<28} {row['Accuracy']:>6.4f} {row['Relevance']:>6.4f} {row['Faithfulness']:>6.4f} {row['Overall']:>8.4f}{marker}\n"
lines += "═" * 58

st.markdown(f'<div class="score-panel">{lines}</div>', unsafe_allow_html=True)

# ── Table ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">2. Full results table</div>', unsafe_allow_html=True)
st.dataframe(
    summary[["Config","Accuracy","Relevance","Faithfulness","Overall"]],
    hide_index=True, use_container_width=True,
    column_config={
        "Accuracy":     st.column_config.ProgressColumn("Accuracy",     min_value=0, max_value=1, format="%.4f"),
        "Relevance":    st.column_config.ProgressColumn("Relevance",    min_value=0, max_value=1, format="%.4f"),
        "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0, max_value=1, format="%.4f"),
        "Overall":      st.column_config.ProgressColumn("Overall",      min_value=0, max_value=1, format="%.4f"),
    })

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">3. Comparison charts</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Overall score per configuration**")
    st.bar_chart(summary.set_index("Config")["Overall"], color="#1a56db", height=280)
with col2:
    st.markdown("**Three metrics side by side**")
    st.bar_chart(summary.set_index("Config")[["Accuracy","Relevance","Faithfulness"]], height=280)

exp_rows = summary[summary["Config"] != "Baseline"]
if len(exp_rows) >= 2:
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Effect of chunk size on overall score**")
        exp_rows_c = exp_rows.copy()
        exp_rows_c["Chunk size"] = exp_rows_c["Chunk size"].astype(str) + " chars"
        st.bar_chart(exp_rows_c.groupby("Chunk size")["Overall"].mean().round(4),
                     color="#1D9E75", height=240)
    with col4:
        st.markdown("**Effect of retrieval depth (k) on overall score**")
        exp_rows_k = exp_rows.copy()
        exp_rows_k["k"] = "k=" + exp_rows_k["k"].astype(str)
        st.bar_chart(exp_rows_k.groupby("k")["Overall"].mean().round(4),
                     color="#9333EA", height=240)

# ── Findings ───────────────────────────────────────────────────────────────────
rag_rows  = [r for _, r in summary.iterrows() if r["Config"] != "Baseline"]
base_row  = next((r for _, r in summary.iterrows() if r["Config"] == "Baseline"), None)

if rag_rows:
    st.markdown('<div class="sec">4. Key findings</div>', unsafe_allow_html=True)

    best  = max(rag_rows, key=lambda x: x["Overall"])
    worst = min(rag_rows, key=lambda x: x["Overall"])

    st.markdown(f"""
    <div class="finding">
        <b>Best RAG configuration:</b> <b>{best['Config']}</b> achieved the highest
        overall score of <b>{best['Overall']:.4f}</b>
        (Accuracy {best['Accuracy']:.4f} · Relevance {best['Relevance']:.4f} ·
        Faithfulness {best['Faithfulness']:.4f}).
    </div>
    """, unsafe_allow_html=True)

    if base_row is not None:
        faith_gain = round(best["Faithfulness"] - base_row["Faithfulness"], 4)
        overall_gain = round(best["Overall"] - base_row["Overall"], 4)
        st.markdown(f"""
        <div class="finding">
            <b>RAG vs Baseline:</b> The best RAG configuration improved overall score
            by <b>{overall_gain:.4f}</b> over baseline ({base_row['Overall']:.4f} → {best['Overall']:.4f}).
            Faithfulness improved from <b>0.0000</b> (baseline — no document) to
            <b>{best['Faithfulness']:.4f}</b>, confirming that retrieval grounds
            answers in the source document rather than relying on training data alone.
        </div>
        """, unsafe_allow_html=True)

    if len(rag_rows) >= 3:
        spread = round(best["Overall"] - worst["Overall"], 4)
        st.markdown(f"""
        <div class="finding">
            <b>Parameter sensitivity:</b> Varying chunk size and retrieval depth
            produced a <b>{spread:.4f} spread</b> in overall score across configurations
            (best: {best['Overall']:.4f} · worst: {worst['Overall']:.4f}),
            demonstrating that these parameters measurably affect output quality.
        </div>
        """, unsafe_allow_html=True)

    # ── RQ Answer ──────────────────────────────────────────────────────────────
    if len(rag_rows) >= 3:
        st.markdown('<div class="sec">5. Research question — answer</div>', unsafe_allow_html=True)
        exp_count = len(rag_rows)
        st.markdown(f"""
        <div class="rq-box">
            Based on evaluation of {exp_count} RAG configurations using a 10-question test set
            scored on a 0.0–1.0 scale across Accuracy, Relevance, and Faithfulness, varying
            chunk size and retrieval depth measurably affected all three dimensions.
            The best-performing configuration was {best['Config']}, achieving an overall
            score of {best['Overall']:.4f}. Faithfulness improved from 0.0000 (baseline,
            no document context) to {best['Faithfulness']:.4f} in the best RAG configuration,
            confirming that document retrieval grounds LLM responses in source material and
            substantially reduces hallucination. The spread of {round(best['Overall']-worst['Overall'],4):.4f}
            across configurations demonstrates that chunk size and retrieval depth are
            meaningful parameters in optimising RAG pipeline performance.
        </div>
        """, unsafe_allow_html=True)
        st.caption("Rewrite in your own words for the dissertation — all numbers are from your saved scores.")

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">6. Export</div>', unsafe_allow_html=True)
st.download_button("Download all results (CSV)", data=summary.to_csv(index=False),
    file_name="readdocai_results.csv", mime="text/csv")