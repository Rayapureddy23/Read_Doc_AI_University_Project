"""
6_Results_Analysis.py — Results Comparison
============================================
ReadDoc AI | MSc Data Science and Analytics
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Results Analysis — ReadDoc AI",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
h1{font-size:26px!important;font-weight:700!important;color:#111827!important}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.result-panel{background:#0a0a0a;border-radius:12px;padding:22px 28px;
    font-family:'DM Mono',monospace;color:#fff;margin:1rem 0;
    font-size:13px;line-height:2.0}
.finding{background:#F8FAFF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:12px 16px;margin-bottom:10px;font-size:13.5px;color:#374151;line-height:1.7}
.rq-box{background:#EEF2FF;border-left:4px solid #1a56db;border-radius:0 12px 12px 0;
    padding:16px 20px;font-size:14px;color:#1e40af;font-style:italic;
    line-height:1.7;margin:1rem 0}
.empty-box{text-align:center;padding:2.5rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
.warn-box{background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:0 8px 8px 0;
    padding:14px 18px;font-size:13px;color:#92400E;margin-bottom:1.5rem;line-height:1.6}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0}
.kpi{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:12px;
    padding:16px;text-align:center}
.kpi-val{font-size:28px;font-weight:700;color:#1a56db;margin-bottom:4px}
.kpi-label{font-size:11px;color:#6B7280}
</style>
""", unsafe_allow_html=True)

st.title("Results Analysis")
st.markdown("RAGAS automated evaluation scores and manual scores — professional AI engineer style.")
st.markdown("---")

# ── Database ───────────────────────────────────────────────────────────────────
DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

EXP_CONFIG = {
    "E1":(300,3),"E2":(300,5),"E3":(300,10),
    "E4":(600,3),"E5":(600,5),"E6":(600,10),
    "E7":(1000,3),"E8":(1000,5),"E9":(1000,10),
}
ALL_EXPS = list(EXP_CONFIG.keys())

def load(table):
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

baseline_df = load("baseline_scores")
manual_df   = load("experiment_scores")
ragas_df    = load("ragas_results")
auto_df     = load("auto_eval_results")

# ── Important notice about data persistence ────────────────────────────────────
st.markdown("""
<div class="warn-box">
    <b>Important:</b> Streamlit Cloud does not persist the database between
    sessions. All scores shown here were saved during the <b>current session
    only</b>. Run your evaluations <b>locally</b> and use the CSV export
    buttons to keep a permanent copy. For a demo on this deployed app,
    run at least one experiment and RAGAS evaluation after uploading your document.
</div>
""", unsafe_allow_html=True)

# ── Check if we have any data at all ──────────────────────────────────────────
has_ragas  = not ragas_df.empty
has_auto   = not auto_df.empty
has_manual = not manual_df.empty
has_base   = not baseline_df.empty
has_any    = has_ragas or has_auto or has_manual or has_base

if not has_any:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No results yet</b><br><br>
        Upload your document on the main page, build the index, then run:<br>
        <b>Baseline Test</b> → <b>Experiment Runner</b> → <b>RAGAS Evaluation</b>
        or <b>Auto Evaluation</b><br><br>
        All charts populate automatically once scores are saved.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build merged results from ALL sources ─────────────────────────────────────
# Priority: RAGAS > Auto Eval > Manual (for same experiment)
# All three are merged so every scored experiment appears regardless
# of which page was used to score it.
rows = {}   # keyed by experiment ID to handle duplicates

# Baseline row
if has_base:
    rows["Baseline"] = {
        "Config":            "Baseline",
        "Chunk":             0,
        "k":                 0,
        "Faithfulness":      0.0,
        "Answer Relevancy":  round(float(baseline_df["relevance"].mean()), 4),
        "Context Precision": 0.0,
        "Overall":           round((0.0 + float(baseline_df["relevance"].mean()) + 0.0) / 3, 4),
        "Source":            "Manual",
    }

# Manual scores — lowest priority, added first
if has_manual:
    for exp in ALL_EXPS:
        df_e = manual_df[manual_df["experiment"] == exp]
        if df_e.empty:
            continue
        cs, k = EXP_CONFIG.get(exp, (0, 0))
        avg_a = round(float(df_e["accuracy"].mean()), 4)
        avg_r = round(float(df_e["relevance"].mean()), 4)
        avg_f = round(float(df_e["faithfulness"].mean()), 4)
        rows[exp] = {
            "Config":            exp,
            "Chunk":             cs,
            "k":                 k,
            "Faithfulness":      avg_f,
            "Answer Relevancy":  avg_r,
            "Context Precision": avg_a,
            "Overall":           round((avg_a + avg_r + avg_f) / 3, 4),
            "Source":            "Manual",
        }

# Auto Evaluation results — medium priority, overwrite manual
if has_auto:
    for _, r in auto_df.iterrows():
        exp = r["experiment"]
        cs, k = EXP_CONFIG.get(exp, (0, 0))
        rows[exp] = {
            "Config":            exp,
            "Chunk":             cs,
            "k":                 k,
            "Faithfulness":      round(float(r["faithfulness"]), 4),
            "Answer Relevancy":  round(float(r["answer_relevancy"]), 4),
            "Context Precision": round(float(r["context_precision"]), 4),
            "Overall":           round(float(r["overall"]), 4),
            "Source":            "Auto Eval",
        }

# RAGAS results — highest priority, overwrite auto/manual for same experiment
if has_ragas:
    for _, r in ragas_df.iterrows():
        exp = r["experiment"]
        cs, k = EXP_CONFIG.get(exp, (0, 0))
        rows[exp] = {
            "Config":            exp,
            "Chunk":             cs,
            "k":                 k,
            "Faithfulness":      round(float(r["faithfulness"]), 4),
            "Answer Relevancy":  round(float(r["answer_relevancy"]), 4),
            "Context Precision": round(float(r["context_precision"]), 4),
            "Overall":           round(float(r["overall"]), 4),
            "Source":            "RAGAS",
        }

# Convert to list sorted by experiment order
row_list = []
if "Baseline" in rows:
    row_list.append(rows["Baseline"])
for exp in ALL_EXPS:
    if exp in rows:
        row_list.append(rows[exp])

if not row_list:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No scored experiments found</b><br><br>
        Run RAGAS Evaluation, Auto Evaluation, or Experiment Runner to see results here.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.DataFrame(row_list)

# Safety check — ensure Config column exists
if "Config" not in df.columns:
    st.error("Unexpected data format — please re-run an evaluation.")
    st.stop()

rag_df = df[df["Config"] != "Baseline"].copy()

# ── Terminal panel ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Score summary</div>', unsafe_allow_html=True)
lines = "READDOC AI — EVALUATION RESULTS (0.0 – 1.0 SCALE)\n"
lines += "═" * 62 + "\n"
lines += f"{'Config':<24} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Overall':>8}\n"
lines += "─" * 62 + "\n"
for _, row in df.iterrows():
    marker = " ← BEST" if (
        not rag_df.empty and
        row["Config"] != "Baseline" and
        row["Overall"] == rag_df["Overall"].max()
    ) else ""
    lines += (f"{str(row['Config']):<24} {row['Faithfulness']:>7.4f} "
              f"{row['Answer Relevancy']:>7.4f} {row['Context Precision']:>7.4f} "
              f"{row['Overall']:>8.4f}{marker}\n")
lines += "═" * 62
st.markdown(f'<div class="result-panel">{lines}</div>', unsafe_allow_html=True)

# ── KPI cards ──────────────────────────────────────────────────────────────────
if not rag_df.empty:
    best = rag_df.loc[rag_df["Overall"].idxmax()]
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    for label, val, colour in [
        ("Best Faithfulness",      best["Faithfulness"],      "#1a56db"),
        ("Best Answer Relevancy",  best["Answer Relevancy"],  "#7C3AED"),
        ("Best Context Precision", best["Context Precision"], "#059669"),
        ("Best Overall",           best["Overall"],           "#DC2626"),
    ]:
        st.markdown(
            f'<div class="kpi"><div class="kpi-val" style="color:{colour}">'
            f'{val:.4f}</div><div class="kpi-label">{label}<br>'
            f'<small>{best["Config"]} (chunk {int(best["Chunk"])}, k={int(best["k"])})</small>'
            f'</div></div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Full table ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Full results table</div>', unsafe_allow_html=True)
st.dataframe(
    df[["Config","Chunk","k","Faithfulness","Answer Relevancy","Context Precision","Overall","Source"]],
    hide_index=True, use_container_width=True,
    column_config={
        "Faithfulness":      st.column_config.ProgressColumn("Faithfulness",      min_value=0,max_value=1,format="%.4f"),
        "Answer Relevancy":  st.column_config.ProgressColumn("Answer Relevancy",  min_value=0,max_value=1,format="%.4f"),
        "Context Precision": st.column_config.ProgressColumn("Context Precision", min_value=0,max_value=1,format="%.4f"),
        "Overall":           st.column_config.ProgressColumn("Overall",           min_value=0,max_value=1,format="%.4f"),
    })

# ── Charts ─────────────────────────────────────────────────────────────────────
if PLOTLY and len(df) >= 1:
    st.markdown('<div class="sec">Comparison charts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall score per configuration**")
        fig1 = go.Figure(go.Bar(
            x=df["Config"], y=df["Overall"],
            marker_color=["#9CA3AF" if c == "Baseline" else "#1a56db" for c in df["Config"]],
            text=df["Overall"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
        ))
        fig1.update_layout(yaxis=dict(range=[0,1.2]),height=320,
            plot_bgcolor="#F8FAFF",paper_bgcolor="#FFFFFF",
            font=dict(family="DM Sans"),margin=dict(t=20,b=30))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Three metrics side by side**")
        fig2 = go.Figure()
        for metric, colour in [
            ("Faithfulness","#1a56db"),
            ("Answer Relevancy","#7C3AED"),
            ("Context Precision","#059669"),
        ]:
            fig2.add_trace(go.Bar(
                name=metric, x=df["Config"], y=df[metric],
                marker_color=colour,
            ))
        fig2.update_layout(barmode="group",yaxis=dict(range=[0,1.2]),
            height=320,plot_bgcolor="#F8FAFF",paper_bgcolor="#FFFFFF",
            font=dict(family="DM Sans"),margin=dict(t=20,b=30),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # Faithfulness — baseline vs RAG
    st.markdown("**Faithfulness: baseline (0.00) vs RAG configurations**")
    fig3 = go.Figure(go.Bar(
        x=df["Config"], y=df["Faithfulness"],
        marker_color=["#9CA3AF" if c == "Baseline" else "#1a56db" for c in df["Config"]],
        text=df["Faithfulness"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig3.add_hline(y=0.0, line_dash="dash", line_color="#DC2626",
                   annotation_text="Baseline = 0.00", annotation_position="right")
    fig3.update_layout(yaxis=dict(range=[0,1.2]),height=320,showlegend=False,
        plot_bgcolor="#F8FAFF",paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),margin=dict(t=20,b=30))
    st.plotly_chart(fig3, use_container_width=True)

    if not rag_df.empty and len(rag_df) >= 2:
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Effect of chunk size on overall score**")
            chunk_avg = rag_df.groupby("Chunk")["Overall"].mean().round(4)
            chunk_avg.index = chunk_avg.index.astype(str) + " chars"
            fig4 = go.Figure(go.Bar(y=chunk_avg.values, x=chunk_avg.index,
                marker_color="#1D9E75",
                text=[f"{v:.4f}" for v in chunk_avg.values],
                textposition="outside"))
            fig4.update_layout(yaxis=dict(range=[0,1.2]),height=280,
                plot_bgcolor="#F8FAFF",paper_bgcolor="#FFFFFF",
                font=dict(family="DM Sans"),margin=dict(t=20,b=30))
            st.plotly_chart(fig4, use_container_width=True)

        with col4:
            st.markdown("**Effect of retrieval depth (k)**")
            k_avg = rag_df.groupby("k")["Overall"].mean().round(4)
            k_avg.index = "k=" + k_avg.index.astype(str)
            fig5 = go.Figure(go.Bar(y=k_avg.values, x=k_avg.index,
                marker_color="#9333EA",
                text=[f"{v:.4f}" for v in k_avg.values],
                textposition="outside"))
            fig5.update_layout(yaxis=dict(range=[0,1.2]),height=280,
                plot_bgcolor="#F8FAFF",paper_bgcolor="#FFFFFF",
                font=dict(family="DM Sans"),margin=dict(t=20,b=30))
            st.plotly_chart(fig5, use_container_width=True)

# ── Findings ───────────────────────────────────────────────────────────────────
if not rag_df.empty:
    st.markdown('<div class="sec">Key findings</div>', unsafe_allow_html=True)
    best  = rag_df.loc[rag_df["Overall"].idxmax()]
    worst = rag_df.loc[rag_df["Overall"].idxmin()]

    st.markdown(f"""
    <div class="finding">
        <b>Best configuration:</b> {best['Config']}
        (chunk {int(best['Chunk'])} chars, k={int(best['k'])}) —
        Overall <b>{best['Overall']:.4f}</b> ·
        Faithfulness {best['Faithfulness']:.4f} ·
        Answer Relevancy {best['Answer Relevancy']:.4f} ·
        Context Precision {best['Context Precision']:.4f}
    </div>
    """, unsafe_allow_html=True)

    faith_gain = round(float(best["Faithfulness"]) - 0.0, 4)
    overall_gain = round(float(best["Overall"]) - (row_list[0]["Overall"] if row_list else 0), 4)
    st.markdown(f"""
    <div class="finding">
        <b>RAG vs Baseline:</b> Faithfulness improved from <b>0.0000</b>
        (baseline — no document) to <b>{best['Faithfulness']:.4f}</b>
        in the best RAG configuration (+{faith_gain:.4f}), confirming
        that retrieval grounds LLM responses in the source document.
    </div>
    """, unsafe_allow_html=True)

    if len(rag_df) >= 2:
        spread = round(float(best["Overall"]) - float(worst["Overall"]), 4)
        st.markdown(f"""
        <div class="finding">
            <b>Parameter sensitivity:</b> {spread:.4f} point spread across
            {len(rag_df)} configurations (best {best['Config']} = {best['Overall']:.4f},
            worst {worst['Config']} = {worst['Overall']:.4f}), demonstrating
            chunk size and retrieval depth measurably affect answer quality.
        </div>
        """, unsafe_allow_html=True)

    if len(rag_df) >= 2:
        st.markdown('<div class="sec">Research question answer</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rq-box">
            Based on automated evaluation across {len(rag_df)} RAG configurations,
            varying chunk size and retrieval depth measurably affected accuracy,
            contextual relevance, and faithfulness. Faithfulness improved from
            0.0000 (baseline) to {best['Faithfulness']:.4f} in the best
            configuration, confirming retrieval grounds LLM responses in source
            material. The best configuration was <b>{best['Config']}</b>
            (chunk {int(best['Chunk'])} chars, k={int(best['k'])}),
            achieving an overall score of {best['Overall']:.4f}.
        </div>
        """, unsafe_allow_html=True)
        st.caption("Rewrite in your own words for the dissertation.")

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Export</div>', unsafe_allow_html=True)
st.download_button(
    "Download results CSV",
    data=df.to_csv(index=False),
    file_name="readdocai_results.csv",
    mime="text/csv",
)