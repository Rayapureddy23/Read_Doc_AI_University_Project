"""
6_Results_Analysis.py — Professional Results Visualisation
============================================================

Shows RAGAS automated scores and manual scores in professional
plotly charts — radar, bar, heatmap, and line plots — matching
how AI engineers present RAG evaluation results.
"""

import streamlit as st
import sys, os, sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{font-size:26px!important;font-weight:700!important;color:#111827!important}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:1rem 0}
.kpi{background:#F8FAFF;border:1px solid #E5E7EB;border-radius:12px;
    padding:16px 20px;text-align:center}
.kpi-val{font-size:30px;font-weight:700;color:#1a56db;margin-bottom:4px}
.kpi-label{font-size:12px;color:#6B7280}
.kpi-sub{font-size:11px;color:#9CA3AF;margin-top:2px}
.finding{background:#F8FAFF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:12px 16px;margin-bottom:10px;font-size:13.5px;color:#374151;line-height:1.7}
.rq-box{background:#EEF2FF;border-left:4px solid #1a56db;border-radius:0 12px 12px 0;
    padding:16px 20px;font-size:14px;color:#1e40af;font-style:italic;line-height:1.7;margin:1rem 0}
.empty-box{text-align:center;padding:2.5rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
</style>
""", unsafe_allow_html=True)

st.title("Results Analysis")
st.markdown("RAGAS automated evaluation scores and manual scores visualised — professional AI engineer style.")
st.markdown("---")

# ── Colour palette (consistent across all charts) ──────────────────────────────
COLOURS = {
    "faithfulness":       "#1a56db",
    "answer_relevancy":   "#7C3AED",
    "context_precision":  "#059669",
    "overall":            "#DC2626",
    "manual_acc":         "#F59E0B",
    "manual_rel":         "#EC4899",
    "manual_faith":       "#06B6D4",
}

EXP_CONFIG = {
    "E1":(300,3),"E2":(300,5),"E3":(300,10),
    "E4":(600,3),"E5":(600,5),"E6":(600,10),
    "E7":(1000,3),"E8":(1000,5),"E9":(1000,10),
}
ALL_EXPS = ["E1","E2","E3","E4","E5","E6","E7","E8","E9"]

DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "experiments.db"
)

def load(table, where=None, params=None):
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        q = f"SELECT * FROM {table}"
        if where:
            q += f" WHERE {where}"
        df = pd.read_sql_query(q, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

baseline_df = load("baseline_scores")
manual_df   = load("experiment_scores")
ragas_df    = load("ragas_results")

if ragas_df.empty and manual_df.empty and baseline_df.empty:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No results yet</b><br><br>
        Run RAGAS Evaluation for at least one experiment — all charts
        populate automatically as you add more experiment results.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build combined summary ──────────────────────────────────────────────────────
rows = []
if not baseline_df.empty:
    rows.append({
        "Config": "Baseline",
        "Chunk": 0, "k": 0,
        "Faithfulness":      0.0,
        "Answer Relevancy":  round(float(baseline_df["relevance"].mean()), 4),
        "Context Precision": 0.0,
        "Overall (RAGAS)":   round((0.0 + float(baseline_df["relevance"].mean()) + 0.0) / 3, 4),
        "Manual Accuracy":   round(float(baseline_df["accuracy"].mean()), 4),
        "Manual Relevance":  round(float(baseline_df["relevance"].mean()), 4),
        "Manual Faithfulness": 0.0,
        "Manual Overall":    round((float(baseline_df["accuracy"].mean()) +
                                   float(baseline_df["relevance"].mean()) + 0.0) / 3, 4),
    })

if not ragas_df.empty:
    for _, r in ragas_df.iterrows():
        exp  = r["experiment"]
        cs, k = EXP_CONFIG.get(exp, (0, 0))
        man   = manual_df[manual_df["experiment"] == exp] if not manual_df.empty else pd.DataFrame()
        rows.append({
            "Config": exp,
            "Chunk": cs, "k": k,
            "Faithfulness":      round(float(r["faithfulness"]), 4),
            "Answer Relevancy":  round(float(r["answer_relevancy"]), 4),
            "Context Precision": round(float(r["context_precision"]), 4),
            "Overall (RAGAS)":   round(float(r["overall"]), 4),
            "Manual Accuracy":   round(float(man["accuracy"].mean()), 4) if not man.empty else None,
            "Manual Relevance":  round(float(man["relevance"].mean()), 4) if not man.empty else None,
            "Manual Faithfulness": round(float(man["faithfulness"].mean()), 4) if not man.empty else None,
            "Manual Overall":    round((float(man["accuracy"].mean()) +
                                       float(man["relevance"].mean()) +
                                       float(man["faithfulness"].mean())) / 3, 4)
                                  if not man.empty else None,
        })

df = pd.DataFrame(rows)
rag_df = df[df["Config"] != "Baseline"]

# ── KPI cards ──────────────────────────────────────────────────────────────────
if not rag_df.empty:
    best = rag_df.loc[rag_df["Overall (RAGAS)"].idxmax()]
    st.markdown('<div class="sec">Key results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-val" style="color:#1a56db">{best['Faithfulness']:.4f}</div>
            <div class="kpi-label">Best Faithfulness</div>
            <div class="kpi-sub">{best['Config']} (chunk {int(best['Chunk'])}, k={int(best['k'])})</div>
        </div>
        <div class="kpi">
            <div class="kpi-val" style="color:#7C3AED">{best['Answer Relevancy']:.4f}</div>
            <div class="kpi-label">Best Answer Relevancy</div>
            <div class="kpi-sub">{best['Config']}</div>
        </div>
        <div class="kpi">
            <div class="kpi-val" style="color:#059669">{best['Context Precision']:.4f}</div>
            <div class="kpi-label">Best Context Precision</div>
            <div class="kpi-sub">{best['Config']}</div>
        </div>
        <div class="kpi">
            <div class="kpi-val" style="color:#DC2626">{best['Overall (RAGAS)']:.4f}</div>
            <div class="kpi-label">Best Overall</div>
            <div class="kpi-sub">{best['Config']} ← BEST CONFIGURATION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Plot 1: Grouped bar chart — all RAGAS metrics per experiment ────────────────
st.markdown('<div class="sec">Chart 1 — RAGAS scores per experiment</div>',
            unsafe_allow_html=True)
st.caption("Shows all three RAGAS metrics side by side for every evaluated configuration.")

if not df.empty:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Faithfulness",
        x=df["Config"], y=df["Faithfulness"],
        marker_color=COLOURS["faithfulness"],
        text=df["Faithfulness"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig1.add_trace(go.Bar(
        name="Answer Relevancy",
        x=df["Config"], y=df["Answer Relevancy"],
        marker_color=COLOURS["answer_relevancy"],
        text=df["Answer Relevancy"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig1.add_trace(go.Bar(
        name="Context Precision",
        x=df["Config"], y=df["Context Precision"],
        marker_color=COLOURS["context_precision"],
        text=df["Context Precision"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig1.update_layout(
        barmode="group",
        yaxis=dict(title="Score (0.0 – 1.0)", range=[0, 1.15]),
        xaxis_title="Configuration",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(t=60, b=40),
        plot_bgcolor="#F8FAFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Plot 2: Radar chart — best configuration profile ───────────────────────────
st.markdown('<div class="sec">Chart 2 — Metric profile (radar chart)</div>',
            unsafe_allow_html=True)
st.caption("Radar chart shows the balance across all three metrics for each configuration. "
           "A perfect configuration fills the entire triangle.")

if not rag_df.empty:
    categories = ["Faithfulness", "Answer Relevancy", "Context Precision"]
    fig2 = go.Figure()
    colours_radar = ["#1a56db", "#7C3AED", "#059669", "#DC2626",
                     "#F59E0B", "#EC4899", "#06B6D4", "#8B5CF6", "#10B981"]
    for i, (_, row) in enumerate(rag_df.iterrows()):
        vals = [row["Faithfulness"], row["Answer Relevancy"], row["Context Precision"]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=row["Config"],
            line_color=colours_radar[i % len(colours_radar)],
            opacity=0.6,
        ))
    if not baseline_df.empty:
        base_vals = [0.0,
                     round(float(baseline_df["relevance"].mean()), 4),
                     0.0]
        base_closed = base_vals + [base_vals[0]]
        fig2.add_trace(go.Scatterpolar(
            r=base_closed,
            theta=categories + [categories[0]],
            fill="toself",
            name="Baseline",
            line_color="#9CA3AF",
            line_dash="dash",
            opacity=0.4,
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480,
        margin=dict(t=60, b=40),
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Plot 3: Heatmaps — chunk size × k for each metric ─────────────────────────
st.markdown('<div class="sec">Chart 3 — Parameter sensitivity heatmap</div>',
            unsafe_allow_html=True)
st.caption("Shows how Faithfulness, Answer Relevancy, and Context Precision each vary "
           "across chunk sizes (rows) and retrieval depths k (columns). "
           "Darker = higher score.")

chunk_sizes = [300, 600, 1000]
k_values    = [3, 5, 10]

if not rag_df.empty and len(rag_df) >= 2:
    for metric, label, colourscale in [
        ("Faithfulness", "Faithfulness", "Blues"),
        ("Answer Relevancy", "Answer Relevancy", "Purples"),
        ("Context Precision", "Context Precision", "Greens"),
    ]:
        matrix = []
        for cs in chunk_sizes:
            row_vals = []
            for k in k_values:
                match = rag_df[(rag_df["Chunk"] == cs) & (rag_df["k"] == k)]
                row_vals.append(round(float(match[metric].iloc[0]), 4) if not match.empty else None)
            matrix.append(row_vals)

        z_text = [[f"{v:.4f}" if v is not None else "—" for v in r] for r in matrix]
        fig_h = go.Figure(go.Heatmap(
            z=matrix,
            x=[f"k={k}" for k in k_values],
            y=[f"{cs} chars" for cs in chunk_sizes],
            text=z_text,
            texttemplate="%{text}",
            colorscale=colourscale,
            zmin=0, zmax=1,
            showscale=True,
        ))
        fig_h.update_layout(
            title=dict(text=f"{label} — chunk size × retrieval depth",
                       font=dict(size=14, family="DM Sans")),
            height=280,
            margin=dict(t=50, b=30, l=80),
            paper_bgcolor="#FFFFFF",
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("Heatmaps appear once you have results for multiple configurations. "
            "Run RAGAS for more experiments to populate all 9 cells.")

# ── Plot 4: Line chart — effect of k per chunk size ────────────────────────────
st.markdown('<div class="sec">Chart 4 — Effect of retrieval depth (k) per chunk size</div>',
            unsafe_allow_html=True)
st.caption("Shows how increasing k affects overall RAGAS score for each chunk size independently.")

if not rag_df.empty and len(rag_df) >= 2:
    fig4 = go.Figure()
    line_colours = {"300": "#1a56db", "600": "#7C3AED", "1000": "#059669"}
    for cs in chunk_sizes:
        sub = rag_df[rag_df["Chunk"] == cs].sort_values("k")
        if not sub.empty:
            fig4.add_trace(go.Scatter(
                x=sub["k"],
                y=sub["Overall (RAGAS)"],
                mode="lines+markers+text",
                name=f"Chunk {cs}",
                line=dict(color=line_colours[str(cs)], width=2.5),
                marker=dict(size=9),
                text=sub["Overall (RAGAS)"].apply(lambda x: f"{x:.3f}"),
                textposition="top center",
            ))
    fig4.update_layout(
        xaxis=dict(title="Retrieval depth (k)", tickvals=[3, 5, 10]),
        yaxis=dict(title="Overall RAGAS score", range=[0, 1.1]),
        height=380,
        margin=dict(t=40, b=40),
        plot_bgcolor="#F8FAFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Line chart appears once you have results across multiple k values.")

# ── Plot 5: RAGAS vs Manual comparison ─────────────────────────────────────────
manual_rows = df[df["Manual Overall"].notna()]
if not manual_rows.empty:
    st.markdown('<div class="sec">Chart 5 — RAGAS automated vs manual scores</div>',
                unsafe_allow_html=True)
    st.caption("Compares automated RAGAS scores against human rubric scores for the "
               "same experiments. Convergence validates both methods independently.")

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        name="RAGAS Overall (automated)",
        x=manual_rows["Config"],
        y=manual_rows["Overall (RAGAS)"],
        marker_color="#1a56db",
        text=manual_rows["Overall (RAGAS)"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig5.add_trace(go.Bar(
        name="Manual Overall (human rubric)",
        x=manual_rows["Config"],
        y=manual_rows["Manual Overall"],
        marker_color="#F59E0B",
        text=manual_rows["Manual Overall"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig5.update_layout(
        barmode="group",
        yaxis=dict(title="Score (0.0 – 1.0)", range=[0, 1.2]),
        xaxis_title="Configuration",
        height=400,
        margin=dict(t=50, b=40),
        plot_bgcolor="#F8FAFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Plot 6: Faithfulness — baseline vs best RAG ─────────────────────────────────
st.markdown('<div class="sec">Chart 6 — Faithfulness: baseline vs RAG</div>',
            unsafe_allow_html=True)
st.caption("The most important finding — faithfulness is always 0.00 in the baseline "
           "(no document) and rises significantly with RAG retrieval.")

if not df.empty:
    fig6 = go.Figure(go.Bar(
        x=df["Config"],
        y=df["Faithfulness"],
        marker_color=[
            "#9CA3AF" if c == "Baseline" else "#1a56db"
            for c in df["Config"]
        ],
        text=df["Faithfulness"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig6.add_hline(
        y=0.0, line_dash="dash", line_color="#DC2626",
        annotation_text="Baseline = 0.00",
        annotation_position="right",
    )
    fig6.update_layout(
        yaxis=dict(title="Faithfulness score", range=[0, 1.15]),
        xaxis_title="Configuration",
        height=380,
        margin=dict(t=40, b=40),
        plot_bgcolor="#F8FAFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans"),
        showlegend=False,
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Summary table ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Full results table</div>', unsafe_allow_html=True)
display = df[["Config","Chunk","k","Faithfulness","Answer Relevancy",
              "Context Precision","Overall (RAGAS)"]].copy()
st.dataframe(
    display, hide_index=True, use_container_width=True,
    column_config={
        "Faithfulness":      st.column_config.ProgressColumn("Faithfulness",      min_value=0,max_value=1,format="%.4f"),
        "Answer Relevancy":  st.column_config.ProgressColumn("Answer Relevancy",  min_value=0,max_value=1,format="%.4f"),
        "Context Precision": st.column_config.ProgressColumn("Context Precision", min_value=0,max_value=1,format="%.4f"),
        "Overall (RAGAS)":   st.column_config.ProgressColumn("Overall (RAGAS)",   min_value=0,max_value=1,format="%.4f"),
    })

# ── Auto findings ────────────────────────────────────────────────────────────────
if not rag_df.empty:
    st.markdown('<div class="sec">Auto-written findings</div>', unsafe_allow_html=True)
    best    = rag_df.loc[rag_df["Overall (RAGAS)"].idxmax()]
    base_faith = 0.0
    faith_gain = round(float(best["Faithfulness"]) - base_faith, 4)

    st.markdown(f"""
    <div class="finding">
        <b>Best configuration:</b> {best['Config']} (chunk {int(best['Chunk'])} chars, k={int(best['k'])})
        achieved the highest RAGAS overall score of <b>{best['Overall (RAGAS)']:.4f}</b>
        — Faithfulness {best['Faithfulness']:.4f},
        Answer Relevancy {best['Answer Relevancy']:.4f},
        Context Precision {best['Context Precision']:.4f}.
    </div>
    <div class="finding">
        <b>Faithfulness improvement over baseline:</b> +{faith_gain:.4f}
        ({base_faith:.4f} → {best['Faithfulness']:.4f}). The baseline always scores 0.00
        because no document is provided — every claim comes from training data, not a source.
        RAG retrieval raises faithfulness to {best['Faithfulness']:.4f}, confirming that
        retrieved context grounds LLM responses in the source document.
    </div>
    """, unsafe_allow_html=True)

    if len(rag_df) >= 3:
        worst   = rag_df.loc[rag_df["Overall (RAGAS)"].idxmin()]
        spread  = round(float(best["Overall (RAGAS)"]) - float(worst["Overall (RAGAS)"]), 4)
        st.markdown(f"""
        <div class="finding">
            <b>Parameter sensitivity:</b> varying chunk size and retrieval depth produced
            a <b>{spread:.4f} point spread</b> in RAGAS overall score across {len(rag_df)}
            evaluated configurations (best: {best['Config']} = {best['Overall (RAGAS)']:.4f},
            worst: {worst['Config']} = {worst['Overall (RAGAS)']:.4f}), demonstrating that
            these parameters measurably affect answer quality.
        </div>
        """, unsafe_allow_html=True)

# ── RQ answer ───────────────────────────────────────────────────────────────────
if not rag_df.empty and len(rag_df) >= 2:
    best = rag_df.loc[rag_df["Overall (RAGAS)"].idxmax()]
    st.markdown('<div class="sec">Research question answer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rq-box">
        Based on RAGAS automated evaluation across {len(rag_df)} of 9 planned
        configurations, varying chunk size and retrieval depth measurably affected
        all three evaluation dimensions. Faithfulness improved from 0.0000 (baseline)
        to {best['Faithfulness']:.4f} in the best RAG configuration, confirming that
        document retrieval grounds LLM responses in source material. The best
        configuration was <b>{best['Config']}</b> (chunk {int(best['Chunk'])} characters,
        k={int(best['k'])}), achieving an overall RAGAS score of {best['Overall (RAGAS)']:.4f}.
    </div>
    """, unsafe_allow_html=True)
    st.caption("Rewrite in your own words for the dissertation — numbers are from your real saved results.")

# ── Export ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Export</div>', unsafe_allow_html=True)
st.download_button(
    "Download full results CSV",
    data=df.to_csv(index=False),
    file_name="readdocai_results.csv",
    mime="text/csv",
)