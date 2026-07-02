"""
3_EDA.py — Exploratory Data Analysis
======================================
"""

import streamlit as st
import sys, os, pickle, re
import pandas as pd
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="EDA — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem;
    padding:2.5rem 3rem!important;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.page-badge{display:inline-block;background:#EEF2FF;color:#1a56db;font-size:12px;
    font-weight:600;padding:4px 12px;border-radius:20px;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:10px}
.page-title{font-size:24px;font-weight:700;color:#111827;margin-bottom:6px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:1.5rem;line-height:1.6}
.sec{font-size:17px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:1.5rem}
.stat-card{background:#F8FAFF;border-radius:10px;padding:16px;text-align:center;border:.5px solid #E5E9F5}
.stat-val{font-size:26px;font-weight:700;color:#1a56db}
.stat-label{font-size:12px;color:#6B7280;margin-top:3px}
.insight{background:#EEF2FF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0 1rem;line-height:1.6}
.empty-box{text-align:center;padding:3rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
.conclusion{background:#1a56db;border-radius:12px;padding:24px 28px;color:#fff;margin-top:1.5rem}
.mini-card{flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;
    padding:12px 16px}
.mini-label{font-size:11px;color:#6B7280;margin-bottom:4px}
.mini-val{font-size:16px;font-weight:600;color:#1a56db}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Exploratory Data Analysis</div>
<div class="page-title">Document and corpus analysis</div>
<div class="page-sub">
    EDA of the uploaded document corpus — chunk distribution, word frequency,
    text density, and chunk size comparison. These visualisations characterise
    the data before experimentation, supporting the methodology chapter.
</div>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

def find_chunk_files():
    """Find all chunks_*.pkl files regardless of naming scheme (old or new)."""
    if not os.path.isdir(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR)
            if f.startswith("chunks_") and f.endswith(".pkl")]

@st.cache_data
def load_chunks_by_label(filename, _mtime):
    """_mtime is included only so the cache busts if the file is rebuilt."""
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def build_readable_label(filename):
    """
    Cache files are named chunks_{chunk_size}_{hash}.pkl — not readable
    on their own. This builds a label like:
    "600 chars · DSML.pdf · 2,377 chunks · Jun 19, 21:45"
    """
    cache_key  = filename.replace("chunks_", "").replace(".pkl", "")
    chunk_size = cache_key.split("_", 1)[0]
    mtime      = os.path.getmtime(os.path.join(DATA_DIR, filename))
    mtime_str  = datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M")

    data = load_chunks_by_label(filename, mtime)
    if not data:
        return f"{chunk_size} chars · (could not load)"

    doc_name = data[0].get("file_name", "unknown file")
    return f"{chunk_size} chars · {doc_name} · {len(data):,} chunks · {mtime_str}"

chunk_files = find_chunk_files()

if not chunk_files:
    st.markdown("""
    <div class="empty-box">
        <b style="color:#374151;font-size:15px">No document indexed yet</b><br><br>
        Upload a document on the main ReadDoc AI page and click Build index.<br>
        Then come back here to see the full EDA.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build readable labels for the selector (most recent first)
chunk_files_sorted = sorted(
    chunk_files,
    key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)),
    reverse=True
)
labels = [build_readable_label(f) for f in chunk_files_sorted]

sel_label  = st.selectbox("Select an indexed document to analyse", labels, index=0)
sel_file   = chunk_files_sorted[labels.index(sel_label)]
sel_mtime  = os.path.getmtime(os.path.join(DATA_DIR, sel_file))
chunk_data = load_chunks_by_label(sel_file, sel_mtime)

if not chunk_data:
    st.warning("Could not load this index. Try rebuilding it on the main page.")
    st.stop()

df = pd.DataFrame(chunk_data)
df["text_length"]   = df["text"].str.len()
df["word_count"]    = df["text"].str.split().str.len()
df["page_number_n"] = pd.to_numeric(df["page_number"], errors="coerce")
files = df["file_name"].unique()

# ── Section 1: Document statistics ────────────────────────────────────────────
st.markdown('<div class="sec">1. Document statistics</div>', unsafe_allow_html=True)
total_chunks  = len(df)
total_words   = int(df["word_count"].sum())
total_chars   = int(df["text_length"].sum())
unique_pages  = int(df["page_number_n"].nunique())
avg_len       = int(df["text_length"].mean())

st.markdown(f"""<div class="stat-grid">
  <div class="stat-card"><div class="stat-val">{total_chunks:,}</div><div class="stat-label">Total chunks</div></div>
  <div class="stat-card"><div class="stat-val">{unique_pages:,}</div><div class="stat-label">Pages indexed</div></div>
  <div class="stat-card"><div class="stat-val">{total_words:,}</div><div class="stat-label">Total words</div></div>
  <div class="stat-card"><div class="stat-val">{total_chars:,}</div><div class="stat-label">Total characters</div></div>
  <div class="stat-card"><div class="stat-val">{avg_len}</div><div class="stat-label">Avg chunk (chars)</div></div>
  <div class="stat-card"><div class="stat-val">{len(files)}</div><div class="stat-label">File(s)</div></div>
</div>""", unsafe_allow_html=True)

st.markdown(f'<div class="insight">Document: <b>{files[0]}</b> — currently indexed with <b>{total_chunks:,} chunks</b> across <b>{unique_pages} pages</b>.</div>', unsafe_allow_html=True)

# ── Section 2: Chunk length distribution ──────────────────────────────────────
st.markdown('<div class="sec">2. Chunk length distribution</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Character length per chunk**")
    hist = pd.cut(
        df["text_length"],
        bins=[0,100,200,300,400,500,600,700,800,900,1000,1200],
        labels=["0-100","100-200","200-300","300-400","400-500","500-600",
                "600-700","700-800","800-900","900-1000","1000+"]
    ).value_counts().sort_index()
    st.bar_chart(hist, color="#1a56db", height=260)
with c2:
    st.markdown("**Word count per chunk**")
    wh = pd.cut(
        df["word_count"],
        bins=[0,20,40,60,80,100,120,140,160,180,200],
        labels=["0-20","20-40","40-60","60-80","80-100","100-120","120-140","140-160","160-180","180+"]
    ).value_counts().sort_index()
    st.bar_chart(wh, color="#1D9E75", height=260)

st.markdown(f"""
<div style="display:flex;gap:12px;margin:.5rem 0 1rem">
    <div class="mini-card"><div class="mini-label">Min length</div><div class="mini-val">{int(df['text_length'].min())} chars</div></div>
    <div class="mini-card"><div class="mini-label">Max length</div><div class="mini-val">{int(df['text_length'].max())} chars</div></div>
    <div class="mini-card"><div class="mini-label">Mean length</div><div class="mini-val">{int(df['text_length'].mean())} chars</div></div>
    <div class="mini-card"><div class="mini-label">Std deviation</div><div class="mini-val">{int(df['text_length'].std())} chars</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="insight">Chunk length distribution shows most chunks cluster around <b>{int(df["text_length"].mode()[0])} characters</b>. A tight distribution (low std dev) means consistent retrieval units across all experiment runs.</div>', unsafe_allow_html=True)

# ── Section 3: Chunks per page ─────────────────────────────────────────────────
st.markdown('<div class="sec">3. Chunks per page</div>', unsafe_allow_html=True)
page_df = df[df["page_number_n"].notna()]
if not page_df.empty:
    cpp = page_df.groupby("page_number_n").size().reset_index(name="chunk_count").sort_values("page_number_n")
    st.markdown("**Number of chunks extracted from each page of the document**")
    st.area_chart(cpp.set_index("page_number_n")["chunk_count"], color="#1a56db", height=240)

    dense  = cpp.loc[cpp["chunk_count"].idxmax()]
    sparse = cpp.loc[cpp["chunk_count"].idxmin()]
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:.5rem 0 1rem">
        <div class="mini-card"><div class="mini-label">Avg chunks/page</div><div class="mini-val">{round(cpp['chunk_count'].mean(),1)}</div></div>
        <div class="mini-card"><div class="mini-label">Densest page</div><div class="mini-val">Page {int(dense['page_number_n'])} ({int(dense['chunk_count'])} chunks)</div></div>
        <div class="mini-card"><div class="mini-label">Sparsest page</div><div class="mini-val">Page {int(sparse['page_number_n'])} ({int(sparse['chunk_count'])} chunk)</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="insight">Pages with more chunks are text-dense chapters (theory, algorithms, equations). Sparse pages are figures or section openers. This explains why some questions retrieve from a narrow page range.</div>', unsafe_allow_html=True)
else:
    st.info("Page numbers not available for this document type (e.g. HTML source).")

# ── Section 4: Word frequency ──────────────────────────────────────────────────
st.markdown('<div class="sec">4. Most frequent domain words</div>', unsafe_allow_html=True)
STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were",
        "be","been","have","has","had","do","does","did","will","would","could","should","may","this",
        "that","these","those","it","its","by","from","as","into","we","our","i","you","they","their",
        "not","also","which","such","if","so","when","then","than","more","all","about","each","other",
        "some","there","where","while","up","out","no","let","thus","hence","therefore","however",
        "given","using","used","since","both","based","per","two","one","three","figure","see","table",
        "chapter","section","example","note","equation","above","below","where","here","new","use"}
all_text = " ".join(df["text"]).lower()
words    = re.findall(r'\b[a-z]{4,}\b', all_text)
freq     = Counter(w for w in words if w not in STOP).most_common(30)
top5     = [w for w,_ in freq[:5]] if freq else []

if freq:
    fw = pd.DataFrame(freq, columns=["Word","Count"]).set_index("Word")
    w1, w2 = st.columns([2,1])
    with w1:
        st.markdown("**Top 30 most frequent substantive terms**")
        st.bar_chart(fw, color="#7F77DD", height=320)
    with w2:
        st.markdown("**Top 15 ranked**")
        st.dataframe(pd.DataFrame(freq[:15], columns=["Word","Count"]), hide_index=True, use_container_width=True)
    st.markdown(f'<div class="insight">Most frequent terms: <b>{", ".join(top5)}</b>. These confirm the document domain and show the vocabulary the embedding model will use most strongly in semantic search queries.</div>', unsafe_allow_html=True)

# ── Section 5: Text density ───────────────────────────────────────────────────
if not page_df.empty:
    st.markdown('<div class="sec">5. Text density across the document</div>', unsafe_allow_html=True)
    tpp = page_df.groupby("page_number_n")["text_length"].sum().reset_index().rename(columns={"text_length":"chars"}).sort_values("page_number_n")
    st.markdown("**Total characters extracted per page — shows where content is concentrated**")
    st.area_chart(tpp.set_index("page_number_n")["chars"], color="#1D9E75", height=220)
    st.markdown('<div class="insight">Peaks = text-heavy theory chapters. Troughs = figure pages or chapter openers with limited extractable text. Affects retrieval accuracy — factual questions about dense sections score higher.</div>', unsafe_allow_html=True)

# ── Section 6: Chunk size comparison ──────────────────────────────────────────
st.markdown('<div class="sec">6. Chunk size comparison — all indexed configurations</div>', unsafe_allow_html=True)
comp = []
for fname in chunk_files_sorted:
    fmtime = os.path.getmtime(os.path.join(DATA_DIR, fname))
    cd = load_chunks_by_label(fname, fmtime)
    if cd:
        cache_key  = fname.replace("chunks_", "").replace(".pkl", "")
        chunk_size_label = cache_key.split("_", 1)[0]
        doc_name   = cd[0].get("file_name", "unknown")
        comp.append({
            "Configuration":   f"{chunk_size_label} chars ({doc_name})",
            "Total chunks":    len(cd),
            "Avg words/chunk": round(sum(len(c["text"].split()) for c in cd)/len(cd),1),
            "Avg chars/chunk": round(sum(len(c["text"]) for c in cd)/len(cd),1),
        })

if comp:
    cdf = pd.DataFrame(comp)
    st.dataframe(cdf, hide_index=True, use_container_width=True)
    st.markdown("**Total chunks created per indexed configuration**")
    st.bar_chart(cdf.set_index("Configuration")["Total chunks"], color="#1a56db", height=200)
    st.markdown('<div class="insight">Smaller chunk sizes create more chunks with less context per chunk. Larger sizes create fewer chunks with more surrounding text. This tradeoff is the first research variable.</div>', unsafe_allow_html=True)

# ── Section 7: Dissertation summary ───────────────────────────────────────────
st.markdown('<div class="sec">7. EDA summary — copy into dissertation</div>', unsafe_allow_html=True)
std     = int(df["text_length"].std())
density = "relatively uniform" if std < 150 else "moderately variable"
densest_text = ""
if not page_df.empty:
    densest_text = f" The densest page was page {int(dense['page_number_n'])} ({int(dense['chunk_count'])} chunks), consistent with algorithm-heavy content."

st.markdown(f"""
<div class="conclusion">
  <div style="font-size:16px;font-weight:600;margin-bottom:12px">Key EDA findings — paste into Chapter 3 or 4</div>
  <p style="font-size:13px;opacity:.95;margin-bottom:8px">
    The document corpus consisted of <b>{total_chunks:,} chunks</b> extracted from <b>{unique_pages:,} pages</b>
    of {files[0]}. The total corpus contained approximately <b>{total_words:,} words</b> across <b>{total_chars:,} characters</b>.
  </p>
  <p style="font-size:13px;opacity:.95;margin-bottom:8px">
    The mean chunk length was <b>{avg_len} characters</b> (std = {std} chars), indicating {density}
    chunk sizes.{densest_text}
  </p>
  <p style="font-size:13px;opacity:.95;margin:0">
    {"Word frequency analysis confirmed the document domain — the most frequent substantive terms were <b>" + ", ".join(top5) + "</b>. These high-frequency domain terms are expected to anchor semantic search queries effectively, supporting the retrieval quality observed in the RAG experiment results." if top5 else ""}
  </p>
</div>
""", unsafe_allow_html=True)