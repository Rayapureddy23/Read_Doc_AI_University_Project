"""
3_EDA.py — Exploratory Data Analysis
======================================
ReadDoc AI
"""


import streamlit as st
import sys, os, pickle, re
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="EDA — ReadDoc AI", page_icon="🔵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
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
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:1.5rem}
.stat-card{background:#F8FAFF;border-radius:10px;padding:16px;text-align:center;border:.5px solid #E5E9F5}
.stat-val{font-size:26px;font-weight:700;color:#1a56db}
.stat-label{font-size:12px;color:#6B7280;margin-top:3px}
.insight{background:#EEF2FF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0 1rem;line-height:1.6}
.empty-box{text-align:center;padding:3rem;color:#9CA3AF;background:#F8FAFF;
    border-radius:12px;border:1px dashed #C7D3F5}
.conclusion{background:#1a56db;border-radius:12px;padding:24px 28px;color:#fff;margin-top:1.5rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Exploratory Data Analysis</div>
<div class="page-title">Document and corpus analysis</div>
<div class="page-sub">
    EDA of the DSML.pdf corpus — chunk distribution, word frequency,
    text density, and chunk size comparison. These visualisations characterise
    the data before experimentation, supporting the methodology chapter.
</div>
""", unsafe_allow_html=True)

def load_chunks(sz):
    p = os.path.join("data", f"chunks_{sz}.pkl")
    if os.path.exists(p):
        with open(p,"rb") as f: return pickle.load(f)
    return None

chunk_data, found_size = None, None
for sz in [600,300,1000]:
    chunk_data = load_chunks(sz)
    if chunk_data: found_size = sz; break

if not chunk_data:
    st.markdown('<div class="empty-box"><b style="color:#374151;font-size:15px">No document indexed yet</b><br><br>Upload DSML.pdf on the main ReadDoc AI page and click Build index.</div>', unsafe_allow_html=True)
    st.stop()

sel = st.selectbox("Chunk size to analyse", [300,600,1000], index=[300,600,1000].index(found_size))
if sel != found_size:
    cd = load_chunks(sel)
    if cd: chunk_data, found_size = cd, sel
    else: st.warning(f"No index for chunk {sel}. Build it first."); st.stop()

df = pd.DataFrame(chunk_data)
df["text_length"]   = df["text"].str.len()
df["word_count"]    = df["text"].str.split().str.len()
df["page_number_n"] = pd.to_numeric(df["page_number"], errors="coerce")
files = df["file_name"].unique()

# ── Statistics ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">1. Document statistics</div>', unsafe_allow_html=True)
total_chunks = len(df); total_words = int(df["word_count"].sum())
total_chars  = int(df["text_length"].sum()); unique_pages = int(df["page_number_n"].nunique())
avg_len      = int(df["text_length"].mean())

st.markdown(f"""<div class="stat-grid">
  <div class="stat-card"><div class="stat-val">{total_chunks:,}</div><div class="stat-label">Total chunks</div></div>
  <div class="stat-card"><div class="stat-val">{unique_pages:,}</div><div class="stat-label">Pages indexed</div></div>
  <div class="stat-card"><div class="stat-val">{total_words:,}</div><div class="stat-label">Total words</div></div>
  <div class="stat-card"><div class="stat-val">{total_chars:,}</div><div class="stat-label">Total characters</div></div>
  <div class="stat-card"><div class="stat-val">{avg_len}</div><div class="stat-label">Avg chunk (chars)</div></div>
  <div class="stat-card"><div class="stat-val">{found_size}</div><div class="stat-label">Chunk size</div></div>
</div>""", unsafe_allow_html=True)

# ── Chunk length distribution ──────────────────────────────────────────────────
st.markdown('<div class="sec">2. Chunk length distribution</div>', unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    st.markdown("**Character length per chunk**")
    hist = pd.cut(df["text_length"], bins=[0,100,200,300,400,500,600,700,800,900,1000,1200],
                  labels=["0-100","100-200","200-300","300-400","400-500","500-600",
                          "600-700","700-800","800-900","900-1000","1000+"]).value_counts().sort_index()
    st.bar_chart(hist, color="#1a56db", height=260)
with c2:
    st.markdown("**Word count per chunk**")
    wh = pd.cut(df["word_count"], bins=[0,20,40,60,80,100,120,140,160,180,200],
                labels=["0-20","20-40","40-60","60-80","80-100","100-120","120-140","140-160","160-180","180+"]).value_counts().sort_index()
    st.bar_chart(wh, color="#1D9E75", height=260)

st.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:1rem">
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Min length</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">{int(df['text_length'].min())} chars</div>
    </div>
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Max length</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">{int(df['text_length'].max())} chars</div>
    </div>
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Mean length</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">{int(df['text_length'].mean())} chars</div>
    </div>
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Std deviation</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">{int(df['text_length'].std())} chars</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="insight">Chunk length distribution shows most chunks cluster around <b>{int(df["text_length"].mode()[0])} characters</b>. A tight distribution (low std dev) means consistent retrieval units across all experiment runs.</div>', unsafe_allow_html=True)

# ── Chunks per page ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">3. Chunks per page</div>', unsafe_allow_html=True)
page_df = df[df["page_number_n"].notna()]
cpp     = page_df.groupby("page_number_n").size().reset_index(name="chunk_count").sort_values("page_number_n")
st.markdown("**Number of chunks extracted from each page of the document**")
st.area_chart(cpp.set_index("page_number_n")["chunk_count"], color="#1a56db", height=240)

dense  = cpp.loc[cpp["chunk_count"].idxmax()]
sparse = cpp.loc[cpp["chunk_count"].idxmin()]

st.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:1rem">
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Avg chunks/page</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">{round(cpp['chunk_count'].mean(),1)}</div>
    </div>
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Densest page</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">Page {int(dense['page_number_n'])} ({int(dense['chunk_count'])} chunks)</div>
    </div>
    <div style="flex:1;background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;padding:12px 16px">
        <div style="font-size:11px;color:#6B7280;margin-bottom:4px">Sparsest page</div>
        <div style="font-size:16px;font-weight:600;color:#1a56db">Page {int(sparse['page_number_n'])} ({int(sparse['chunk_count'])} chunk)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Word frequency ─────────────────────────────────────────────────────────────
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
top5     = [w for w,_ in freq[:5]]

fw = pd.DataFrame(freq, columns=["Word","Count"]).set_index("Word")
w1,w2 = st.columns([2,1])
with w1:
    st.markdown("**Top 30 most frequent substantive terms**")
    st.bar_chart(fw, color="#7F77DD", height=320)
with w2:
    st.markdown("**Top 15 ranked**")
    st.dataframe(pd.DataFrame(freq[:15], columns=["Word","Count"]), hide_index=True, use_container_width=True)

st.markdown(f'<div class="insight">Most frequent terms: <b>{", ".join(top5)}</b>. These confirm the document domain and show the vocabulary the embedding model will use most strongly in semantic search queries.</div>', unsafe_allow_html=True)

# ── Text density ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">5. Text density across the document</div>', unsafe_allow_html=True)
tpp = page_df.groupby("page_number_n")["text_length"].sum().reset_index().rename(columns={"text_length":"chars"}).sort_values("page_number_n")
st.markdown("**Total characters extracted per page — shows where content is concentrated**")
st.area_chart(tpp.set_index("page_number_n")["chars"], color="#1D9E75", height=220)
st.markdown('<div class="insight">Peaks = text-heavy theory chapters. Troughs = figure pages or chapter openers with limited extractable text. Affects retrieval accuracy — factual questions about dense sections score higher.</div>', unsafe_allow_html=True)

# ── Chunk size comparison ──────────────────────────────────────────────────────
st.markdown('<div class="sec">6. Chunk size comparison — all configurations</div>', unsafe_allow_html=True)
comp = []
for sz in [300,600,1000]:
    p = os.path.join("data",f"chunks_{sz}.pkl")
    if os.path.exists(p):
        with open(p,"rb") as f: cd = pickle.load(f)
        comp.append({"Chunk size":f"{sz} chars","Total chunks":len(cd),
                     "Avg words/chunk":round(sum(len(c["text"].split()) for c in cd)/len(cd),1),
                     "Avg chars/chunk":round(sum(len(c["text"]) for c in cd)/len(cd),1)})

if comp:
    cdf = pd.DataFrame(comp)
    st.dataframe(cdf, hide_index=True, use_container_width=True)
    st.markdown("**Total chunks created per chunk size setting**")
    st.bar_chart(cdf.set_index("Chunk size")["Total chunks"], color="#1a56db", height=200)
    st.markdown('<div class="insight">Smaller chunk sizes create more chunks with less context per chunk. Larger sizes create fewer chunks with more surrounding text. This tradeoff is the first research variable.</div>', unsafe_allow_html=True)
else:
    st.info("Build indexes for all three chunk sizes to see the full comparison.")

# ── Dissertation summary ───────────────────────────────────────────────────────
st.markdown('<div class="sec">7. EDA summary -</div>', unsafe_allow_html=True)
std = int(df["text_length"].std())
density = "relatively uniform" if std < 150 else "moderately variable"

st.markdown(f"""
<div class="conclusion">
  <div style="font-size:16px;font-weight:600;margin-bottom:12px">Key EDA findings </div>
  <p style="font-size:13px;opacity:.95;margin-bottom:8px">
    The document corpus consisted of <b>{total_chunks:,} chunks</b> extracted from <b>{unique_pages:,} pages</b>
    of {files[0]} using a chunk size of {found_size} characters with 100-character overlap.
    The total corpus contained approximately <b>{total_words:,} words</b> across <b>{total_chars:,} characters</b>.
  </p>
  <p style="font-size:13px;opacity:.95;margin-bottom:8px">
    The mean chunk length was <b>{avg_len} characters</b> (std = {std} chars), indicating {density}
    chunk sizes. The densest page was page {int(dense['page_number_n'])} ({int(dense['chunk_count'])} chunks),
    consistent with algorithm-heavy content in Kroese et al. (2020).
  </p>
  <p style="font-size:13px;opacity:.95;margin:0">
    Word frequency analysis confirmed the document domain — the most frequent substantive terms were
    <b>{", ".join(top5)}</b>, consistent with a Data Science and Machine Learning textbook.
    These high-frequency domain terms are expected to anchor semantic search queries effectively,
    supporting the retrieval quality observed in the RAG experiment results.
  </p>
</div>
""", unsafe_allow_html=True)