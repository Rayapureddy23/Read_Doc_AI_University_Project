"""
2_How_It_Works.py — Pipeline Explained Step by Step
=====================================================

Shows the supervisor exactly how the code works:
  - Each pipeline stage with code snippets
  - Why each decision was made
  - How the models work
"""

import streamlit as st

st.set_page_config(page_title="How It Works — ReadDoc AI", page_icon="🔵", layout="wide")

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
.page-title{font-size:26px;font-weight:700;color:#111827;margin-bottom:6px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:2rem;line-height:1.6}
.sec{font-size:18px;font-weight:600;color:#111827;margin:2rem 0 .8rem;
    padding-bottom:8px;border-bottom:2px solid #EEF2FF}
.step-card{border:.5px solid #E5E9F5;border-radius:12px;overflow:hidden;margin-bottom:1.5rem}
.step-head{display:flex;align-items:center;gap:12px;padding:14px 18px;
    background:#F8FAFF;border-bottom:.5px solid #E5E9F5}
.step-num{width:32px;height:32px;border-radius:50%;background:#1a56db;color:#fff;
    display:flex;align-items:center;justify-content:center;font-size:13px;
    font-weight:600;flex-shrink:0}
.step-title{font-size:15px;font-weight:600;color:#111827}
.step-subtitle{font-size:12px;color:#6B7280;margin-top:2px}
.step-body{padding:16px 20px}
.why-box{background:#EEF2FF;border-left:3px solid #1a56db;border-radius:0 8px 8px 0;
    padding:10px 14px;font-size:13px;color:#1e40af;margin-top:12px;line-height:1.6}
.model-card{background:#F8FAFF;border:.5px solid #E5E9F5;border-radius:10px;
    padding:14px 18px;margin-bottom:10px}
.model-name{font-size:14px;font-weight:600;color:#111827;margin-bottom:4px}
.model-detail{font-size:13px;color:#374151;line-height:1.7}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-badge">Technical implementation</div>
<div class="page-title">How ReadDoc AI works — step by step</div>
<div class="page-sub">
    A detailed walkthrough of every component in the RAG pipeline —
    from uploading a PDF to generating a sourced answer.
    Each step includes the actual code used and the reasoning behind each design decision.
</div>
""", unsafe_allow_html=True)

# ── Pipeline overview ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">Pipeline overview</div>', unsafe_allow_html=True)
st.code("""
User uploads PDF / HTML
        ↓
[rag.py]  Text extraction    — PyPDF reads each page; BeautifulSoup parses HTML
        ↓
[rag.py]  Text chunking      — split into overlapping segments (300 / 600 / 1000 chars)
        ↓
[rag.py]  Sentence embedding — all-MiniLM-L6-v2 converts each chunk to a 384-dim vector
        ↓
[rag.py]  FAISS index        — IndexFlatL2 stores vectors and enables fast search
        ↓
User asks a question
        ↓
[rag.py]  Semantic search    — question → vector → find top-k nearest chunks (k = 3/5/10)
        ↓
[llm.py]  LLM generation     — Llama 3.3 reads retrieved chunks + answers from them only
        ↓
[app.py]  Streaming display  — answer streams word by word + source citations shown
""", language="text")

# ── Step 1: Text extraction ────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 1 — Text extraction</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">1</div>
    <div><div class="step-title">PDF and HTML ingestion</div>
    <div class="step-subtitle">rag.py — load_pdf() and load_html()</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
from pypdf import PdfReader
from bs4 import BeautifulSoup

def load_pdf(file_path):
    reader = PdfReader(file_path)
    pages  = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():          # skip blank pages
            pages.append({
                "file_name":   os.path.basename(file_path),
                "page_number": page_num + 1,  # 1-indexed for human readability
                "text":        text,
                "source_type": "pdf",
            })
    return pages

# Each page becomes a dict with file_name, page_number, text, source_type.
# Page numbers are preserved so the final answer can cite the exact page.
""", language="python")

st.markdown("""
    <div class="why-box">
      </b> PyPDF is a pure-Python library that extracts text reliably
      from PDFs including those with equations and tables. Each page is stored
      separately with its page number — this is critical for source citation in answers.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 2: Chunking ───────────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 2 — Text chunking (research variable 1)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">2</div>
    <div><div class="step-title">Splitting text into overlapping segments</div>
    <div class="step-subtitle">rag.py — split_text() | Variable: chunk_size = 300 / 600 / 1000</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
def split_text(text, chunk_size=600, overlap=100):
    \"\"\"
    Split text into overlapping chunks.

    overlap=100 means consecutive chunks share 100 characters.
    This prevents important sentences from being cut mid-sentence
    between chunks, which would cause retrieval to miss them.

    Example with chunk_size=20, overlap=5, text="ABCDEFGHIJKLMNOPQRSTU":
      Chunk 1: ABCDEFGHIJKLMNOPQRST  (start=0)
      Chunk 2: OPQRSTUVWXYZ...       (start=15)  ← 5-char overlap
    \"\"\"
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap       # step back by overlap
    return chunks

# THIS IS THE KEY RESEARCH VARIABLE.
# chunk_size=300 → more chunks, less context per chunk
# chunk_size=600 → balanced (hypothesis: optimal)
# chunk_size=1000 → fewer chunks, more context per chunk
""", language="python")

st.markdown("""
    <div class="why-box">
      </b>
      Chunk size controls how much text the model sees per retrieved passage.
      Too small — answers may be incomplete because the relevant sentence is split across chunks.
      Too large — irrelevant surrounding text is retrieved, reducing answer precision.
      This tradeoff is exactly what the experiment tests.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 3: Embedding ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 3 — Sentence embedding</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">3</div>
    <div><div class="step-title">Converting text to semantic vectors</div>
    <div class="step-subtitle">rag.py — SentenceTransformer("all-MiniLM-L6-v2")</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
from sentence_transformers import SentenceTransformer

# Load the model once at startup (not on every query)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all chunks — each chunk becomes a 384-dimensional vector
texts      = [chunk["text"] for chunk in chunk_data]
embeddings = embedding_model.encode(texts,
                                    show_progress_bar=True,
                                    batch_size=64)

# embeddings.shape = (num_chunks, 384)
# Each number in the 384-dim vector captures a semantic dimension of meaning.
# Similar meaning → similar vector → small distance in vector space.

# Example:
# "supervised learning uses labelled data"   → [0.12, -0.34, 0.89, ...]
# "labelled examples for training"           → [0.11, -0.31, 0.91, ...]
# "the weather in London today"              → [-0.78, 0.92, -0.12, ...]
# The first two will be close together; the third will be far away.
""", language="python")

st.markdown("""
    <div class="why-box">
      </b> This model (Reimers & Gurevych, 2019 — Sentence-BERT)
      produces high-quality 384-dimensional embeddings. It is small enough to run
      on a CPU without a GPU, making it practical for this research platform.
      It outperforms keyword-based methods (BM25, TF-IDF) on semantic similarity tasks
      because it understands meaning, not just word overlap.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 4: FAISS ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 4 — FAISS vector index</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">4</div>
    <div><div class="step-title">Storing and searching vectors at scale</div>
    <div class="step-subtitle">rag.py — faiss.IndexFlatL2</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
import faiss
import numpy as np

# Build the index
dimension = embeddings.shape[1]           # 384
index     = faiss.IndexFlatL2(dimension)  # exact L2 (Euclidean) distance
index.add(np.array(embeddings).astype("float32"))

# Save to disk — each chunk_size gets its own file
faiss.write_index(index, f"data/faiss_{chunk_size}.index")

# Later: search for the top-k nearest chunks
def search(question, top_k=5):
    q_vector = embedding_model.encode([question])
    distances, indices = index.search(
        np.array(q_vector).astype("float32"), top_k
    )
    # distances[0] = similarity scores (lower L2 = more similar)
    # indices[0]   = positions of matching chunks in chunk_data list
    return [chunk_data[i] for i in indices[0] if i >= 0]

# IndexFlatL2 = exact brute-force search
# Alternative: IndexIVFFlat (approximate, faster for millions of vectors)
# For this research scale (~2000-6000 chunks), exact search is appropriate.
""", language="python")

st.markdown("""
    <div class="why-box">
      </b> Developed by Facebook AI Research (Johnson et al., 2021),
      FAISS is the industry standard for vector similarity search.
      IndexFlatL2 performs exact L2 distance search — guaranteeing we always
      find the true nearest neighbours. This is important for the research
      because approximate search would introduce noise into the experiment results.
      Each chunk size configuration is saved as a separate index file so switching
      between 300/600/1000 during experiments is instant.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 5: LLM generation ─────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 5 — LLM answer generation</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">5</div>
    <div><div class="step-title">Generating answers from retrieved context</div>
    <div class="step-subtitle">llm.py — Llama 3.3 via Groq API</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
from groq import Groq
client = Groq(gsk_ZQQWsdQeN4j3YXIIO8E1WGdyb3FYcAY8w2SGV8eCW4vOn2uXzen)

def ask_llama_streaming(question, retrieved_chunks, history):
    # Step 1: Build context string from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"--- Chunk {i+1} "
            f"[{chunk['file_name']} | Page {chunk['page_number']}] ---\\n"
            f"{chunk['text']}\\n"
        )

    # Step 2: Build the prompt — context + question
    user_message = f\"\"\"Context from uploaded documents:
{context}
---
Question: {question}\"\"\"

    # Step 3: Send to LLM with full conversation history
    messages = (
        [{"role": "system",  "content": SYSTEM_PROMPT}]
        + history               # all previous Q&A pairs
        + [{"role": "user", "content": user_message}]
    )

    # Step 4: Stream response token by token
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=messages,
        stream=True,            # yields tokens as they are generated
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta         # caller displays each token as it arrives
""", language="python")

st.markdown("""
    <div class="why-box">
      </b> Llama 3.3-70b is a state-of-the-art open-source
      instruction-following model that follows the structured answer format reliably.
      Groq's inference hardware provides fast streaming with no cost on the free tier,
      making it practical for running hundreds of experiment queries.
      Critically, the system prompt instructs the model to answer ONLY from
      the provided context — this is what makes it a RAG system rather than
      a general LLM. Without this constraint, the model would answer from
      general training knowledge, which is exactly what the baseline experiment tests.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Step 6: Retrieval depth ────────────────────────────────────────────────────
st.markdown('<div class="sec">Step 6 — Retrieval depth (research variable 2)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-card">
  <div class="step-head">
    <div class="step-num">6</div>
    <div><div class="step-title">How many chunks to retrieve per question</div>
    <div class="step-subtitle">rag.py — search(question, top_k=5) | Variable: k = 3 / 5 / 10</div></div>
  </div>
  <div class="step-body">
""", unsafe_allow_html=True)

st.code("""
# top_k controls how many chunks are retrieved and passed to the LLM.

# k=3: retrieve 3 most similar chunks
#   Pro: focused, less noise
#   Con: may miss relevant content if it falls in chunk 4 or 5

# k=5: retrieve 5 most similar chunks (hypothesis: balanced)
#   Pro: good recall without too much noise
#   Con: may include some weakly related content

# k=10: retrieve 10 most similar chunks
#   Pro: highest recall — unlikely to miss relevant content
#   Con: weakly related chunks can confuse the model (dilution effect)

retrieved_chunks = search(question, top_k=5)

# The chunks are ordered by relevance (lowest L2 distance = most relevant).
# The LLM reads all k chunks and synthesises an answer from them.
# If the answer exists in any of the k chunks, the LLM should find it.
# If none of the k chunks contain the answer, the model should say so.

# This is the second key research variable:
# Does increasing k always improve answers, or is there a sweet spot?
""", language="python")

st.markdown("""
    <div class="why-box">
      </b>
      Retrieval depth is the second independent variable. The hypothesis is
      that there is an optimal k — too few misses answers, too many adds noise.
      The 9-configuration experiment (3 chunk sizes × 3 k values) tests
      all combinations to find which pairing performs best.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── How scoring works ──────────────────────────────────────────────────────────
st.markdown('<div class="sec">Evaluation — how answers are scored</div>', unsafe_allow_html=True)

rubric = [
    ("Accuracy", "Is the answer factually correct compared to the document?",
     "5=Fully correct · 4=Mostly correct · 3=Partial · 2=Mostly wrong · 1=Hallucinated"),
    ("Contextual relevance", "Does the answer actually address what was asked?",
     "5=Directly answers · 4=Mostly relevant · 3=Partially · 2=Mostly irrelevant · 1=Unrelated"),
    ("Faithfulness", "Is every claim grounded in the document — nothing invented?",
     "5=Fully grounded + source cited · 4=Mostly grounded · 3=Some unsupported claims · 2=Mostly ungrounded · 1=Hallucinated"),
]

for name, desc, scale in rubric:
    st.markdown(
        f'<div class="model-card"><div class="model-name">{name}</div>'
        f'<div class="model-detail">{desc}<br>'
        f'<span style="font-size:12px;color:#1a56db">{scale}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )