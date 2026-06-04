"""
rag.py — Retrieval-Augmented Generation Pipeline
=================================================
ReadDoc AI | MSc Data Science and Analytics

This module handles the complete document processing pipeline:
  1. Document ingestion  — PDF and HTML text extraction
  2. Text chunking       — splitting text into overlapping segments
  3. Embedding           — converting text to semantic vectors
  4. FAISS indexing      — storing and searching vectors
  5. Retrieval           — finding relevant chunks for a query

Research variable: chunk_size (300 / 600 / 1000 chars)
Each chunk size is cached separately so switching is instant
after the first build.
"""

import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dimensional sentence embeddings
OVERLAP         = 100                   # character overlap between chunks

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load embedding model once at startup ──────────────────────────────────────
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready.")

# ── In-memory state ───────────────────────────────────────────────────────────
chunk_data: list = []
index            = None


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKING
# Split a long text into overlapping character-based segments.
# Overlap ensures sentences are not cut mid-sentence between chunks.
# ──────────────────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int = 600, overlap: int = OVERLAP) -> list:
    """
    Split text into overlapping chunks of chunk_size characters.

    Example (chunk_size=10, overlap=3):
      Text : "ABCDEFGHIJKLMNOP"
      Chunk1: ABCDEFGHIJ  (0 → 10)
      Chunk2: HIJKLMNOPQ  (7 → 17)   ← overlaps by 3 chars
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# DOCUMENT INGESTION — PDF
# ──────────────────────────────────────────────────────────────────────────────
def load_pdf(file_path: str) -> list:
    """Extract text from every page of a PDF file."""
    documents = []
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "file_name":   os.path.basename(file_path),
                    "page_number": page_num + 1,
                    "text":        text,
                    "source_type": "pdf",
                })
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return documents


# ──────────────────────────────────────────────────────────────────────────────
# DOCUMENT INGESTION — HTML
# ──────────────────────────────────────────────────────────────────────────────
def load_html(file_path: str) -> list:
    """Extract visible text from an HTML file using BeautifulSoup."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return [{
            "file_name":   os.path.basename(file_path),
            "page_number": "Web page",
            "text":        text,
            "source_type": "html",
        }]
    except Exception as e:
        print(f"Error reading HTML {file_path}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# BUILD INDEX
# Main function — ingest → chunk → embed → FAISS index → save to disk
# Each chunk_size gets its own cached index file.
# ──────────────────────────────────────────────────────────────────────────────
def build_index(file_paths: list, chunk_size: int = 600) -> dict:
    """
    Build or load a FAISS index for the given chunk size.

    Returns:
        dict with total_files, total_chunks, cached (bool)
    """
    global chunk_data, index

    index_path  = os.path.join(DATA_DIR, f"faiss_{chunk_size}.index")
    chunks_path = os.path.join(DATA_DIR, f"chunks_{chunk_size}.pkl")

    # ── Return cached index instantly if available ─────────────────────────
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunk_data = pickle.load(f)
            print(f"Loaded cached index (chunk {chunk_size}): {len(chunk_data)} chunks")
            return {
                "total_files":  len({c["file_name"] for c in chunk_data}),
                "total_chunks": len(chunk_data),
                "cached":       True,
            }
        except Exception as e:
            print(f"Cache load failed, rebuilding: {e}")

    # ── Step 1: Ingest documents ───────────────────────────────────────────
    all_documents = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            all_documents.extend(load_pdf(path))
        elif ext in [".html", ".htm"]:
            all_documents.extend(load_html(path))

    if not all_documents:
        return {"total_files": 0, "total_chunks": 0, "cached": False}

    # ── Step 2: Chunk every page ───────────────────────────────────────────
    chunk_data = []
    for doc in all_documents:
        for chunk_text in split_text(doc["text"], chunk_size=chunk_size):
            if chunk_text.strip():
                chunk_data.append({
                    "file_name":   doc["file_name"],
                    "page_number": doc["page_number"],
                    "text":        chunk_text,
                    "source_type": doc["source_type"],
                })

    # ── Step 3: Embed all chunks ───────────────────────────────────────────
    texts = [c["text"] for c in chunk_data]
    print(f"Embedding {len(texts)} chunks (chunk size = {chunk_size})...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=64)

    # ── Step 4: Build FAISS index ──────────────────────────────────────────
    dimension = embeddings.shape[1]           # 384 for all-MiniLM-L6-v2
    index     = faiss.IndexFlatL2(dimension)  # exact L2 distance search
    index.add(np.array(embeddings).astype("float32"))

    # ── Step 5: Save to disk ───────────────────────────────────────────────
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunk_data, f)

    print(f"Index built and saved: {len(chunk_data)} chunks")
    return {
        "total_files":  len(file_paths),
        "total_chunks": len(chunk_data),
        "cached":       False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LOAD INDEX FROM DISK (called at app startup)
# ──────────────────────────────────────────────────────────────────────────────
def load_index_from_disk(chunk_size: int = 600) -> bool:
    global chunk_data, index
    index_path  = os.path.join(DATA_DIR, f"faiss_{chunk_size}.index")
    chunks_path = os.path.join(DATA_DIR, f"chunks_{chunk_size}.pkl")
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunk_data = pickle.load(f)
            print(f"Loaded index (chunk {chunk_size}): {len(chunk_data)} chunks.")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH — retrieve top-k most relevant chunks for a question
# Research variable: top_k (3 / 5 / 10)
# ──────────────────────────────────────────────────────────────────────────────
def search(question: str, top_k: int = 5) -> list:
    """
    Encode the question, search FAISS for the top_k nearest chunks.
    Returns list of chunk dicts with distance field added.
    Lower distance = more semantically similar.
    FAISS returns -1 for missing results — these are filtered out.
    """
    if index is None or len(chunk_data) == 0:
        return []

    q_vector   = embedding_model.encode([question])
    distances, indices = index.search(
        np.array(q_vector).astype("float32"), top_k
    )

    results = []
    seen    = set()
    for dist, i in zip(distances[0], indices[0]):
        if i < 0 or i >= len(chunk_data):   # FAISS returns -1 for no match
            continue
        item = chunk_data[i]
        key  = (item["file_name"], item["page_number"])
        if key not in seen:
            seen.add(key)
            results.append({**item, "distance": round(float(dist), 3)})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# STATUS — for the UI
# ──────────────────────────────────────────────────────────────────────────────
def get_status() -> dict:
    return {
        "loaded":       index is not None,
        "total_chunks": len(chunk_data),
        "files":        list({c["file_name"] for c in chunk_data}),
    }