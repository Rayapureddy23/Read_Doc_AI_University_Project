"""
rag.py — Retrieval-Augmented Generation Pipeline
=================================================


FIXED: Cache is now keyed by BOTH chunk_size AND the uploaded file(s).
Previously the cache only checked chunk_size, so uploading a NEW file
while keeping the same chunk size setting would incorrectly load the
OLD file's cached index. Now a hash of the file names + sizes is
included in the cache key, so a new upload always rebuilds correctly.
"""

import os
import pickle
import hashlib
import numpy as np
import faiss
from pypdf import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
OVERLAP          = 100

os.makedirs(DATA_DIR, exist_ok=True)

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.")
    return embedding_model

# ── In-memory state ───────────────────────────────────────────────────────────
chunk_data: list = []
index            = None
current_cache_key = None   # tracks which file+chunk_size is currently loaded


# ──────────────────────────────────────────────────────────────────────────────
# CACHE KEY — based on file names + file sizes + chunk size
# This ensures a NEW upload always produces a NEW cache key,
# so the old cached index is never mistakenly reused.
# ──────────────────────────────────────────────────────────────────────────────
def make_cache_key(file_paths: list, chunk_size: int) -> str:
    """
    Build a unique cache key from the file names, their byte sizes,
    and the chunk size. Same files + same chunk size → same key
    (instant cache load). Different files → different key (rebuild).
    """
    parts = []
    for path in sorted(file_paths):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        parts.append(f"{os.path.basename(path)}:{size}")

    signature = "|".join(parts) + f"|chunk{chunk_size}"
    digest    = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
    return f"{chunk_size}_{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKING
# ──────────────────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int = 600, overlap: int = OVERLAP) -> list:
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
# Cache key now includes file identity, not just chunk_size.
# ──────────────────────────────────────────────────────────────────────────────
def build_index(file_paths: list, chunk_size: int = 600) -> dict:
    """
    Build or load a FAISS index for the given files + chunk size.

    The cache key is derived from the file names, file sizes, and
    chunk size together. This guarantees that uploading a different
    document always triggers a fresh build, even if the chunk size
    setting is unchanged from a previous session.
    """
    global chunk_data, index, current_cache_key

    cache_key   = make_cache_key(file_paths, chunk_size)
    index_path  = os.path.join(DATA_DIR, f"faiss_{cache_key}.index")
    chunks_path = os.path.join(DATA_DIR, f"chunks_{cache_key}.pkl")

    # ── Return cached index instantly if it matches THIS file + chunk size ──
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunk_data = pickle.load(f)
            current_cache_key = cache_key
            print(f"Loaded cached index [{cache_key}]: {len(chunk_data)} chunks")
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
    print(f"Embedding {len(texts)} chunks [{cache_key}]...")
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # ── Step 4: Build FAISS index ──────────────────────────────────────────
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # ── Step 5: Save to disk under the file-aware cache key ────────────────
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunk_data, f)

    current_cache_key = cache_key
    print(f"Index built and saved [{cache_key}]: {len(chunk_data)} chunks")
    return {
        "total_files":  len(file_paths),
        "total_chunks": len(chunk_data),
        "cached":       False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BUILD ALL CHUNK SIZES AT ONCE
# Pre-builds and caches every chunk size variant for the given file(s) in
# one pass. After this runs, switching the chunk-size slider only needs to
# call build_index() again — which will load instantly from cache instead
# of re-embedding the whole document.
# ──────────────────────────────────────────────────────────────────────────────
def build_all_sizes(file_paths: list, chunk_sizes=(300, 600, 1000),
                     activate: int = 600, progress_callback=None) -> dict:
    """
    Build (or load from cache) an index for every chunk size in chunk_sizes.

    Args:
        file_paths:        list of uploaded file paths
        chunk_sizes:        which chunk sizes to pre-build
        activate:           which chunk size should be the active one when done
        progress_callback:  optional fn(step, total, chunk_size, result) for UI progress

    Returns:
        dict mapping chunk_size -> build_index() result dict
    """
    results = {}
    total   = len(chunk_sizes)
    for step, size in enumerate(chunk_sizes, start=1):
        result = build_index(file_paths, chunk_size=size)
        results[size] = result
        if progress_callback:
            progress_callback(step, total, size, result)

    # Leave the global index/chunk_data pointed at the requested active size.
    # This call is instant — that size was just built/cached above.
    if activate in chunk_sizes:
        build_index(file_paths, chunk_size=activate)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LOAD INDEX FROM DISK (app startup — best effort, no file context yet)
# ──────────────────────────────────────────────────────────────────────────────
def load_index_from_disk(chunk_size: int = 600) -> bool:
    """
    At app startup we don't yet know which file was uploaded, so this
    only looks for ANY previously-built index matching the chunk size
    prefix. Once the user uploads a file and clicks Build index,
    build_index() takes over with the correct file-aware cache key.
    """
    global chunk_data, index, current_cache_key
    prefix = f"faiss_{chunk_size}_"
    if not os.path.isdir(DATA_DIR):
        return False

    matches = [f for f in os.listdir(DATA_DIR)
               if f.startswith(prefix) and f.endswith(".index")]
    if not matches:
        return False

    # Load the most recently modified matching index
    matches.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
    chosen      = matches[0]
    cache_key   = chosen[len("faiss_"):-len(".index")]
    index_path  = os.path.join(DATA_DIR, chosen)
    chunks_path = os.path.join(DATA_DIR, f"chunks_{cache_key}.pkl")

    if os.path.exists(chunks_path):
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunk_data = pickle.load(f)
            current_cache_key = cache_key
            print(f"Loaded most recent index [{cache_key}]: {len(chunk_data)} chunks.")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────────────
def search(question: str, top_k: int = 5) -> list:
    if index is None or len(chunk_data) == 0:
        return []

    model = get_embedding_model()
    q_vector = model.encode([question])
    distances, indices  = index.search(
        np.array(q_vector).astype("float32"), top_k
    )

    results = []
    seen    = set()
    for dist, i in zip(distances[0], indices[0]):
        if i < 0 or i >= len(chunk_data):
            continue
        item = chunk_data[i]
        key  = (item["file_name"], item["page_number"])
        if key not in seen:
            seen.add(key)
            results.append({**item, "distance": round(float(dist), 3)})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# STATUS
# ──────────────────────────────────────────────────────────────────────────────
def get_status() -> dict:
    return {
        "loaded":       index is not None,
        "total_chunks": len(chunk_data),
        "files":        list({c["file_name"] for c in chunk_data}),
        "cache_key":    current_cache_key,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLEAR CACHE — utility to wipe all old cached indexes
# Useful if old caches from previous documents are cluttering data/
# ──────────────────────────────────────────────────────────────────────────────
def clear_all_cached_indexes():
    """Delete every cached .index and .pkl file in data/. Use with caution."""
    global chunk_data, index, current_cache_key
    removed = 0
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("faiss_") or fname.startswith("chunks_"):
                try:
                    os.remove(os.path.join(DATA_DIR, fname))
                    removed += 1
                except OSError:
                    pass
    chunk_data         = []
    index              = None
    current_cache_key  = None
    return removed