"""
llm.py — Language Model Integration
=====================================
  All 9 experiments fit in one day with headroom to spare.
"""

import os
import streamlit as st
from groq import Groq

# ── Client ─────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to .streamlit/secrets.toml or Streamlit Cloud secrets."
    )
    st.stop()

client = Groq(api_key=api_key)
MODEL  = "llama-3.1-8b-instant"


# ── Context builder ────────────────────────────────────────────────────────────
def build_user_message(question: str, retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return question
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"\n--- Chunk {i+1} "
            f"[{chunk['file_name']} | Page {chunk['page_number']}] ---\n"
            f"{chunk['text']}\n"
        )
    return f"Context from uploaded documents:\n{context}\n\n---\n\nQuestion: {question}"

# ── Streaming (main chat) ──────────────────────────────────────────────────────
def ask_llama_streaming(question: str, retrieved_chunks: list, history: list):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    stream = client.chat.completions.create(
        model=MODEL, max_tokens=1024, messages=messages, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ── Non-streaming (Local Metrics answer generation) ───────────────────────────
def ask_llama(question: str, retrieved_chunks: list, history: list) -> str:
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    response = client.chat.completions.create(
        model=MODEL, max_tokens=1024, messages=messages,
    )
    return response.choices[0].message.content

# ── Baseline (no document context) ────────────────────────────────────────────
def ask_baseline(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer using only your general knowledge. Be direct and concise."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content