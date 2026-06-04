"""
llm.py — Language Model Integration
=====================================
ReadDoc AI | MSc Data Science and Analytics

Uses Groq API (free) with Llama 3.3 for answer generation.
Groq uses the OpenAI-compatible chat.completions API.

Two modes:
  ask_llama_streaming() — RAG mode (with document context)
  ask_baseline()         — Baseline mode (no document context)
"""

import os
import streamlit as st
from groq import Groq

# ── Client setup ───────────────────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to .streamlit/secrets.toml or Streamlit Cloud secrets."
    )
    st.stop()

client = Groq(api_key=api_key)
MODEL  = "llama-3.3-70b-versatile"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ReadDoc AI — a smart, accurate document assistant.

A user has uploaded one or more documents. Answer questions using ONLY the provided context.

For every document question, respond in this exact format:

**Answer:**
[Clear, structured answer using bullet points where appropriate. Bold key terms and numbers.]

**Source:**
[Document name and page number(s)]

Rules:
- NEVER make up information. Only use what is in the provided context.
- If the answer is not in the context say: "I could not find this in your uploaded documents."
- Works with ANY document type — academic, legal, financial, technical, or general.

For greetings and general chat — respond naturally and introduce yourself as ReadDoc AI.
"""

# ── Build context message ──────────────────────────────────────────────────────
def build_user_message(question: str, retrieved_chunks: list) -> str:
    """Combine question + retrieved chunks into a single prompt."""
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


# ── RAG streaming (main chat) ──────────────────────────────────────────────────
def ask_llama_streaming(question: str, retrieved_chunks: list, history: list):
    """
    Stream answer tokens one by one for real-time display.
    Sends full conversation history so the model remembers previous turns.
    """
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
        if delta is not None:
            yield delta


# ── RAG non-streaming (experiment runner) ─────────────────────────────────────
def ask_llama(question: str, retrieved_chunks: list, history: list) -> str:
    """Returns full answer as a string — used in the experiment runner."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    response = client.chat.completions.create(
        model=MODEL, max_tokens=1024, messages=messages,
    )
    return response.choices[0].message.content


# ── Baseline — no document context ────────────────────────────────────────────
def ask_baseline(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question using only your general knowledge. Be direct and concise.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content