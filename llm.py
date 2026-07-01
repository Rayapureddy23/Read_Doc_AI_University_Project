"""
llm.py — Language Model Integration
=====================================

Uses Groq API with Llama 3.3 70B for answer generation.
RAGAS has been dropped from the evaluation pipeline — Local Metrics
covers all three RQ constructs (accuracy, contextual relevance,
faithfulness) using only the 20 answer-generation calls per experiment,
with zero additional judge calls.

Token budget across all 9 experiments:
  20 answers × 9 experiments × ~400 tokens = ~72,000 tokens total
  Groq free limit: 100,000 tokens/day
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