"""
llm.py — Language Model Integration
=====================================
PRIMARY: Ollama (fully local, zero API dependency).
  - Runs on your own machine at http://localhost:11434
  - Zero rate limits, zero quota, zero cost, works offline
  - Uses the same OpenAI-compatible API so function signatures unchanged
  - Model: llama3.2:3b (change OLLAMA_MODEL below if you pulled a different one)

CLOUD FALLBACK: Google Gemini (for Streamlit Cloud deployment only).
  - Ollama is a local server — Streamlit Cloud cannot reach localhost:11434
  - When the app detects it cannot connect to Ollama, it falls back to Gemini
  - Set GOOGLE_API_KEY in Streamlit Cloud Settings → Secrets for this to work
  - Only affects the deployed demo; local evaluation always uses Ollama

HOW IT DECIDES:
  - Tries to connect to Ollama first (quick 2-second timeout)
  - If Ollama responds → uses it for everything (local mode)
  - If Ollama is not reachable → uses Gemini (cloud/demo mode)
  - A small banner in the app tells you which provider is active

Same public functions as before (unchanged signatures):
  ask_llama_streaming()  — RAG mode, streams tokens
  ask_llama()            — RAG mode, non-streaming
  ask_baseline()         — no document context
"""

import os
import streamlit as st

# ── Model names ────────────────────────────────────────────────────────────────
# Change OLLAMA_MODEL to match whichever model you pulled:
#   ollama pull llama3.2:1b   → OLLAMA_MODEL = "llama3.2:1b"
#   ollama pull llama3.2:3b   → OLLAMA_MODEL = "llama3.2:3b"  (recommended)
#   ollama pull gemma2:2b     → OLLAMA_MODEL = "gemma2:2b"
OLLAMA_MODEL  = "llama3.2:3b"
OLLAMA_URL    = "http://localhost:11434/v1"
GEMINI_MODEL  = "gemini-1.5-flash-001"


# ── Detect which provider is available ────────────────────────────────────────
@st.cache_resource
def _detect_provider():
    """Try to reach Ollama once at startup. Cache the result so we only
    do the network probe once per session, not on every single call."""
    try:
        from openai import OpenAI
        probe = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
        probe.models.list()           # fast probe — just checks connectivity
        return "ollama"
    except Exception:
        return "gemini"


def _provider():
    return _detect_provider()


# ── Ollama client (lazy — only created if Ollama is reachable) ─────────────────
@st.cache_resource
def _ollama_client():
    from openai import OpenAI
    return OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ── Gemini client (lazy — only created if needed) ─────────────────────────────
@st.cache_resource
def _gemini_configured():
    key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return False
    import google.generativeai as genai
    genai.configure(api_key=key)
    return True


def _gemini_model(system_prompt: str):
    import google.generativeai as genai
    return genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)


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

BASELINE_PROMPT = ("You are a helpful assistant. Answer the question using only "
                   "your general knowledge. Be direct and concise.")


# ── Build context message ──────────────────────────────────────────────────────
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


# ── Internal helpers ───────────────────────────────────────────────────────────
def _call_ollama(messages: list, max_tokens: int, stream: bool):
    client = _ollama_client()
    return client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        stream=stream,
    )


def _messages_to_gemini(messages: list):
    system = None
    turns  = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            turns.append({"role": "user",  "parts": [m["content"]]})
        elif m["role"] == "assistant":
            turns.append({"role": "model", "parts": [m["content"]]})
    return system, turns


def _call_gemini(messages: list, max_tokens: int, stream: bool):
    import google.generativeai as genai
    system, turns = _messages_to_gemini(messages)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    *history, last = turns
    chat = model.start_chat(history=history)
    return chat.send_message(
        last["parts"][0],
        generation_config={"max_output_tokens": max_tokens},
        stream=stream,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def ask_llama_streaming(question: str, retrieved_chunks: list, history: list):
    """Stream answer tokens — RAG mode with document context."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    if _provider() == "ollama":
        stream = _call_ollama(messages, max_tokens=1024, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    else:
        if not _gemini_configured():
            yield "⚠️ No LLM available. Add GOOGLE_API_KEY to Streamlit secrets for the deployed app."
            return
        response = _call_gemini(messages, max_tokens=1024, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


def ask_llama(question: str, retrieved_chunks: list, history: list) -> str:
    """Return full answer as string — RAG mode, non-streaming."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    if _provider() == "ollama":
        resp = _call_ollama(messages, max_tokens=1024, stream=False)
        return resp.choices[0].message.content
    else:
        if not _gemini_configured():
            return "⚠️ No LLM available. Add GOOGLE_API_KEY to Streamlit secrets for the deployed app."
        resp = _call_gemini(messages, max_tokens=1024, stream=False)
        return resp.text


def ask_baseline(question: str) -> str:
    """Return answer with no document context — baseline evaluation mode."""
    messages = [
        {"role": "system", "content": BASELINE_PROMPT},
        {"role": "user",   "content": question},
    ]
    if _provider() == "ollama":
        resp = _call_ollama(messages, max_tokens=512, stream=False)
        return resp.choices[0].message.content
    else:
        if not _gemini_configured():
            return "⚠️ No LLM available. Add GOOGLE_API_KEY to Streamlit secrets for the deployed app."
        resp = _call_gemini(messages, max_tokens=512, stream=False)
        return resp.text