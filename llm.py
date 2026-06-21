"""
llm.py — Language Model Integration


PRIMARY: Groq API with Llama 3.3 70B — this is the system the dissertation
actually measures and reports on.

FALLBACK: Google Gemini (free tier) — kicks in ONLY when Groq specifically
returns a daily-token-quota error (the "tokens per day" 429 you've been
hitting). Every other Groq error still raises normally, unchanged. This
keeps Groq/Llama 3.3 as the system of record while removing the daily
quota wall as something that can block a whole evaluation run.

Gemini's free tier (1,500 requests/day on Gemini 2.5 Flash, no card
required) is far more generous than Groq's free tier (~100K tokens/day,
which in practice caps out around 100 calls/day) — so a handful of
overflow calls per day costs it almost nothing.

Two modes (unchanged from before, same signatures, same return types):
  ask_llama_streaming() — RAG mode (with document context), streams tokens
  ask_llama()            — RAG mode, non-streaming (experiment/eval pages)
  ask_baseline()          — Baseline mode (no document context)
"""

import os
import time
import streamlit as st
from groq import Groq

# ── Groq client setup (primary) ─────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY not found. "
        "Add it to .streamlit/secrets.toml or Streamlit Cloud secrets."
    )
    st.stop()

client = Groq(api_key=api_key)
MODEL  = "llama-3.3-70b-versatile"

# ── Gemini client setup (fallback — only used when Groq's daily quota hits) ────
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_AVAILABLE = False

if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

# Audit trail — every time the Gemini fallback actually fires gets logged
# here, so you can check after a run whether any answers in your dataset
# came from Gemini rather than Llama 3.3 (Groq). Inspect with
# `import llm; llm.FALLBACK_LOG` from any page.
FALLBACK_LOG = []


def _is_quota_error(e: Exception) -> bool:
    """True only for Groq's daily-token-quota error — NOT for other
    failures (bad request, auth issue, network error, etc.), which
    should still raise normally rather than silently rerouting."""
    msg = str(e).lower()
    return "rate_limit" in msg or "429" in msg or "tokens per day" in msg


def _log_fallback(function_name: str, reason: str):
    FALLBACK_LOG.append({
        "time":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "function": function_name,
        "reason":   reason,
    })
    try:
        st.toast(f"Groq daily quota hit — this answer used the Gemini fallback instead.", icon="⚠️")
    except Exception:
        pass  # st.toast can fail outside a normal Streamlit run context — non-fatal


def _messages_to_gemini(messages: list):
    """
    Convert an OpenAI-style messages list (system/user/assistant) into
    Gemini's format: a system_instruction string plus a list of
    {"role": "user"|"model", "parts": [...]} turns. Gemini uses "model"
    where OpenAI/Groq use "assistant".
    """
    system_instruction = None
    gemini_turns = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
        elif m["role"] == "user":
            gemini_turns.append({"role": "user", "parts": [m["content"]]})
        elif m["role"] == "assistant":
            gemini_turns.append({"role": "model", "parts": [m["content"]]})
    return system_instruction, gemini_turns


def _raise_if_no_gemini():
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "Groq's daily token quota was hit, and the Gemini fallback isn't "
            "available (GOOGLE_API_KEY not set or google-generativeai not "
            "installed). Add GOOGLE_API_KEY to .streamlit/secrets.toml and "
            "run: pip install google-generativeai"
        )


def _ask_gemini(messages: list, max_tokens: int, function_name: str, reason: str) -> str:
    """Non-streaming Gemini fallback call — same prompt, different provider."""
    _raise_if_no_gemini()
    _log_fallback(function_name, reason)

    system_instruction, gemini_turns = _messages_to_gemini(messages)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_instruction)
    *history_turns, last_turn = gemini_turns
    chat = model.start_chat(history=history_turns)
    response = chat.send_message(
        last_turn["parts"][0],
        generation_config={"max_output_tokens": max_tokens},
    )
    return response.text


def _ask_gemini_streaming(messages: list, max_tokens: int, function_name: str, reason: str):
    """Streaming Gemini fallback — used by ask_llama_streaming()."""
    _raise_if_no_gemini()
    _log_fallback(function_name, reason)

    system_instruction, gemini_turns = _messages_to_gemini(messages)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_instruction)
    *history_turns, last_turn = gemini_turns
    chat = model.start_chat(history=history_turns)
    response = chat.send_message(
        last_turn["parts"][0],
        generation_config={"max_output_tokens": max_tokens},
        stream=True,
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


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
    Falls back to Gemini, still streaming, if Groq's daily quota is hit.
    """
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    try:
        stream = client.chat.completions.create(
            model=MODEL, max_tokens=1024, messages=messages, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
    except Exception as e:
        if not _is_quota_error(e):
            raise
        yield from _ask_gemini_streaming(
            messages, max_tokens=1024,
            function_name="ask_llama_streaming", reason=str(e),
        )


# ── RAG non-streaming (experiment runner, RAGAS, Local Metrics) ────────────────
def ask_llama(question: str, retrieved_chunks: list, history: list) -> str:
    """Returns full answer as a string. Falls back to Gemini if Groq's
    daily quota is hit — same signature and return type either way, so
    every existing caller (Experiment Runner, RAGAS, Local Metrics) needs
    no changes."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    try:
        response = client.chat.completions.create(
            model=MODEL, max_tokens=1024, messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        if not _is_quota_error(e):
            raise
        return _ask_gemini(
            messages, max_tokens=1024,
            function_name="ask_llama", reason=str(e),
        )


# ── Baseline — no document context ────────────────────────────────────────────
def ask_baseline(question: str) -> str:
    """Falls back to Gemini if Groq's daily quota is hit, same as above."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question using only your general knowledge. Be direct and concise.",
        },
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL, max_tokens=512, messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        if not _is_quota_error(e):
            raise
        return _ask_gemini(
            messages, max_tokens=512,
            function_name="ask_baseline", reason=str(e),
        )