"""
llm.py — Language Model Integration
=====================================
Uses Google Gemini API (free tier) for answer generation — Groq has been
fully removed from this file. Groq's free-tier daily token quota
(~100K tokens/day, effectively ~100 calls/day) repeatedly blocked
evaluation runs partway through. Gemini's free tier (1,500 requests/day
on Gemini 2.5 Flash, no credit card required) is far more generous for
this workload and resets daily.

IMPORTANT — dissertation consistency: if your Project Overview /
Methodology chapter states "Llama 3.3 via Groq API" as the tech stack,
that text now needs updating to say Gemini, since this is the system
that actually generates every answer going forward. Keep your written
methodology matching what was actually run.

NOTE: the RAGAS Evaluation page's judge model is a SEPARATE Groq client
(llama-3.1-8b-instant, its own much larger daily quota) defined inside
8_RAGAS_Evaluation.py — this file does not touch that. If you want Groq
removed from the judge too, that's a separate change, just say so.

Three functions (same signatures and return types as before, so no
other file in the project needs to change):
  ask_llama_streaming() — RAG mode (with document context), streams tokens
  ask_llama()            — RAG mode, non-streaming (experiment/eval pages)
  ask_baseline()          — Baseline mode (no document context)
"""

import os
import streamlit as st
import google.generativeai as genai

# ── Client setup ───────────────────────────────────────────────────────────────
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY not found. "
        "Add it to .streamlit/secrets.toml (local) AND Streamlit Cloud's "
        "Settings → Secrets (deployed app — these are two separate stores, "
        "and the app must be rebooted after adding a secret there)."
    )
    st.stop()

genai.configure(api_key=api_key)
MODEL = "gemini-2.5-flash"


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
    system_instruction, gemini_turns = _messages_to_gemini(messages)
    model = genai.GenerativeModel(MODEL, system_instruction=system_instruction)
    *history_turns, last_turn = gemini_turns
    chat = model.start_chat(history=history_turns)
    response = chat.send_message(
        last_turn["parts"][0],
        generation_config={"max_output_tokens": 1024},
        stream=True,
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── RAG non-streaming (experiment runner, RAGAS, Local Metrics) ────────────────
def ask_llama(question: str, retrieved_chunks: list, history: list) -> str:
    """Returns full answer as a string — used by Experiment Runner, RAGAS
    Evaluation, and Local Metrics."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": build_user_message(question, retrieved_chunks)}]
    )
    system_instruction, gemini_turns = _messages_to_gemini(messages)
    model = genai.GenerativeModel(MODEL, system_instruction=system_instruction)
    *history_turns, last_turn = gemini_turns
    chat = model.start_chat(history=history_turns)
    response = chat.send_message(
        last_turn["parts"][0],
        generation_config={"max_output_tokens": 1024},
    )
    return response.text


# ── Baseline — no document context ────────────────────────────────────────────
def ask_baseline(question: str) -> str:
    model = genai.GenerativeModel(
        MODEL,
        system_instruction="You are a helpful assistant. Answer the question using only your general knowledge. Be direct and concise.",
    )
    response = model.generate_content(
        question,
        generation_config={"max_output_tokens": 512},
    )
    return response.text