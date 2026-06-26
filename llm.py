import os
import streamlit as st
from groq import Groq

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)
MODEL  = "llama-3.3-70b-versatile"

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

For greetings and general chat — respond naturally and introduce yourself as ReadDoc AI."""

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

def ask_llama_streaming(question: str, retrieved_chunks: list, history: list):
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
            if delta:
                yield delta
    except Exception as e:
        yield f"Something went wrong: {str(e)}"

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
