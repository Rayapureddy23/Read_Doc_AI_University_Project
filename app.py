"""
app.py — ReadDoc AI Main Chat Interface
========================================
ReadDoc AI | MSc Data Science and Analytics

Main chat page. Upload documents, ask questions, get sourced answers.
"""

import streamlit as st
import os
import database as db
import rag
from llm import ask_llama_streaming

st.set_page_config(
    page_title="ReadDoc AI",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.stApp{background:#F0F4FF}
.main .block-container{background:#fff;border-radius:16px;margin:1rem 1rem 0 .5rem;
    padding:2rem 2.5rem!important;
    box-shadow:0 1px 3px rgba(0,0,0,.06),0 8px 32px rgba(26,86,219,.05);
    min-height:calc(100vh - 2rem)}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #E5E9F5!important}
[data-testid="stSidebar"]>div{padding:0!important}
[data-testid="stSidebarContent"]{padding:1.5rem 1.2rem!important}
.brand-header{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.brand-logo{width:36px;height:36px;background:#1a56db;border-radius:8px;
    display:flex;align-items:center;justify-content:center;flex-shrink:0}
.brand-name{font-size:17px;font-weight:700;color:#111827}
.brand-name span{color:#1a56db}
.brand-sub{font-size:12px;color:#9CA3AF;margin-bottom:1.2rem}
.sidebar-label{font-size:10.5px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
    letter-spacing:.08em;margin:1.2rem 0 .4rem}
[data-testid="stSidebar"] .stButton:first-of-type>button{
    background:transparent!important;border:1.5px solid #1a56db!important;
    color:#1a56db!important;border-radius:8px!important;font-weight:600!important;font-size:13px!important}
[data-testid="stSidebar"] .stButton:first-of-type>button:hover{
    background:#1a56db!important;color:#fff!important}
[data-testid="stSidebar"] .stButton>button{
    background:transparent!important;border:none!important;color:#374151!important;
    text-align:left!important;font-size:13px!important;padding:7px 10px!important;
    border-radius:7px!important;white-space:nowrap!important;overflow:hidden!important;
    text-overflow:ellipsis!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#EEF2FF!important;color:#1a56db!important}
[data-testid="stSidebar"] .element-container:last-child .stButton>button{
    background:#1a56db!important;color:white!important;border:none!important;
    border-radius:8px!important;font-weight:600!important}
[data-testid="stFileUploader"]{background:#F8FAFF!important;
    border:1.5px dashed #C7D3F5!important;border-radius:10px!important;padding:6px!important}
[data-testid="stFileUploader"] label{display:none}
.sdot{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:#10B981;margin-right:5px;vertical-align:middle}
.sdot.off{background:#D1D5DB}
.idx-status{font-size:12px;color:#6B7280;margin-top:6px}
.chat-title{font-size:19px;font-weight:700;color:#111827;margin-bottom:1.4rem;
    padding-bottom:.8rem;border-bottom:1px solid #F3F4F6}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
    background:#F8FAFF!important;border:1px solid #E5E9F5!important;
    border-radius:18px 18px 18px 4px!important;padding:14px 20px!important;
    max-width:80%!important;color:#1F2937!important}
[data-testid="stChatMessageAvatarAssistant"]{background:#1a56db!important;border-radius:50%!important}
[data-testid="stChatInput"]{border:1.5px solid #E5E9F5!important;
    border-radius:12px!important;background:#F8FAFF!important}
[data-testid="stChatInput"]:focus-within{border-color:#1a56db!important}
[data-testid="stChatInput"] textarea{font-size:14px!important;color:#111827!important}
[data-testid="stChatInputSubmitButton"] button{background:#1a56db!important;
    border-radius:8px!important;border:none!important}
.src-row{margin-top:10px}
.src-label{font-size:10.5px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
    letter-spacing:.06em;margin-bottom:5px}
.src-tag{display:inline-block;background:#EEF2FF;border:1px solid #C7D3F5;color:#1a56db;
    font-size:11.5px;font-weight:500;padding:3px 10px;border-radius:20px;
    margin:2px 4px 2px 0;font-family:'DM Mono',monospace}
.empty-wrap{text-align:center;padding:4rem 2rem}
.empty-icon-box{width:60px;height:60px;background:#1a56db;border-radius:16px;
    margin:0 auto 1.4rem;display:flex;align-items:center;justify-content:center}
.empty-wrap h3{font-size:20px;font-weight:700;color:#111827;margin-bottom:8px}
.empty-wrap p{font-size:14px;color:#6B7280;max-width:340px;margin:0 auto 1.5rem;line-height:1.6}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.chip{background:#EEF2FF;border:1px solid #C7D3F5;color:#1a56db;
    font-size:12px;font-weight:500;padding:5px 14px;border-radius:20px}
hr{border:none!important;border-top:1px solid #F3F4F6!important;margin:1rem 0!important}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:#E5E9F5;border-radius:10px}
</style>
""", unsafe_allow_html=True)

SVG = '''<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
  stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
  <polyline points="14 2 14 8 20 8"/>
  <line x1="16" y1="13" x2="8" y2="13"/>
  <line x1="16" y1="17" x2="8" y2="17"/>
</svg>'''

# ── Setup ──────────────────────────────────────────────────────────────────────
db.init_db()
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = rag.load_index_from_disk(
        chunk_size=st.session_state.get("chunk_size", 600)
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="brand-header">
        <div class="brand-logo">{SVG}</div>
        <div class="brand-name">ReadDoc <span>AI</span></div>
    </div>
    <div class="brand-sub">Reads your documents. Answers your questions.</div>
    """, unsafe_allow_html=True)

    if st.button("+ New chat", use_container_width=True):
        new_id = db.create_conversation()
        st.session_state.conversation_id = new_id
        st.rerun()

    conversations = db.list_conversations()
    if conversations:
        st.markdown('<div class="sidebar-label">Recent chats</div>', unsafe_allow_html=True)
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            is_active  = st.session_state.get("conversation_id") == conv["id"]
            with col1:
                if st.button(
                    ("» " if is_active else "  ") + conv["title"],
                    key=f"conv_{conv['id']}", use_container_width=True
                ):
                    st.session_state.conversation_id = conv["id"]
                    st.rerun()
            with col2:
                if st.button("✕", key=f"del_{conv['id']}"):
                    db.delete_conversation(conv["id"])
                    if st.session_state.get("conversation_id") == conv["id"]:
                        del st.session_state["conversation_id"]
                    st.rerun()

    st.divider()

    # Experiment settings
    st.markdown('<div class="sidebar-label">Experiment settings</div>', unsafe_allow_html=True)
    chunk_size = st.select_slider(
        "Chunk size (chars)",
        options=[300, 600, 1000],
        value=st.session_state.get("chunk_size", 600),
    )
    st.session_state.chunk_size = chunk_size

    top_k = st.select_slider(
        "Retrieval depth (k)",
        options=[3, 5, 10],
        value=st.session_state.get("top_k", 5),
    )
    st.session_state.top_k = top_k

    st.markdown(
        f'<div class="idx-status" style="margin-top:2px">'
        f'Chunk: <b>{chunk_size}</b> chars &nbsp;·&nbsp; k: <b>{top_k}</b></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Upload
    st.markdown('<div class="sidebar-label">Upload documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF or HTML", type=["pdf", "html", "htm"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if st.button("Build index", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            saved_paths = []
            os.makedirs("data/uploads", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("data/uploads", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)
            with st.spinner(f"Building index — chunk {chunk_size} chars..."):
                result = rag.build_index(saved_paths, chunk_size=chunk_size)
            st.session_state.index_loaded = True
            if result.get("cached"):
                st.success(f"Loaded from cache — {result['total_chunks']} chunks")
            else:
                st.success(f"Built — {result['total_chunks']} chunks · {result['total_files']} file(s)")
            st.session_state.close_sidebar = True

    status = rag.get_status()
    if status["loaded"]:
        st.markdown(
            f'<div class="idx-status"><span class="sdot"></span>'
            f'{status["total_chunks"]} chunks · {len(status["files"])} file(s)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="idx-status"><span class="sdot off"></span>No index — upload files above</div>',
            unsafe_allow_html=True,
        )

if st.session_state.get("close_sidebar"):
    st.session_state.close_sidebar = False
    st.markdown("""<script>
        const btn=window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if(btn)btn.click();</script>""", unsafe_allow_html=True)

# ── Main chat area ─────────────────────────────────────────────────────────────
status = rag.get_status()
if "conversation_id" not in st.session_state:
    convs = db.list_conversations()
    st.session_state.conversation_id = convs[0]["id"] if convs else db.create_conversation()

conversation_id = st.session_state.conversation_id
messages        = db.load_messages(conversation_id)
conv_list       = db.list_conversations()
current_conv    = next((c for c in conv_list if c["id"] == conversation_id), None)
title           = current_conv["title"] if current_conv else "New chat"

st.markdown(f'<div class="chat-title">{title}</div>', unsafe_allow_html=True)

if not messages:
    st.markdown(f"""
    <div class="empty-wrap">
        <div class="empty-icon-box">{SVG}</div>
        <h3>How can I help you today?</h3>
        <p>Upload any document — PDF or HTML — and ask me anything about it.</p>
        <div class="chips">
            <span class="chip">PDF &amp; HTML support</span>
            <span class="chip">Semantic search</span>
            <span class="chip">Source citations</span>
            <span class="chip">Real-time answers</span>
        </div>
    </div>""", unsafe_allow_html=True)

# Render history
for msg in messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    else:
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:.5rem 0">'
            f'<div style="background:#1a56db;color:#fff;border-radius:18px 18px 4px 18px;'
            f'padding:12px 18px;max-width:70%;font-size:14px;line-height:1.6;'
            f'font-family:DM Sans,sans-serif">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )

user_input = st.chat_input(
    "Ask a question about your documents..." if status["loaded"]
    else "Upload and index a document to start chatting...",
    disabled=not status["loaded"],
)

if user_input:
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:.5rem 0">'
        f'<div style="background:#1a56db;color:#fff;border-radius:18px 18px 4px 18px;'
        f'padding:12px 18px;max-width:70%;font-size:14px;line-height:1.6;'
        f'font-family:DM Sans,sans-serif">{user_input}</div></div>',
        unsafe_allow_html=True,
    )
    db.save_message(conversation_id, "user", user_input)
    if db.get_message_count(conversation_id) == 1:
        db.auto_title(conversation_id, user_input)

    retrieved_chunks = rag.search(user_input, top_k=st.session_state.get("top_k", 5))
    history          = db.load_messages(conversation_id)
    history_for_api  = history[:-1]

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        try:
            for chunk in ask_llama_streaming(user_input, retrieved_chunks, history_for_api):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Something went wrong: {str(e)}"
            placeholder.error(full_response)

        if retrieved_chunks:
            seen = set()
            tags = ""
            for src in retrieved_chunks:
                key = f"{src['file_name']}·p{src['page_number']}"
                if key not in seen:
                    seen.add(key)
                    tags += f'<span class="src-tag">{src["file_name"]} · p.{src["page_number"]}</span>'
            st.markdown(
                f'<div class="src-row"><div class="src-label">Sources</div>{tags}</div>',
                unsafe_allow_html=True,
            )

    db.save_message(conversation_id, "assistant", full_response)
    st.rerun()