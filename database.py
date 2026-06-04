"""
database.py — Persistent Chat History
=======================================
ReadDoc AI | MSc Data Science and Analytics

SQLite database with two tables:
  conversations — one row per chat session
  messages      — one row per message (user or assistant)
"""

import sqlite3
import datetime
import os

DB_PATH = os.path.join("data", "chat.db")
os.makedirs("data", exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT    NOT NULL DEFAULT 'New chat',
            created_at TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT    NOT NULL,
            content         TEXT    NOT NULL,
            created_at      TEXT    NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(title: str = "New chat") -> int:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    now  = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def list_conversations() -> list:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    rows = [{"id": r[0], "title": r[1], "created_at": r[2]} for r in c.fetchall()]
    conn.close()
    return rows


def delete_conversation(conversation_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages      WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?",              (conversation_id,))
    conn.commit()
    conn.close()


def save_message(conversation_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, now),
    )
    conn.commit()
    conn.close()


def load_messages(conversation_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    )
    rows = [{"role": r[0], "content": r[1]} for r in c.fetchall()]
    conn.close()
    return rows


def get_message_count(conversation_id: int) -> int:
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,))
    count = c.fetchone()[0]
    conn.close()
    return count


def auto_title(conversation_id: int, first_question: str):
    title = (first_question.strip()[:60] + "...") if len(first_question) > 60 else first_question.strip()
    conn  = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    conn.commit()
    conn.close()