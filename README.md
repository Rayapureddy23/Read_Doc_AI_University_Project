# ReadDoc AI

**Optimising Retrieval-Augmented Generation for Document Question Answering: An Empirical Study of Chunking Strategies and Retrieval Depth**

*MSc Data Science — Empirical Research Project*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://readdocai.streamlit.app)

---

## Research Question

> *"How does varying chunk size and retrieval depth in a Retrieval-Augmented Generation pipeline affect the accuracy, contextual relevance, and faithfulness of answers generated from unstructured documents?"*

---

## Overview

ReadDoc AI is a full-stack Retrieval-Augmented Generation (RAG) system built as an empirical research platform. It allows systematic investigation of how two key pipeline parameters — **chunk size** and **retrieval depth (k)** — affect the quality of LLM-generated answers from unstructured PDF documents.

The system implements a 3 × 3 factorial experimental design (3 chunk sizes × 3 retrieval depths = 9 configurations), evaluated against a zero-retrieval baseline, using a standardised 10-question test set scored on a 0.0–1.0 rubric across three dimensions: **accuracy**, **contextual relevance**, and **faithfulness**.

---

## The RAG Pipeline

```
PDF Upload
    ↓
Text Extraction (PyPDF / BeautifulSoup4)
    ↓
Text Chunking (300 / 600 / 1000 chars, 100-char overlap)
    ↓
Embedding (all-MiniLM-L6-v2, 384 dimensions)
    ↓
FAISS Index (IndexFlatL2, exact nearest-neighbour)
    ↓
Query Embedding → Retrieve top-k chunks (k = 3 / 5 / 10)
    ↓
LLM Answer Generation (Llama 3.3 70B via Groq API)
    ↓
Human Evaluation (0.0–1.0 rubric, three metrics)
    ↓
Results Analysis & Comparison
```

---

## Experimental Design

| Experiment | Chunk size | k | Description |
|---|---|---|---|
| Baseline | — | — | Zero document context (control condition) |
| E1 | 300 | 3 | Short chunks, focused retrieval |
| E2 | 300 | 5 | Short chunks, medium retrieval |
| E3 | 300 | 10 | Short chunks, broad retrieval |
| E4 | 600 | 3 | Mid chunks, focused retrieval |
| E5 | 600 | 5 | Mid chunks, medium retrieval |
| E6 | 600 | 10 | Mid chunks, broad retrieval |
| E7 | 1000 | 3 | Long chunks, focused retrieval |
| E8 | 1000 | 5 | Long chunks, medium retrieval |
| E9 | 1000 | 10 | Long chunks, broad retrieval |

---

## Evaluation Metrics (0.0 – 1.0 Scale)

| Metric | Description | Baseline value |
|---|---|---|
| **Accuracy** | Factual correctness of the answer against the document | Scored vs general knowledge |
| **Contextual Relevance** | Whether the answer addresses the question asked | Same scale |
| **Faithfulness** | Whether every claim is grounded in retrieved chunks (no hallucination) | Always 0.00 — no document |

**Scoring rubric:** 1.00 = Perfect · 0.75 = Good · 0.50 = Moderate · 0.25 = Poor · 0.00 = Wrong/None

The overall score per configuration is the average of the three metrics.

---

## Application Pages

| Page | Purpose |
|---|---|
| **app** | Live RAG chat — upload documents, build index, ask questions with source citations |
| **Project Overview** | Research question, objectives, experimental design, metrics, references |
| **How It Works** | Step-by-step pipeline walkthrough with technical explanation |
| **EDA** | Exploratory data analysis of the source document and chunk statistics |
| **Baseline Test** | Run 10 questions with zero context, score manually (control condition) |
| **Experiment Runner** | Run 10 questions for E1–E9, score accuracy/relevance/faithfulness |
| **Results Analysis** | Comparison table, charts, auto-written findings, CSV export |

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend / App framework | Streamlit |
| Document parsing | PyPDF, BeautifulSoup4 |
| Embedding model | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector index | FAISS IndexFlatL2 |
| Language model | Llama 3.3 70B (llama-3.3-70b-versatile) |
| LLM inference | Groq API |
| Data storage | SQLite |
| Language | Python 3.11 |

---

## Installation & Setup

### Prerequisites

- Python 3.11
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Rayapureddy23/Read_Doc_AI_University_Project.git
cd Read_Doc_AI_University_Project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create .streamlit/secrets.toml with:
# GROQ_API_KEY = "your-key-here"

# Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork or connect this repository to [share.streamlit.io](https://share.streamlit.io)
2. Set `GROQ_API_KEY` in Settings → Secrets
3. The app deploys automatically on every push to `main`

---

## How to Run the Experiments

1. **Upload your document** on the main ReadDoc AI page and click **Build index — all chunk sizes**. Wait for all three sizes to complete.

2. **Run the Baseline Test** (Page 4) — click Generate, score all 10 answers, click Save per question.

3. **Run Experiments E1–E9** (Page 5) — select each configuration, generate answers, score and save.

4. **View Results** (Page 6) — the comparison table, charts, and auto-written findings appear automatically once scores are saved.

5. **Export** — download the full results CSV from the Results Analysis page for dissertation appendix.

---

## Project Structure

```
ReadDocumentAI/
├── app.py                    # Main chat interface
├── rag.py                    # RAG pipeline (chunking, embedding, retrieval)
├── llm.py                    # LLM integration (Groq API)
├── database.py               # SQLite chat history
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml          # GROQ_API_KEY (not committed)
├── pages/
│   ├── 1_Project_Overview.py
│   ├── 2_How_It_Works.py
│   ├── 3_EDA.py
│   ├── 4_Baseline_Test.py
│   ├── 5_Experiment_Runner.py
│   └── 6_Results_Analysis.py
└── data/                     # Created at runtime (gitignored)
    ├── experiments.db        # All evaluation scores
    ├── faiss_*.index         # Cached FAISS indexes
    └── uploads/              # Uploaded documents
```

---

## Key References

1. **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S.** (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*. https://arxiv.org/abs/2309.15217

2. **Lewis, P., et al.** (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

3. **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

4. **Johnson, J., Douze, M., & Jégou, H.** (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547.

5. **Kroese, D.P., et al.** (2020). *Data Science and Machine Learning: Mathematical and Statistical Methods.* CRC Press. *(Source document used in evaluation)*

---

## Author

**Dharmendrakumar Reddy Rayapureddy**
MSc Data Science Project — 2025/2026

*ReadDoc AI is built as an empirical research platform for an MSc dissertation. The live app is available at [readdocai.streamlit.app](https://readdocai.streamlit.app).*