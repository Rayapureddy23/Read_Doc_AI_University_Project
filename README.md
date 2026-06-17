# ReadDoc AI

**AI-powered document question answering system**
*MSc Data Science — University of Hertfordshire*

---

## Research question

> How does varying chunk size and retrieval depth in a Retrieval-Augmented Generation pipeline affect the accuracy, contextual relevance, and faithfulness of answers generated from unstructured documents?

---

## What this project does

ReadDoc AI is a RAG-based document assistant that lets users upload any PDF or HTML document and ask questions about it in natural language. It retrieves the most relevant sections semantically and generates a sourced answer using Llama 3.3.

The system is also the experimental platform for an empirical study investigating how two configuration parameters — chunk size and retrieval depth — affect answer quality across 9 configurations.

---

## Project structure

```
readdoc-ai/
├── app.py                      Main chat interface
├── rag.py                      RAG pipeline (ingestion, chunking, embedding, FAISS, retrieval)
├── llm.py                      LLM integration (Groq + Llama 3.3)
├── database.py                 SQLite chat history
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml             Theme settings
└── pages/
    ├── 1_Project_Overview.py   Research question, aim, objectives, timeline
    ├── 2_How_It_Works.py       Pipeline explained with code and reasoning
    ├── 3_EDA.py                Document exploratory data analysis
    ├── 4_Baseline_Test.py      Control experiment — LLM with no documents
    ├── 5_Experiment_Runner.py  Run all 9 RAG configurations
    └── 6_Results_Analysis.py   Charts and dissertation findings
```

---

## RAG pipeline

```
PDF / HTML upload
    → Text extraction (PyPDF / BeautifulSoup)
    → Chunking (300 / 600 / 1000 chars + 100-char overlap)
    → Embedding (all-MiniLM-L6-v2 → 384-dim vectors)
    → FAISS IndexFlatL2 (exact L2 search)
    → User question → semantic search (top-k = 3 / 5 / 10)
    → Llama 3.3 generates answer from retrieved chunks only
    → Streamed answer + source citations
```

---

## Technology stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 3.3 via Groq API |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Vector search | FAISS IndexFlatL2 |
| PDF ingestion | PyPDF |
| HTML ingestion | BeautifulSoup4 |
| Chat history | SQLite |

---

## Quick start

```bash
git clone https://github.com/Rayapureddy23/Read_Doc_AI_University_Project.git
cd Read_Doc_AI_University_Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Edit .streamlit/secrets.toml
streamlit run app.py
```

---

## Experiment design

| Configuration | Chunk size | Retrieval depth (k) |
|---|---|---|
| E1 | 300 | 3 |
| E2 | 300 | 5 |
| E3 | 300 | 10 |
| E4 | 600 | 3 |
| E5 | 600 | 5 |
| E6 | 600 | 10 |
| E7 | 1000 | 3 |
| E8 | 1000 | 5 |
| E9 | 1000 | 10 |
| BASELINE | n/a | n/a |

Test document: Kroese, D.P. et al. (2020). *Data Science and Machine Learning*. CRC Press.

---

## Project stages

| Stage | Content | Status |
|---|---|---|
| 1 | Foundation — RAG pipeline, EDA, GitHub repo | Complete |
| 2 | Experiments — Run E1–E9 + baseline, score results | In progress |
| 3 | Dissertation — Chapters 1–5 | Upcoming |
| 4 | Final — Deploy, submit | Upcoming |

---

## Key references

- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain QA. *EMNLP*.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.
- Johnson, J. et al. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
- Gao, Y. et al. (2023). RAG for Large Language Models: A Survey. *arXiv:2312.10997*.

---

## Author
Dharmendra kumar Reddy Rayapureddy
rayapureddydharmendra0404@gmail.com
MSc Data Science — University of Hertfordshire