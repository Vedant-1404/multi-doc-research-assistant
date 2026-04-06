# Multi-Doc Research Assistant

> LlamaIndex · FAISS · Groq · Streamlit

Extends RAG to **multiple sources simultaneously** with source-level citation, retrieval scoring, and document management. Built to demonstrate production-grade RAG orchestration thinking.

## What this adds over PDF Q&A Chatbot
| Capability | PDF Q&A Chatbot | Multi-Doc Research Assistant |
|---|---|---|
| Sources | Single PDF | Multiple PDFs + URLs |
| Vector store | ChromaDB | FAISS (local, fast) |
| Orchestration | LangChain | LlamaIndex |
| Citations | Basic | Per-source with score + page |
| Retrieval | Fixed top-k | Hybrid + reranking |
| UI | FastAPI REST | Streamlit full UI |

## Project Structure
```
multi-doc-research-assistant/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── config.py            # Settings & constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py        # PDF + URL document loaders
│   │   └── pipeline.py      # Chunking, embedding, FAISS indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── engine.py        # LlamaIndex query engine
│   │   └── reranker.py      # Retrieval quality / reranking logic
│   ├── generation/
│   │   ├── __init__.py
│   │   └── synthesizer.py   # Response synthesis with citations
│   └── ui/
│       ├── __init__.py
│       ├── components.py    # Reusable Streamlit components
│       └── styles.py        # Custom CSS
├── storage/                 # FAISS indexes (gitignored)
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/Vedant-1404/multi-doc-research-assistant.git
cd multi-doc-research-assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 5. Run
PYTHONPATH=. streamlit run app/main.py
```

## Key Design Decisions

**Why LlamaIndex over LangChain here?**
LlamaIndex's `VectorStoreIndex` gives native node-level metadata (source file, page number, score) — critical for per-source citations without extra plumbing.

**Why FAISS over ChromaDB?**
FAISS is purely in-memory/on-disk — no server needed, deterministic retrieval, easy to snapshot and reload per session. Shows you can pick the right vector store for the use case.

**Retrieval quality signals shown in UI:**
- Cosine similarity score per retrieved chunk
- Source document + page number
- Token count of retrieved context
- Whether answer was found vs. synthesized from partial context
