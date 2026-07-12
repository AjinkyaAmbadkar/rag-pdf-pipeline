# RAG PDF Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that ingests PDFs and text files, chunks and embeds them, stores the embeddings in a ChromaDB vector store, and answers questions over the content using a Groq-hosted LLM.

## Features

- **Document ingestion** — load PDFs (via PyMuPDF/PyPDF) and `.txt` files (via LangChain's `TextLoader`/`DirectoryLoader`)
- **Chunking** — split documents into overlapping chunks with `RecursiveCharacterTextSplitter`
- **Embeddings** — generate sentence embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store** — persist and query embeddings with ChromaDB
- **Retrieval** — similarity search with configurable `top_k` and score threshold
- **Generation** — answer questions over retrieved context using Groq's LLM (`llama-3.1-8b-instant` via `langchain-groq`), with an advanced variant that also returns sources and a confidence score

## Project Structure

```
rag-pdf-pipeline/
├── data/
│   ├── pdf_files/         # Source PDFs
│   ├── text_files/        # Source text files
│   └── vector_store/      # Persisted ChromaDB store (generated, gitignored)
├── notebook/
│   ├── document.ipynb     # LangChain Document/loader exploration
│   └── pdf_loader.ipynb   # Full pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate
├── main.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Setup

**Requirements:** Python >= 3.13

1. Clone the repo and install dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your [Groq API key](https://console.groq.com/keys):

   ```
   GROQ_API_KEY="your-api-key-here"
   ```

## Usage

The pipeline is driven from `notebook/pdf_loader.ipynb`:

1. Place PDFs in `data/pdf_files/` (and/or text files in `data/text_files/`).
2. Run the notebook top to bottom to ingest documents, generate embeddings, and populate the ChromaDB vector store at `data/vector_store/`.
3. Use `rag_simple(query, rag_retriever, llm)` for a quick answer, or `rag_advanced(query, rag_retriever, llm, top_k, min_score, return_context=True)` for an answer with sources and a confidence score.

`notebook/document.ipynb` is a smaller companion notebook exploring LangChain's `Document` structure and loaders.

## Tech Stack

- [LangChain](https://python.langchain.com/) (community loaders, text splitters, Groq integration)
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [Groq](https://groq.com/) for LLM inference
