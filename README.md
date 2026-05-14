# Nigerian Data Protection Act (NDPA) 2023 RAG System

A full lifecycle Retrieval-Augmented Generation (RAG) system built using the Nigerian Data Protection Act (NDPA) 2023 as the knowledge base.

This project demonstrates how to:

- Extract and process PDF documents
- Handle OCR for scanned PDFs
- Chunk and clean legal text
- Generate embeddings
- Build a FAISS vector index
- Implement hybrid retrieval (Dense + BM25)
- Integrate an open-source LLM
- Build a FastAPI inference API
- Persist and reload vector indexes
- Serve production-style RAG responses

The project was built primarily for learning practical AI systems engineering and production-oriented RAG architecture.

---

# Architecture Overview

```text
PDF Document
    ↓
OCR + Text Extraction
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Index
    ↓
Hybrid Retrieval (FAISS + BM25)
    ↓
Prompt Construction
    ↓
TinyLlama Generation
    ↓
FastAPI Deployment
```

---

# Tech Stack

## Core ML / NLP

- Python
- Sentence Transformers
- HuggingFace Transformers
- TinyLlama
- FAISS
- BM25

## OCR & Document Processing

- PyMuPDF
- Tesseract OCR
- pdf2image
- Poppler

## API & Deployment

- FastAPI
- Uvicorn

---

# Project Structure

```text
RAG1/
│
├── notebooks/
│   └── NDPA-2023.ipynb
│
├── data/
│   └── NDPA_2023.pdf
│
├── app.py
├── requirements.txt
│
├── ndpa_faiss.index
├── chunks.pkl
├── metadata.pkl
│
└── README.md
```

---

# Features

- OCR support for scanned legal PDFs
- Hybrid retrieval:
  - Dense semantic search
  - BM25 lexical search
- Open-source local LLM inference
- Persistent FAISS indexing
- Modular FastAPI backend
- Retrieval-augmented generation pipeline
- Legal document question answering

---

# Why Hybrid Retrieval?

Pure embedding retrieval struggles with:

- exact legal terminology
- clause references
- statutory language
- compliance wording

This project combines:

```text
Dense Retrieval + BM25
```

to improve legal-domain retrieval quality.

---

# Open Source Models Used

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

## LLM

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

TinyLlama was selected because it:

- runs locally
- is lightweight
- is fully open source
- works on CPU laptops

---

# Additional Dependencies (Windows)

## Install Tesseract OCR

Download:

https://github.com/UB-Mannheim/tesseract/wiki

Add to PATH:

```text
C:\Program Files\Tesseract-OCR
```

Verify:

```powershell
tesseract --version
```

---

## Install Poppler

Download:

https://github.com/oschwartz10612/poppler-windows/releases

Extract and add to PATH:

```text
C:\poppler\Library\bin
```

Verify:

```powershell
pdfinfo -v
```

---
# Future Improvements

Potential upgrades include:

- Qdrant instead of FAISS
- reranking models
- metadata filtering
- streaming responses
- authentication
- Docker deployment
- Ollama integration
- vLLM inference
- GPU acceleration
- LangChain/LlamaIndex orchestration

---

# Learning Goals

This project was built to understand:

- end-to-end RAG systems
- production AI architecture
- local open-source LLM deployment
- retrieval systems
- enterprise AI engineering concepts

---

# Disclaimer

This project is for educational and research purposes only.

Generated responses may contain inaccuracies or hallucinations. Always verify legal interpretations with qualified legal professionals.

---

# References

## NDPA 2023

Nigerian Data Protection Act 2023

## Libraries

- HuggingFace Transformers
- Sentence Transformers
- FAISS
- FastAPI
- PyMuPDF
- Tesseract OCR

---
