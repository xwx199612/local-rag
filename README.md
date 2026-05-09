# 📄 Offline RAG AI Assistant

A fully local Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions using a local LLM (Ollama).

## Features
- PDF document QA
- Fully offline (no API required)
- FAISS vector search
- Local LLM (Llama3 / Mistral)
- Streamlit UI

## Architecture
PDF → Chunking → Embedding → FAISS → Retrieval → LLM → Answer

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Install Ollama
https://ollama.com

Pull model:
ollama pull llama3.1

### 3. Run app
streamlit run app/main.py

## Tech Stack
- Python
- FAISS
- Sentence Transformers
- Ollama
- Streamlit
