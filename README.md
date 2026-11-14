# DocuMind: Enterprise RAG & Evaluation Pipeline

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to answer complex questions from dense technical documents (e.g., academic papers, financial filings). 

## Architecture
- **Ingestion:** PyPDF + `RecursiveCharacterTextSplitter` for parsing and chunking PDFs.
- **Embeddings:** OpenAI `text-embedding-3-small`.
- **Vector Store:** ChromaDB (local persistence).
- **Backend:** FastAPI (Python).
- **Evaluation:** A custom "LLM-as-a-Judge" pipeline to grade the system against a golden dataset.

## Key Features
- **Citation Grounding:** The API returns the exact page number and text snippet where the answer was found, preventing hallucinations.
- **Automated Evals:** A Python script (`evaluate.py`) runs a suite of 5 test questions against the API and uses an LLM to grade the accuracy, preventing regressions.
- **Guardrails:** The system explicitly says "I don't have enough information" if the retrieved context does not contain the answer.

## Current Performance
The system scores **80% (4/5)** on the golden dataset. 
The eval harness successfully caught a failure mode where the retriever pulled the base model's BLEU score (28.4) instead of the state-of-the-art model's score (41.8). Future iterations will implement a cross-encoder reranker or metadata filtering to resolve conflicting numerical data.

## How to Run
1. `pip install -r requirements.txt`
2. Add `OPENAI_API_KEY` to `.env`
3. `python ingest.py`
4. `uvicorn main:app --reload`
5. `python evaluate.py`