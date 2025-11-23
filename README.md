# LangGraph RAG Agent

A FastAPI-based intelligent agent using LangGraph and Qwen3 (via OpenAI-compatible API) for RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Code Review Agent**: Specialized agent for strict code review against coding standards.
- **RAG Capabilities**: Upload PDF and DOCX documents (coding standards, guidelines) to build a knowledge base.
- **Vector Search**: Uses FAISS for efficient similarity search.
- **FastAPI Server**: Provides REST endpoints for chat completions and document management.
- **OpenAI Compatible**: The chat endpoint is compatible with OpenAI client libraries.

## Project Structure

```
langgraph-agent/
├── app/
│   ├── agent/          # LangGraph workflow and nodes
│   ├── rag/            # Document processing and vector store
│   ├── llm/            # Qwen3 client wrapper
│   ├── main.py         # FastAPI application
│   └── config.py       # Configuration settings
├── data/               # Storage for uploaded docs and FAISS index
├── requirements.txt    # Python dependencies
└── .env                # Environment variables
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    - Copy `.env` template (if provided) or create one with:
      ```
      QWEN_API_KEY=your_key
      QWEN_BASE_URL=https://api.qwen.ai/v1
      ```

3.  **Run Server**:
    ```bash
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

## Usage

- **Upload Documents**: POST `/documents/upload`
- **Chat**: POST `/v1/chat/completions`

See `walkthrough.md` for a detailed guide.
