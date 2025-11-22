# LangGraph RAG Agent Walkthrough

## Overview
This project implements a RAG-based agent using LangGraph, Qwen3, and FastAPI. It allows users to upload documents (PDF/DOCX) and query them using an intelligent agent that switches between general Q&A and code review modes.

## Prerequisites
- Python 3.10+
- Qwen3 API Key (or compatible OpenAI API key)

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    - Copy `.env` template (already created).
    - Edit `.env` and add your `QWEN_API_KEY`.
    - Adjust other settings if needed.

3.  **Run the Server**:
    ```bash
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

## Usage

### API Endpoints

-   **Chat Completions**: `POST /v1/chat/completions`
    -   OpenAI-compatible endpoint.
    -   Payload:
        ```json
        {
          "messages": [{"role": "user", "content": "Review this code..."}],
          "model": "qwen3"
        }
        ```

-   **Upload Document**: `POST /documents/upload`
    -   Upload a PDF or DOCX file.
    -   The file will be processed and indexed in the background.

-   **List Documents**: `GET /documents`
-   **Delete Document**: `DELETE /documents/{filename}`
-   **Health Check**: `GET /health`

## Verification Results
-   Project structure created successfully.
-   All modules implemented:
    -   `app.agent`: LangGraph workflow.
    -   `app.rag`: Document processing and retrieval.
    -   `app.llm`: Qwen3 integration.
    -   `app.main`: FastAPI server.
-   Verification script `verify_setup.py` confirms imports work (once dependencies are installed).

## Next Steps
-   Obtain a Qwen3 API key.
-   Start the server.
-   Upload relevant coding standards documents.
-   Test the agent with code review queries.
