# LangGraph RAG Agent Implementation Plan

## Goal Description
Generate a FastAPI-based LangGraph RAG Agent project as described in `plan.md`. The system will include a FastAPI server, LangGraph agent with RAG capabilities, Qwen3 integration, and FAISS vector store.

## Proposed Changes

### Project Structure
Create the following directory structure:
```
langgraph-agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectorstore.py
│   │   ├── loaders.py
│   │   └── retriever.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── qwen.py
│   └── config.py
├── data/
│   ├── documents/
│   └── faiss_index/
├── requirements.txt
├── .env
└── README.md
```

### Dependencies
- `langgraph`, `langchain`, `langchain-community`
- `faiss-cpu`
- `fastapi`, `uvicorn`, `python-multipart`
- `pypdf`, `python-docx`
- `openai`
- `pydantic`, `python-dotenv`

### Components
1.  **Configuration**: `config.py` for managing settings.
2.  **LLM**: `app/llm/qwen.py` wrapper for Qwen3 via OpenAI client.
3.  **RAG**:
    *   `loaders.py`: PDF/DOCX loaders.
    *   `vectorstore.py`: FAISS index management.
    *   `retriever.py`: Retrieval logic.
4.  **Agent**:
    *   `state.py`: `AgentState` definition.
    *   `nodes.py`: Mode detector, RAG decision, Retrieval, Response generation.
    *   `graph.py`: LangGraph construction.
5.  **Server**: `app/main.py` with `/v1/chat/completions` and `/documents/upload` endpoints.

## Verification Plan

### Automated Tests
- Create a simple test script `test_setup.py` to verify:
    - Imports work.
    - Directories exist.
    - Environment variables are loaded.

### Manual Verification
- Run the server: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Check health endpoint: `curl http://localhost:8000/health`
