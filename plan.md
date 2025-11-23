# LangGraph RAG Agent Implementation Plan

## Architecture Overview

The system will consist of:

- **FastAPI server** with two main endpoints: `/v1/chat/completions` (OpenAI-compatible) and `/documents/upload` (document management)
- **LangGraph agent** with RAG capabilities using FAISS vector store
- **Qwen3 integration** for LLM and embeddings (OpenAI-compatible client)
- **Persistent FAISS index** saved to disk
- **Automatic mode detection** in the agent's decision logic

## Implementation Steps

### 1. Project Setup and Dependencies

Create a Python project structure:

```
langgraph-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph workflow
│   │   ├── nodes.py         # Agent nodes (RAG, routing, etc.)
│   │   └── state.py         # Agent state definition
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectorstore.py   # FAISS management
│   │   ├── loaders.py       # PDF/DOCX loaders
│   │   └── retriever.py     # Retrieval logic
│   ├── llm/
│   │   ├── __init__.py
│   │   └── qwen.py          # Qwen3 client wrapper
│   └── config.py            # Configuration management
├── data/
│   ├── documents/           # Uploaded documents
│   └── faiss_index/         # Persistent FAISS storage
├── requirements.txt
├── .env                     # Configuration file
└── README.md
```

**Dependencies to include:**

- `langgraph` - Agent framework
- `langchain` and `langchain-community` - Document loaders, text splitters
- `faiss-cpu` (or `faiss-gpu`) - Vector store
- `fastapi` and `uvicorn` - API server
- `python-multipart` - File upload support
- `pypdf` - PDF parsing
- `python-docx` - DOCX parsing
- `openai` - To connect to Qwen3 (OpenAI-compatible)
- `pydantic` - Data validation
- `python-dotenv` - Environment configuration

### 2. Configuration Management

Create `config.py` to manage:

- Qwen3 base URL and API key
- Qwen3 model names (chat and embeddings)
- FAISS index path
- Document storage path
- Chunk size and overlap for text splitting
- Context window size (32000 tokens)
- Temperature and other LLM parameters

Use environment variables from `.env` file for sensitive data.

### 3. Qwen3 Integration

Create wrapper in `app/llm/qwen.py`:

- Use `openai` Python client with custom `base_url` pointing to Qwen3
- Create two clients: one for chat completions, one for embeddings
- Implement retry logic and error handling
- Create a `get_embeddings()` function that returns vectors
- Create a `chat_completion()` function for LLM calls

### 4. Document Processing and RAG

**Document Loaders** (`app/rag/loaders.py`):

- PDF loader using `pypdf` (fallback to `PyMuPDF` if needed)
- DOCX loader using `python-docx`
- Text splitter with configurable chunk size (consider the 32k context window)
- Extract metadata (filename, page numbers, etc.)

**Vector Store Management** (`app/rag/vectorstore.py`):

- Initialize FAISS index on startup (load from disk if exists)
- Create `add_documents()` method that:
  - Loads document (PDF/DOCX)
  - Splits into chunks with RecursiveCharacterTextSplitter
  - Generates embeddings using Qwen3
  - Adds to FAISS index
  - Saves index to disk
- Create `similarity_search()` method for retrieval
- Implement `save_index()` and `load_index()` for persistence
- Track document metadata separately (JSON file)

**Retriever** (`app/rag/retriever.py`):

- Wrap FAISS with retrieval logic
- Implement relevance scoring
- Return top-k chunks with metadata
- Handle empty index gracefully

### 5. LangGraph Agent Implementation

**State Definition** (`app/agent/state.py`):
- Define `AgentState` with TypedDict:
  - `messages`: List of chat messages
  - `retrieved_docs`: Retrieved document chunks
  - `final_answer`: Generated response

**Agent Nodes** (`app/agent/nodes.py`):

1. **Retrieval Node**:

   - Query FAISS with user's question
   - Retrieve top 3-5 relevant chunks
   - Store in `state['retrieved_docs']`

2. **Response Generation Node**:

   - Build prompt for **Code Review**: "You are a strict code reviewer. Use the provided coding standards to evaluate the code. Be thorough and cite specific standards."
   - Include retrieved documents in context
   - Call Qwen3 LLM
   - Manage context window (32k tokens - reserve space for retrieved docs)
   - Store response in `state['final_answer']`

**Graph Construction** (`app/agent/graph.py`):

- Create StateGraph with conditional edges:
  ```
  START → retrieval → response_generation → END
  ```

- Compile the graph
- Export `run_agent()` function that takes messages and returns final answer

### 6. FastAPI Server Implementation

**Main Server** (`app/main.py`):

1. **Initialize on startup**:

   - Load FAISS index from disk (or create empty)
   - Initialize Qwen3 clients
   - Set up LangGraph agent

2. **POST /v1/chat/completions**:

   - Accept OpenAI-compatible request body:
     - `model`: (ignored, always use Qwen3)
     - `messages`: Array of message objects
     - `temperature`, `max_tokens`, `stream`: Standard OpenAI params
   - Run LangGraph agent with messages
   - Return OpenAI-compatible response format:
     ```json
     {
       "id": "chatcmpl-xxx",
       "object": "chat.completion",
       "created": timestamp,
       "model": "qwen3",
       "choices": [{
         "index": 0,
         "message": {"role": "assistant", "content": "..."},
         "finish_reason": "stop"
       }],
       "usage": {...}
     }
     ```

   - Support streaming if requested (optional, but recommended)

3. **POST /documents/upload**:

   - Accept file upload (PDF or DOCX)
   - Validate file type
   - Save to `data/documents/`
   - Process and add to FAISS index
   - Return success response with document ID

4. **GET /documents**:

   - List all indexed documents
   - Return metadata (filename, page count, upload date)

5. **DELETE /documents/{doc_id}**:

   - Remove document from index (if supported)
   - Delete file from storage

6. **GET /health**:

   - Health check endpoint
   - Verify Qwen3 connection
   - Report index status (document count)

### 7. Mode Detection Logic

Implement intelligent mode detection in the mode detector node:

**Code Review Indicators**:

- Keywords: "review", "check", "complies", "compliance", "coding standard", "style guide", "function", "class", "violate", "meets requirements"
- Presence of code snippets in the message (detect by triple backticks or indentation)
- Phrases like "does this", "is this", "should I", "correct"

**System Prompts**:

- **Code review mode prompt**:
  ```
  You are a strict code reviewer specializing in [company] coding standards. 
  Carefully analyze the provided code against the coding standards in the context.
  Be specific about violations and cite the relevant standards.
  Provide clear, actionable feedback.
  ```

- **General mode prompt**:
  ```
  You are a helpful technical assistant with access to company documentation.
  Answer questions clearly and cite sources when using specific document information.
  ```


### 8. FAISS Persistence Strategy

**Saving**:

- Save after every document addition
- Use `faiss.write_index()` to save binary index
- Save metadata separately (JSON file with document info)
- Save in `data/faiss_index/` directory

**Loading**:

- On startup, check if index exists
- Use `faiss.read_index()` to load binary index
- Load metadata from JSON
- If no index exists, create empty one

**Index Management**:

- Consider periodic re-indexing for optimization
- Implement backup mechanism (copy index files before updates)

### 9. Error Handling and Logging

- Add comprehensive logging (use Python `logging` module)
- Log all API requests and responses
- Log Qwen3 API calls and errors
- Handle common errors:
  - Qwen3 connection failures
  - Document parsing errors
  - Empty FAISS index (no documents)
  - Invalid file uploads
  - Context window overflow

### 10. Windows 10 Deployment

**Installation Steps**:

1. Install Python 3.10+ (from python.org)
2. Clone/copy the project to local machine
3. Create virtual environment: `python -m venv venv`
4. Activate: `venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Configure `.env` file with Qwen3 endpoint details
7. Run: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`

**Windows Service** (optional):

- Use `nssm` (Non-Sucking Service Manager) to run as Windows service
- Or use Task Scheduler to start on boot

**Configuration for Intranet**:

- Bind to specific IP if needed
- No external internet required (all local/intranet)
- Consider firewall rules for the port

### 11. Testing Strategy

**Unit Tests**:

- Test document loaders with sample PDF/DOCX
- Test embedding generation
- Test FAISS save/load
- Test mode detection logic

**Integration Tests**:

- Test full agent workflow
- Test API endpoints with curl/Postman
- Test with actual Qwen3 endpoint

**Manual Testing**:

- Upload sample coding standards documents
- Test code review queries
- Test general questions
- Verify FAISS persistence (restart server)

## Key Considerations

- **Context Window Management**: With 32k tokens, reserve ~5-10k for retrieved docs, rest for conversation
- **Chunk Size**: Use ~500-1000 token chunks with 100 token overlap for optimal retrieval
- **Retrieval Count**: Retrieve 3-5 chunks to balance context and relevance
- **Error Recovery**: Gracefully handle Qwen3 downtime (return appropriate error messages)
- **Performance**: FAISS is fast, but consider index size limits (10k+ documents may need optimization)

## Optional Enhancements

- Add conversation history management (store past sessions)
- Implement document re-indexing endpoint
- Add support for more file types (TXT, MD, HTML)
- Create a simple web UI for testing
- Add metrics and monitoring
- Implement caching for frequent queries
- Add user authentication if needed