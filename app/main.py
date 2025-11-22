from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
import os
import shutil
from langchain.schema import HumanMessage, SystemMessage

from app.agent.graph import run_agent
from app.rag.loaders import DocumentProcessor
from app.rag.vectorstore import vector_store
from app.config import settings

app = FastAPI(title="LangGraph RAG Agent")

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

# Endpoints

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # Convert messages to LangChain format
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content))
        # We can ignore assistant messages for now or add them if we want history
    
    # Run agent
    try:
        final_answer = run_agent(lc_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Construct response
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=final_answer),
                finish_reason="stop"
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Placeholder
    )

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX are supported.")
    
    os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)
    file_path = os.path.join(settings.DOCUMENTS_PATH, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    def process_file(path):
        processor = DocumentProcessor()
        chunks = processor.load_and_split(path)
        vector_store.add_documents(chunks)
        
    if background_tasks:
        background_tasks.add_task(process_file, file_path)
    else:
        # Fallback if no background tasks (shouldn't happen in FastAPI)
        process_file(file_path)
        
    return {"message": "File uploaded and processing started", "filename": file.filename}

@app.get("/documents")
async def list_documents():
    # This is a simplified view. In a real app, we'd query a DB or the vector store metadata.
    # Here we just list files in the directory.
    if not os.path.exists(settings.DOCUMENTS_PATH):
        return []
    files = os.listdir(settings.DOCUMENTS_PATH)
    return [{"filename": f} for f in files]

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = os.path.join(settings.DOCUMENTS_PATH, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        # Note: Deleting from FAISS is hard without rebuilding. 
        # For now we just delete the file. Re-indexing would be needed to clean up FAISS.
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "vector_store_size": vector_store.index.ntotal if vector_store.index else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
