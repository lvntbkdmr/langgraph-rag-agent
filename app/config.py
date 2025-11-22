import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://api.qwen.ai/v1")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_CHAT_MODEL: str = os.getenv("QWEN_CHAT_MODEL", "qwen-max")
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
    
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/index.faiss")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "data/documents/")
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    CONTEXT_WINDOW: int = 32000
    
    # Derived settings
    RETRIEVAL_K: int = 5

settings = Settings()
