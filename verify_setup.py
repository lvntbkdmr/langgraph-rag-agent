import sys
import os

print("Verifying imports...")
try:
    from app.config import settings
    print(f"Config loaded. FAISS path: {settings.FAISS_INDEX_PATH}")
    
    from app.llm.qwen import qwen_client
    print("Qwen client loaded.")
    
    from app.rag.loaders import DocumentProcessor
    print("Document processor loaded.")
    
    from app.rag.vectorstore import vector_store
    print("Vector store loaded.")
    
    from app.rag.retriever import retriever
    print("Retriever loaded.")
    
    from app.agent.state import AgentState
    print("Agent state loaded.")
    
    from app.agent.nodes import mode_detector
    print("Agent nodes loaded.")
    
    from app.agent.graph import agent_runner
    print("Agent graph loaded.")
    
    from app.main import app
    print("FastAPI app loaded.")
    
    print("\nALL CHECKS PASSED!")
except Exception as e:
    print(f"\nERROR: {e}")
    sys.exit(1)
