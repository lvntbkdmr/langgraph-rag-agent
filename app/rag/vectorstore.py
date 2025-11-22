import faiss
import numpy as np
import os
import pickle
from langchain.schema import Document
from app.llm.qwen import qwen_client
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []  # List to store metadata/content corresponding to index
        self.dimension = 1536 # Qwen/OpenAI embedding dimension usually, but will verify dynamically if possible or assume 1536/1024. Qwen-embedding-v3 is likely 1024 or 1536. Let's assume 1024 for now or check.
        # Actually, better to initialize on first add or load.
        self._load_index()

    def _load_index(self):
        if os.path.exists(settings.FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
                # Load metadata
                meta_path = settings.FAISS_INDEX_PATH + ".meta"
                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        self.documents = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.index = None
                self.documents = []
        else:
            logger.info("No existing index found. Starting fresh.")

    def _save_index(self):
        if self.index:
            os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
            with open(settings.FAISS_INDEX_PATH + ".meta", "wb") as f:
                pickle.dump(self.documents, f)
            logger.info("Saved FAISS index to disk")

    def add_documents(self, chunks: list[Document]):
        if not chunks:
            return

        embeddings = []
        for chunk in chunks:
            emb = qwen_client.get_embeddings(chunk.page_content)
            embeddings.append(emb)
        
        embeddings_np = np.array(embeddings).astype('float32')
        
        if self.index is None:
            self.dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings_np)
        self.documents.extend(chunks)
        self._save_index()

    def similarity_search(self, query: str, k: int = settings.RETRIEVAL_K) -> list[Document]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = qwen_client.get_embeddings(query)
        query_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                # Optionally add score to metadata
                doc.metadata['score'] = float(distances[0][i])
                results.append(doc)
        
        return results

vector_store = VectorStore()
