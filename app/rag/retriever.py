from app.rag.vectorstore import vector_store
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.vector_store = vector_store

    def retrieve(self, query: str) -> list[Document]:
        try:
            docs = self.vector_store.similarity_search(query)
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

retriever = Retriever()
