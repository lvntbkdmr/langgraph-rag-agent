from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import settings
import os
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_and_split(self, file_path: str) -> list[Document]:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
            return chunks
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
