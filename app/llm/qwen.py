from openai import OpenAI
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL
        )

    def get_embeddings(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=settings.QWEN_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def chat_completion(self, messages: list[dict], temperature: float = 0.7, stream: bool = False):
        try:
            return self.client.chat.completions.create(
                model=settings.QWEN_CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

qwen_client = QwenClient()
