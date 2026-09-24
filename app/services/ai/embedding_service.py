"""Vector embedding service with Gemini and local fallback support."""
import hashlib
from typing import List, Optional
import numpy as np
from app.config import settings
from app.logging_config import get_logger
from app.models.document import DocumentChunk
from app.services.ai.gemini_client import GeminiClient, gemini_client
from app.services.storage.vector_store import VectorStore, vector_store

logger = get_logger(__name__)


class EmbeddingService:
    """Generates high-dimensional vector embeddings with automatic fallback."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        vec_store: Optional[VectorStore] = None,
    ):
        self.client = client or gemini_client
        self.vector_store = vec_store or vector_store

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of strings."""
        if not texts:
            return []

        # Try Gemini embeddings if available
        if self.client.is_available():
            try:
                embeddings = self.client.embed_content(texts)
                logger.debug("Generated %d Gemini embeddings", len(texts))
                return embeddings
            except Exception as e:
                logger.warning("Gemini embedding failed (%s); falling back to local vectors", e)

        # Local deterministic hashing fallback
        return [self._local_fallback_embedding(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * settings.EMBEDDING_DIM

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Compute embeddings for document chunks and add them to VectorStore."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        doc_ids = [c.document_id for c in chunks]

        embeddings = self.embed_texts(texts)
        self.vector_store.add_vectors(
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
            embeddings=embeddings,
        )
        logger.info("Indexed %d chunks into VectorStore", len(chunks))

    @staticmethod
    def _local_fallback_embedding(text: str, dim: int = 384) -> List[float]:
        """Deterministic TF-IDF/n-gram hashing embedding for offline mode."""
        vector = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vector.tolist()

        for word in words:
            # Deterministic hash to bucket
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vector[idx] += sign

        # Character trigrams for morphological similarity
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            h = int(hashlib.sha1(trigram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vector[idx] += 0.5

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()


# Global embedding service
embedding_service = EmbeddingService()
