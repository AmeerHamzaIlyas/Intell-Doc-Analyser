"""Persistent NumPy-based Vector Store with cosine similarity search and metadata filtering."""
import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Thread-safe vector store storing L2-normalized embeddings for fast cosine similarity."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.VECTOR_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.vectors_file = self.storage_dir / "vectors.npy"
        self.meta_file = self.storage_dir / "vector_meta.json"
        
        self._lock = threading.RLock()
        self.chunk_ids: List[str] = []
        self.doc_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, D), float32
        
        self.load()

    def load(self) -> None:
        """Load vector embeddings and metadata from disk."""
        with self._lock:
            if self.vectors_file.exists() and self.meta_file.exists():
                try:
                    self.vectors = np.load(str(self.vectors_file)).astype(np.float32)
                    with open(self.meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.chunk_ids = meta.get("chunk_ids", [])
                    self.doc_ids = meta.get("doc_ids", [])
                    logger.info("Loaded %d vectors from %s", len(self.chunk_ids), self.storage_dir)
                except Exception as e:
                    logger.error("Failed loading vector store from disk: %s", e)
                    self.vectors = None
                    self.chunk_ids = []
                    self.doc_ids = []
            else:
                self.vectors = None
                self.chunk_ids = []
                self.doc_ids = []

    def save(self) -> None:
        """Persist vector embeddings and metadata to disk."""
        with self._lock:
            try:
                if self.vectors is not None and len(self.chunk_ids) > 0:
                    np.save(str(self.vectors_file), self.vectors)
                    with open(self.meta_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "chunk_ids": self.chunk_ids,
                            "doc_ids": self.doc_ids
                        }, f)
                elif self.vectors_file.exists():
                    self.vectors_file.unlink(missing_ok=True)
                    self.meta_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error("Failed saving vector store to disk: %s", e)

    def add_vectors(
        self,
        chunk_ids: List[str],
        doc_ids: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """Add new vector embeddings for chunks and normalize them for cosine similarity."""
        if not chunk_ids or not embeddings:
            return
            
        with self._lock:
            new_vecs = np.array(embeddings, dtype=np.float32)
            # L2 normalize each vector: norm = sqrt(sum(x^2))
            norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0.0] = 1.0
            new_vecs = new_vecs / norms

            if self.vectors is None or len(self.chunk_ids) == 0:
                self.vectors = new_vecs
                self.chunk_ids = list(chunk_ids)
                self.doc_ids = list(doc_ids)
            else:
                # Handle dimension alignment if switching between local and cloud embedding models
                if self.vectors.shape[1] != new_vecs.shape[1]:
                    target_dim = max(self.vectors.shape[1], new_vecs.shape[1])
                    if self.vectors.shape[1] < target_dim:
                        padded_old = np.zeros((self.vectors.shape[0], target_dim), dtype=np.float32)
                        padded_old[:, :self.vectors.shape[1]] = self.vectors
                        self.vectors = padded_old
                    if new_vecs.shape[1] < target_dim:
                        padded_new = np.zeros((new_vecs.shape[0], target_dim), dtype=np.float32)
                        padded_new[:, :new_vecs.shape[1]] = new_vecs
                        new_vecs = padded_new

                self.vectors = np.vstack([self.vectors, new_vecs])
                self.chunk_ids.extend(chunk_ids)
                self.doc_ids.extend(doc_ids)

            self.save()
            logger.debug("Added %d vectors (total: %d)", len(chunk_ids), len(self.chunk_ids))

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search top-k most similar chunks using cosine similarity.
        
        Returns:
            List of (chunk_id, similarity_score) where score in [-1.0, 1.0] (usually [0, 1]).
        """
        with self._lock:
            if self.vectors is None or len(self.chunk_ids) == 0:
                return []

            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            else:
                return []

            # Filter by document IDs if specified
            if document_ids:
                doc_set = set(document_ids)
                indices = [i for i, d in enumerate(self.doc_ids) if d in doc_set]
                if not indices:
                    return []
                subset_vectors = self.vectors[indices]
                subset_chunk_ids = [self.chunk_ids[i] for i in indices]
            else:
                subset_vectors = self.vectors
                subset_chunk_ids = self.chunk_ids

            # Dot product with normalized vectors gives exact cosine similarity
            sims = np.dot(subset_vectors, q_vec)

            # Retrieve top_k
            effective_k = min(top_k, len(subset_chunk_ids))
            if effective_k == 0:
                return []

            # Sort descending
            top_indices = np.argsort(-sims)[:effective_k]

            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if score >= min_score:
                    results.append((subset_chunk_ids[idx], score))

            return results

    def delete_document_vectors(self, document_id: str) -> int:
        """Remove all vectors belonging to a document."""
        with self._lock:
            if not self.chunk_ids or document_id not in self.doc_ids:
                return 0

            keep_indices = [i for i, d in enumerate(self.doc_ids) if d != document_id]
            removed_count = len(self.doc_ids) - len(keep_indices)

            if len(keep_indices) == 0:
                self.vectors = None
                self.chunk_ids = []
                self.doc_ids = []
            else:
                self.vectors = self.vectors[keep_indices]
                self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
                self.doc_ids = [self.doc_ids[i] for i in keep_indices]

            self.save()
            logger.info("Deleted %d vectors for document %s", removed_count, document_id)
            return removed_count

    def count(self) -> int:
        """Total vectors indexed."""
        with self._lock:
            return len(self.chunk_ids)

    def clear(self) -> None:
        """Clear all vectors from memory and disk."""
        with self._lock:
            self.vectors = None
            self.chunk_ids = []
            self.doc_ids = []
            self.save()


# Global vector store instance
vector_store = VectorStore()
