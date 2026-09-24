"""Okapi BM25 sparse keyword indexing and scoring engine."""
import json
import math
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from app.config import settings
from app.logging_config import get_logger
from app.models.document import DocumentChunk

logger = get_logger(__name__)

# Common English stop words
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_-]{2,}\b")


def tokenize(text: str) -> List[str]:
    """Tokenize and filter stop words from text."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in raw_tokens if t not in STOP_WORDS]


class BM25Index:
    """Okapi BM25 Index with inverted term frequencies and persistence."""

    def __init__(self, storage_dir: Optional[Path] = None, k1: float = 1.5, b: float = 0.75):
        self.storage_dir = storage_dir or settings.VECTOR_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "bm25_index.json"
        
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        
        # State
        self.doc_freqs: Dict[str, int] = {}  # term -> number of chunks containing it
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # chunk_id -> {term: count}
        self.doc_lens: Dict[str, int] = {}  # chunk_id -> token length
        self.chunk_to_doc: Dict[str, str] = {}  # chunk_id -> document_id
        self.avg_doc_len: float = 0.0
        self.num_chunks: int = 0
        
        self.load()

    def load(self) -> None:
        """Load BM25 state from disk."""
        with self._lock:
            if self.index_file.exists():
                try:
                    with open(self.index_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.doc_freqs = data.get("doc_freqs", {})
                    self.term_freqs = data.get("term_freqs", {})
                    self.doc_lens = data.get("doc_lens", {})
                    self.chunk_to_doc = data.get("chunk_to_doc", {})
                    self.avg_doc_len = data.get("avg_doc_len", 0.0)
                    self.num_chunks = data.get("num_chunks", 0)
                    logger.info("Loaded BM25 index with %d chunks from disk", self.num_chunks)
                except Exception as e:
                    logger.error("Failed loading BM25 index: %s", e)
                    self._reset()
            else:
                self._reset()

    def _reset(self) -> None:
        """Reset internal BM25 state."""
        self.doc_freqs = {}
        self.term_freqs = {}
        self.doc_lens = {}
        self.chunk_to_doc = {}
        self.avg_doc_len = 0.0
        self.num_chunks = 0

    def save(self) -> None:
        """Persist BM25 state to disk."""
        with self._lock:
            try:
                data = {
                    "doc_freqs": self.doc_freqs,
                    "term_freqs": self.term_freqs,
                    "doc_lens": self.doc_lens,
                    "chunk_to_doc": self.chunk_to_doc,
                    "avg_doc_len": self.avg_doc_len,
                    "num_chunks": self.num_chunks,
                }
                with open(self.index_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error("Failed saving BM25 index: %s", e)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add new chunks to the BM25 inverted index."""
        if not chunks:
            return
            
        with self._lock:
            for chunk in chunks:
                tokens = tokenize(chunk.text)
                tf: Dict[str, int] = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                    
                self.term_freqs[chunk.chunk_id] = tf
                self.doc_lens[chunk.chunk_id] = len(tokens)
                self.chunk_to_doc[chunk.chunk_id] = chunk.document_id
                
                # Update document frequencies
                for term in tf:
                    self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

            self.num_chunks = len(self.term_freqs)
            if self.num_chunks > 0:
                self.avg_doc_len = sum(self.doc_lens.values()) / self.num_chunks
            else:
                self.avg_doc_len = 0.0

            self.save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search top-k matching chunks using Okapi BM25 scoring."""
        with self._lock:
            if self.num_chunks == 0:
                return []
                
            query_tokens = tokenize(query)
            if not query_tokens:
                return []

            target_doc_set = set(document_ids) if document_ids else None
            scores: Dict[str, float] = {}
            N = self.num_chunks

            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue
                    
                df = self.doc_freqs[term]
                # Standard BM25 IDF
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

                for chunk_id, tf_map in self.term_freqs.items():
                    # Filter by document ID if requested
                    if target_doc_set and self.chunk_to_doc.get(chunk_id) not in target_doc_set:
                        continue

                    if term in tf_map:
                        f = tf_map[term]
                        doc_len = self.doc_lens[chunk_id]
                        denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                        score = idf * ((f * (self.k1 + 1.0)) / denom)
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + score

            # Sort results descending
            ranked = sorted(
                [(cid, score) for cid, score in scores.items() if score >= min_score],
                key=lambda x: x[1],
                reverse=True
            )
            return ranked[:top_k]

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks of a document from the BM25 index."""
        with self._lock:
            chunks_to_remove = [
                cid for cid, did in self.chunk_to_doc.items() if did == document_id
            ]
            if not chunks_to_remove:
                return 0

            for cid in chunks_to_remove:
                tf_map = self.term_freqs.pop(cid, {})
                self.doc_lens.pop(cid, None)
                self.chunk_to_doc.pop(cid, None)
                for term in tf_map:
                    if term in self.doc_freqs:
                        self.doc_freqs[term] -= 1
                        if self.doc_freqs[term] <= 0:
                            del self.doc_freqs[term]

            self.num_chunks = len(self.term_freqs)
            if self.num_chunks > 0:
                self.avg_doc_len = sum(self.doc_lens.values()) / self.num_chunks
            else:
                self.avg_doc_len = 0.0

            self.save()
            return len(chunks_to_remove)

    def count(self) -> int:
        """Total chunks in BM25 index."""
        with self._lock:
            return self.num_chunks

    def clear(self) -> None:
        """Clear all indexed terms from memory and disk."""
        with self._lock:
            self._reset()
            self.save()


# Global BM25 index instance
bm25_index = BM25Index()
