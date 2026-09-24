"""SQLite persistence engine for document metadata, chunks, and operational data."""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.config import settings
from app.logging_config import get_logger
from app.models.document import DocumentMetadata, DocumentChunk

logger = get_logger(__name__)


class Database:
    """Thread-safe SQLite manager for document storage."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DB_PATH
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        """Initialize database schema tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_hash TEXT NOT NULL,
                    page_count INTEGER DEFAULT 1,
                    word_count INTEGER DEFAULT 0,
                    char_count INTEGER DEFAULT 0,
                    reading_time_minutes REAL DEFAULT 0.0,
                    detected_language TEXT DEFAULT 'en',
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    title TEXT,
                    author TEXT,
                    extracted_entities TEXT DEFAULT '{}',
                    summary TEXT DEFAULT '{}'
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);")

            # Chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER DEFAULT 1,
                    section_title TEXT DEFAULT 'General',
                    token_count INTEGER DEFAULT 0,
                    char_start INTEGER DEFAULT 0,
                    char_end INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_order ON chunks(document_id, chunk_index);")

            # Chat / RAG session history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);")
            conn.commit()
            logger.debug("Database schema initialized at %s", self.db_path)

    # ------------------ Document Operations ------------------

    def save_document(self, doc: DocumentMetadata) -> None:
        """Insert or replace a document record."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents (
                    document_id, filename, original_filename, file_type, mime_type,
                    file_size, file_hash, page_count, word_count, char_count,
                    reading_time_minutes, detected_language, created_at, status,
                    error_message, title, author, extracted_entities, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.document_id,
                doc.filename,
                doc.original_filename,
                doc.file_type,
                doc.mime_type,
                doc.file_size,
                doc.file_hash,
                doc.page_count,
                doc.word_count,
                doc.char_count,
                doc.reading_time_minutes,
                doc.detected_language,
                doc.created_at,
                doc.status,
                doc.error_message,
                doc.title,
                doc.author,
                json.dumps(doc.extracted_entities or {}),
                json.dumps(doc.summary or {})
            ))
            conn.commit()

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Fetch document metadata by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]:
        """Fetch document by content SHA-256 hash for deduplication."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[DocumentMetadata]:
        """List documents ordered by creation date descending."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            return [self._row_to_document(r) for r in rows]

    def count_documents(self) -> int:
        """Total number of documents."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def update_document_status(self, document_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Update processing status of a document."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE document_id = ?",
                (status, error_message, document_id)
            )
            conn.commit()

    def update_document_summary(self, document_id: str, summary: Dict[str, Any]) -> None:
        """Update summary JSON for a document."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE documents SET summary = ? WHERE document_id = ?",
                (json.dumps(summary), document_id)
            )
            conn.commit()

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and cascade delete its chunks."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ------------------ Chunk Operations ------------------

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Bulk insert or replace document chunks."""
        if not chunks:
            return
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, chunk_index, text, page_number,
                    section_title, token_count, char_start, char_end, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c.chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.page_number,
                    c.section_title,
                    c.token_count,
                    c.char_start,
                    c.char_end,
                    json.dumps(c.metadata or {})
                )
                for c in chunks
            ])
            conn.commit()

    def get_chunks_for_document(self, document_id: str) -> List[DocumentChunk]:
        """Fetch all chunks for a document ordered by index."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
                (document_id,)
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch a single chunk by chunk ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if not row:
                return None
            return self._row_to_chunk(row)

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Fetch chunks by list of chunk IDs preserving order."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
            chunk_dict = {r["chunk_id"]: self._row_to_chunk(r) for r in rows}
            return [chunk_dict[cid] for cid in chunk_ids if cid in chunk_dict]

    def delete_chunks_for_document(self, document_id: str) -> int:
        """Delete chunks belonging to a document."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount

    def count_chunks(self) -> int:
        """Total number of chunks in the database."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_system_stats(self) -> Dict[str, Any]:
        """Compute aggregate system metrics."""
        with self._get_connection() as conn:
            doc_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    COALESCE(SUM(page_count), 0) as total_pages,
                    COALESCE(SUM(word_count), 0) as total_words,
                    COALESCE(SUM(file_size), 0) as total_bytes
                FROM documents
            """).fetchone()
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            return {
                "total_documents": doc_stats["total_docs"],
                "total_pages": doc_stats["total_pages"],
                "total_words": doc_stats["total_words"],
                "total_bytes": doc_stats["total_bytes"],
                "total_chunks": chunk_count,
            }

    # ------------------ Helper Converters ------------------

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentMetadata:
        """Convert a SQLite row to DocumentMetadata."""
        entities = {}
        if row["extracted_entities"]:
            try:
                entities = json.loads(row["extracted_entities"])
            except Exception:
                entities = {}
                
        summary = {}
        if row["summary"]:
            try:
                summary = json.loads(row["summary"])
            except Exception:
                summary = {}

        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            original_filename=row["original_filename"],
            file_type=row["file_type"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
            file_hash=row["file_hash"],
            page_count=row["page_count"],
            word_count=row["word_count"],
            char_count=row["char_count"],
            reading_time_minutes=row["reading_time_minutes"],
            detected_language=row["detected_language"],
            created_at=row["created_at"],
            status=row["status"],
            error_message=row["error_message"],
            title=row["title"],
            author=row["author"],
            extracted_entities=entities,
            summary=summary,
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
        """Convert a SQLite row to DocumentChunk."""
        meta = {}
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = {}

        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            token_count=row["token_count"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            metadata=meta,
        )


# Global database instance
db = Database()
