"""Document ingestion, validation, and processing orchestrator."""
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from app.config import settings
from app.core.constants import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PROCESSING,
)
from app.core.exceptions import (
    FileCorruptedError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.core.security import (
    compute_file_hash,
    get_mime_type,
    sanitize_filename,
    validate_magic_signature,
)
from app.logging_config import get_logger
from app.models.document import DocumentChunk, DocumentMetadata
from app.services.extraction import get_extractor
from app.services.extraction.base import ExtractedDocument
from app.services.processing.cleaner import TextCleaner
from app.services.processing.chunker import IntelligentChunker, chunker
from app.services.processing.structure import StructureDetector
from app.services.storage.database import Database, db

logger = get_logger(__name__)


class DocumentService:
    """Orchestrates document validation, ingestion, extraction, cleaning, and chunking."""

    def __init__(
        self,
        db_instance: Optional[Database] = None,
        chunker_instance: Optional[IntelligentChunker] = None,
    ):
        self.db = db_instance or db
        self.chunker = chunker_instance or chunker

    def validate_file(self, filename: str, content: bytes) -> Tuple[str, str]:
        """Validate file size, extension, and content magic bytes.
        
        Returns:
            Tuple of (extension, mime_type)
        """
        # 1. Size check
        if len(content) == 0:
            raise ValidationError("Uploaded file is empty (0 bytes).")
        if len(content) > settings.max_upload_size_bytes:
            max_mb = settings.MAX_UPLOAD_SIZE_MB
            raise FileSizeExceededError(f"File size exceeds maximum allowed limit of {max_mb} MB.")

        # 2. Extension check
        clean_name = sanitize_filename(filename)
        ext = Path(clean_name).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File extension '{ext}' is not supported. Supported extensions: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            )

        # 3. Magic bytes / header check
        is_valid = validate_magic_signature(content, ext)
        if not is_valid:
            raise FileCorruptedError(
                f"Invalid file signature or corrupted header: '{clean_name}' does not match expected format for {ext}."
            )

        mime_type = get_mime_type(ext)
        return ext, mime_type

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        document_id: Optional[str] = None
    ) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        """Ingest, validate, store raw file, extract, clean, and chunk a document."""
        # Validate
        ext, mime_type = self.validate_file(filename, content)
        doc_hash = compute_file_hash(content)

        # Check deduplication
        existing = self.db.get_document_by_hash(doc_hash)
        if existing and existing.status == STATUS_INDEXED:
            logger.info("Deduplication match: Document '%s' already indexed as '%s'", filename, existing.document_id)
            existing_chunks = self.db.get_chunks_for_document(existing.document_id)
            setattr(existing, "is_duplicate", True)
            if existing.extracted_entities is None:
                existing.extracted_entities = {}
            existing.extracted_entities["is_duplicate"] = True
            return existing, existing_chunks

        # Assign document ID and save file to uploads
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        clean_filename = sanitize_filename(filename)
        saved_path = settings.UPLOAD_DIR / f"{doc_id}_{clean_filename}"

        with open(saved_path, "wb") as f:
            f.write(content)

        # Create initial record
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=clean_filename,
            original_filename=filename,
            file_type=ext.lstrip("."),
            mime_type=mime_type,
            file_size=len(content),
            file_hash=doc_hash,
            status=STATUS_PROCESSING,
        )
        self.db.save_document(metadata)

        try:
            # Extract
            extractor = get_extractor(saved_path)
            extracted_doc: ExtractedDocument = extractor.extract(saved_path)

            # Clean pages
            cleaned_pages = TextCleaner.clean_pages(extracted_doc.pages)

            # Structure detection
            sections = StructureDetector.detect_sections(cleaned_pages)

            # Chunking
            chunks = self.chunker.chunk_document(
                document_id=doc_id,
                pages=cleaned_pages,
                sections=sections,
            )

            # Calculate detailed text metrics & statistics
            full_text = " ".join([p.text for p in cleaned_pages]).strip()
            words = full_text.split()
            word_count = len(words)
            char_count = len(full_text)
            line_count = len(full_text.splitlines()) if full_text else 0
            sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if s.strip()]) if full_text else 0
            reading_time = round(word_count / 200.0, 1)  # average 200 wpm
            avg_word_length = round(char_count / max(1, word_count), 1) if word_count > 0 else 0.0

            # Detect documents with little or no extractable text
            quality_warnings: List[str] = []
            if char_count == 0:
                quality_warnings.append("EMPTY_DOCUMENT: Document contains 0 extractable characters.")
                logger.warning("Document %s contains no extractable text across %d pages", doc_id, extracted_doc.total_pages)
            elif word_count < 10:
                quality_warnings.append("LOW_TEXT_DENSITY: Document contains fewer than 10 words.")
                logger.info("Document %s flagged for low text density (%d words)", doc_id, word_count)

            # Update metadata
            metadata.page_count = max(1, extracted_doc.total_pages)
            metadata.word_count = word_count
            metadata.char_count = char_count
            metadata.reading_time_minutes = reading_time
            metadata.title = extracted_doc.title or clean_filename
            metadata.author = extracted_doc.author
            metadata.status = STATUS_INDEXED
            metadata.extracted_entities = {
                "statistics": {
                    "word_count": word_count,
                    "char_count": char_count,
                    "line_count": line_count,
                    "sentence_count": sentence_count,
                    "avg_word_length": avg_word_length,
                },
                "quality_warnings": quality_warnings,
                "has_ocr": any(p.has_ocr for p in cleaned_pages),
            }

            # Persist chunks and metadata in database
            self.db.save_document(metadata)
            self.db.save_chunks(chunks)

            logger.info(
                "Document %s processed: %d pages, %d words, %d chunks",
                doc_id, metadata.page_count, word_count, len(chunks)
            )
            return metadata, chunks

        except Exception as e:
            logger.error("Processing failed for document %s: %s", doc_id, e)
            self.db.update_document_status(doc_id, STATUS_FAILED, error_message=str(e))
            metadata.status = STATUS_FAILED
            metadata.error_message = str(e)
            raise e


# Global document service instance
document_service = DocumentService()
