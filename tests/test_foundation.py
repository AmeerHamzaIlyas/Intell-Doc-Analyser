"""Unit tests for foundation layer: configuration, security, exceptions, and models."""
import pytest
from pathlib import Path
from app.config import settings
from app.core.constants import SUPPORTED_EXTENSIONS, STATUS_QUEUED
from app.core.exceptions import ValidationError
from app.core.security import sanitize_filename, validate_path_containment, compute_file_hash
from app.models.document import DocumentMetadata, DocumentChunk
from app.models.search_rag import SearchRequest, RAGRequest


def test_settings_initialization():
    """Verify settings loads defaults and creates required directories."""
    assert settings.APP_NAME == "Intelligent Document Understanding & Analysis System"
    assert settings.max_upload_size_bytes == 50 * 1024 * 1024
    assert ".pdf" in settings.ALLOWED_EXTENSIONS
    assert settings.DATA_DIR.exists()
    assert settings.UPLOAD_DIR.exists()
    assert settings.STORAGE_DIR.exists()
    assert settings.VECTOR_DIR.exists()


def test_sanitize_filename():
    """Verify filename sanitization against path traversal and special characters."""
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert sanitize_filename("My Report (Final) 2026!.docx") == "My_Report__Final__2026.docx"
    assert sanitize_filename("   spaced_name.txt   ") == "spaced_name.txt"
    assert sanitize_filename("") == "document"
    assert sanitize_filename("...") == "document"


def test_path_containment():
    """Verify path containment prevents directory escapes."""
    base = settings.UPLOAD_DIR
    valid_child = base / "safe_file.pdf"
    assert validate_path_containment(valid_child, base) == valid_child.resolve()
    
    with pytest.raises(ValidationError):
        validate_path_containment(base / ".." / "secret.txt", base)


def test_compute_file_hash():
    """Verify SHA-256 hash calculation."""
    data = b"Hello Document Intelligence!"
    import hashlib
    expected = hashlib.sha256(data).hexdigest()
    assert compute_file_hash(data) == expected


def test_document_models():
    """Verify Document metadata and chunk models instantiate and validate."""
    doc = DocumentMetadata(
        document_id="doc-123",
        filename="test.pdf",
        original_filename="original_test.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        file_size=1024,
        file_hash="hash123",
        page_count=2,
        word_count=50,
        char_count=300,
    )
    assert doc.status == STATUS_QUEUED
    assert doc.page_count == 2
    
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-123",
        chunk_index=0,
        text="Sample extracted text chunk.",
        page_number=1,
        section_title="Introduction",
        token_count=5,
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.section_title == "Introduction"


def test_search_and_rag_models():
    """Verify search and RAG request schemas."""
    search_req = SearchRequest(query="machine learning", top_k=10, mode="hybrid")
    assert search_req.query == "machine learning"
    assert search_req.top_k == 10
    
    rag_req = RAGRequest(query="What is the total revenue?", top_k=3)
    assert rag_req.query == "What is the total revenue?"
    assert rag_req.top_k == 3


def test_settings_validation_failures():
    """Verify that invalid settings trigger explicit Pydantic validation errors."""
    from pydantic import ValidationError as PydanticValidationError
    from app.config import Settings

    # Invalid max upload size
    with pytest.raises(PydanticValidationError):
        Settings(MAX_UPLOAD_SIZE_MB=-1)

    with pytest.raises(PydanticValidationError):
        Settings(MAX_UPLOAD_SIZE_MB=1000)

    # Invalid port
    with pytest.raises(PydanticValidationError):
        Settings(PORT=0)

    with pytest.raises(PydanticValidationError):
        Settings(PORT=99999)

    # Invalid chunk size
    with pytest.raises(PydanticValidationError):
        Settings(DEFAULT_CHUNK_SIZE=10)

    # Chunk overlap >= chunk size
    with pytest.raises(PydanticValidationError):
        Settings(DEFAULT_CHUNK_SIZE=500, DEFAULT_CHUNK_OVERLAP=500)

    with pytest.raises(PydanticValidationError):
        Settings(DEFAULT_CHUNK_SIZE=500, DEFAULT_CHUNK_OVERLAP=-5)

    # Invalid RRF_K
    with pytest.raises(PydanticValidationError):
        Settings(RRF_K=0)


def test_magic_signature_validation():
    """Verify file magic byte signature verification and anti-spoofing."""
    from app.core.security import validate_magic_signature
    from app.core.exceptions import FileCorruptedError, UnsupportedFileTypeError

    # Valid signatures
    assert validate_magic_signature(b"%PDF-1.7 sample content", ".pdf") is True
    assert validate_magic_signature(b"PK\x03\x04\x14\x00", ".docx") is True
    assert validate_magic_signature(b"\x89PNG\r\n\x1a\n\x00\x00", "png") is True
    assert validate_magic_signature(b"\xff\xd8\xff\xe0", ".jpg") is True
    assert validate_magic_signature(b"BM\x36\x00", ".bmp") is True
    assert validate_magic_signature(b"Plain readable text file without nulls", ".txt") is True

    # Spoofed files
    assert validate_magic_signature(b"This is just plain text masquerading as a PDF", ".pdf") is False
    assert validate_magic_signature(b"Fake docx file", ".docx") is False
    assert validate_magic_signature(b"Corrupted text\x00with null bytes", ".txt") is False

    # Failure cases
    with pytest.raises(FileCorruptedError):
        validate_magic_signature(b"", ".pdf")

    with pytest.raises(UnsupportedFileTypeError):
        validate_magic_signature(b"some bytes", ".exe")


def test_mime_type_resolution():
    """Verify extension to MIME type mappings."""
    from app.core.security import get_mime_type

    assert get_mime_type(".pdf") == "application/pdf"
    assert get_mime_type("report.docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert get_mime_type("notes.txt") == "text/plain"
    assert get_mime_type("README.md") == "text/markdown"
    assert get_mime_type(".png") == "image/png"
    assert get_mime_type("unknown.xyz") == "application/octet-stream"


def test_logging_configuration():
    """Verify logging setup and named logger creation."""
    import logging
    from app.logging_config import setup_logging, get_logger

    setup_logging(level=logging.DEBUG)
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)


def test_exception_hierarchy():
    """Verify custom exception hierarchy and detail retention."""
    from app.core.exceptions import (
        DocumentIntelligenceException,
        ValidationError,
        UnsupportedFileTypeError,
        DocumentNotFoundError,
        StorageError,
    )

    err = DocumentIntelligenceException("System fault", details={"code": 500})
    assert str(err) == "System fault"
    assert err.details["code"] == 500

    val_err = UnsupportedFileTypeError("Bad type")
    assert isinstance(val_err, ValidationError)
    assert isinstance(val_err, DocumentIntelligenceException)

    not_found = DocumentNotFoundError("Document missing")
    assert isinstance(not_found, StorageError)
    assert isinstance(not_found, DocumentIntelligenceException)

