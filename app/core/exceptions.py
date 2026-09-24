"""Custom exceptions hierarchy for the document intelligence platform."""

class DocumentIntelligenceException(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(DocumentIntelligenceException):
    """Raised when file or payload fails validation checks."""
    pass


class UnsupportedFileTypeError(ValidationError):
    """Raised when file extension or magic bytes are unsupported."""
    pass


class FileSizeExceededError(ValidationError):
    """Raised when uploaded file exceeds maximum configured size."""
    pass


class FileCorruptedError(ValidationError):
    """Raised when file content is invalid or corrupted."""
    pass


class ExtractionError(DocumentIntelligenceException):
    """Raised when text/structure extraction fails."""
    pass


class OCRError(ExtractionError):
    """Raised when OCR processing fails."""
    pass


class ChunkingError(DocumentIntelligenceException):
    """Raised when document chunking fails."""
    pass


class EmbeddingError(DocumentIntelligenceException):
    """Raised when embedding generation fails."""
    pass


class StorageError(DocumentIntelligenceException):
    """Raised when database or file storage operations fail."""
    pass


class DocumentNotFoundError(StorageError):
    """Raised when requested document ID does not exist."""
    pass


class SearchError(DocumentIntelligenceException):
    """Raised when hybrid search fails."""
    pass


class RAGError(DocumentIntelligenceException):
    """Raised when question answering or prompt generation fails."""
    pass


class SummarizationError(DocumentIntelligenceException):
    """Raised when summarization fails."""
    pass


class ComparisonError(DocumentIntelligenceException):
    """Raised when document comparison fails."""
    pass
