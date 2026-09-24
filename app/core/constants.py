"""Application constants and configuration constants."""
from typing import Set, Dict

# Supported file extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}

# Mapping of file extensions to MIME types
MIME_TYPE_MAP: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}

# Magic bytes (file signatures) for validation
MAGIC_SIGNATURES: Dict[str, bytes] = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",  # ZIP container header
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "bmp": b"BM",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
}

# Ingestion status enum values
STATUS_QUEUED = "QUEUED"
STATUS_PROCESSING = "PROCESSING"
STATUS_EXTRACTING = "EXTRACTING"
STATUS_CHUNKING = "CHUNKING"
STATUS_EMBEDDING = "EMBEDDING"
STATUS_INDEXED = "INDEXED"
STATUS_FAILED = "FAILED"
