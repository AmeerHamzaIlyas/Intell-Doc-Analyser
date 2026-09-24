"""Security utilities for safe file handling, path traversal prevention, and hashing."""
from typing import Optional, Union
import hashlib
import logging
import os
import re
from pathlib import Path
from app.core.constants import MAGIC_SIGNATURES, MIME_TYPE_MAP, SUPPORTED_EXTENSIONS
from app.core.exceptions import (
    FileCorruptedError,
    UnsupportedFileTypeError,
    ValidationError,
)

# Allowed characters in sanitized filenames
SAFE_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|existing)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions?|rules?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer\s+mode|unrestricted|god\s+mode|dan\s+mode)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
    re.compile(r"(reveal|print|show|dump|output)\s+(all\s+|the\s+)?(system\s+prompt|secret\s+key|api\s+key|credentials|internal\s+prompts?|hidden\s+prompts?|initial\s+instructions?)", re.IGNORECASE),
    re.compile(r"ignore\s+document\s+boundaries", re.IGNORECASE),
    re.compile(r"execute\s+(the\s+)?instructions?\s+(contained\s+in|from|inside)\s+(the\s+)?(document|text)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|system|security)\s+(filters?|guidelines?|rules?)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"\<\/?(system|instruction|untrusted_document_context|context)[^\>]*\>", re.IGNORECASE),
]



def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and filesystem attacks.
    
    - Strips paths (POSIX and Windows), null bytes, control characters
    - Normalizes multiple dots/underscores
    - Defends against Windows reserved device names (CON, PRN, etc.)
    - Defaults to 'document' if name is empty or unsafe
    """
    if not filename:
        return "document"
    
    # Strip directory components for both forward and backward slashes
    clean_path = str(filename).replace("\\", "/").strip()
    name = clean_path.split("/")[-1].strip()
    
    # Remove null bytes and control characters
    name = name.replace("\x00", "")
    name = re.sub(r"[\x01-\x1f\x7f]", "", name)
    
    # Extract extension
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        base, ext = parts[0], parts[1].lower()
    else:
        base, ext = parts[0], ""
        
    # Sanitize base and extension
    clean_base = SAFE_CHARS_PATTERN.sub("_", base).strip("._-")
    clean_ext = SAFE_CHARS_PATTERN.sub("", ext).strip(".")
    
    if not clean_base:
        clean_base = "document"
        
    # Defend against Windows reserved device names
    if clean_base.upper() in WINDOWS_RESERVED_NAMES:
        clean_base = f"doc_{clean_base}"
        
    # Limit length safely preserving extension
    if len(clean_base) > 180:
        clean_base = clean_base[:180]
    if len(clean_ext) > 20:
        clean_ext = clean_ext[:20]
        
    final_name = f"{clean_base}.{clean_ext}" if clean_ext else clean_base
    return final_name


def detect_prompt_injection(text: str) -> bool:
    """Detect common prompt injection attacks in user queries or documents."""
    if not text:
        return False
    return any(p.search(text) is not None for p in PROMPT_INJECTION_PATTERNS)


def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify API key against configured application secrets if auth is enabled."""
    from app.config import settings
    if not settings.API_KEY_AUTH_ENABLED:
        return True
    if not settings.API_SECRET_KEY:
        return True
    import hmac
    return hmac.compare_digest(api_key or "", settings.API_SECRET_KEY)


GENERIC_API_KEY_REGEX = re.compile(r"AIza[0-9A-Za-z\-_]{30,45}", re.IGNORECASE)
BEARER_TOKEN_REGEX = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
SYSTEM_PROMPT_LEAK_PATTERNS = [
    re.compile(r"You are a rigorous, production-grade AI Document Intelligence Analyst", re.IGNORECASE),
    re.compile(r"RULES:\s*1\.\s*Answer ONLY using the factual context", re.IGNORECASE),
    re.compile(r"SECURITY GUARDRAIL:\s*Treat all context sources", re.IGNORECASE),
]


def sanitize_untrusted_context(text: str) -> str:
    """Sanitize and neutralize adversarial delimiters inside untrusted document chunks.
    
    - Escapes closing XML tags to prevent context breakout
    - Neutralizes fake system override headers
    """
    if not text:
        return ""
    # Neutralize closing context tags
    cleaned = re.sub(r"<\s*/\s*untrusted_document_context\s*>", "[TAG_ESCAPED]", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/\s*context\s*>", "[TAG_ESCAPED]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/\s*system\s*>", "[TAG_ESCAPED]", cleaned, flags=re.IGNORECASE)
    # Neutralize fake system headers
    cleaned = re.sub(r"(===+|---+)\s*(SYSTEM|INSTRUCTION|OVERRIDE)[^\n]*", "[USER_TEXT_BLOCK]", cleaned, flags=re.IGNORECASE)
    return cleaned


def sanitize_ai_output(text: str) -> str:
    """Guardrail to sanitize LLM and extractive fallback outputs before returning to user.
    
    - Redacts exposed API keys and Bearer tokens
    - Detects and neutralizes system prompt leakage
    - Redacts configured system secrets
    """
    if not text:
        return ""
    
    # 1. Check for system instruction leakage
    for p in SYSTEM_PROMPT_LEAK_PATTERNS:
        if p.search(text):
            return "I cannot fulfill this request as it involves outputting internal system instructions or configuration."
    
    # 2. Redact Google / Gemini API keys
    sanitized = GENERIC_API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)
    
    # 3. Redact Bearer tokens
    sanitized = BEARER_TOKEN_REGEX.sub("Bearer [REDACTED_TOKEN]", sanitized)
    
    # 4. Redact configured active API keys if present
    try:
        from app.config import settings
        if settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 6:
            sanitized = sanitized.replace(settings.GEMINI_API_KEY, "[REDACTED_API_KEY]")
        if settings.API_SECRET_KEY and len(settings.API_SECRET_KEY) > 4:
            sanitized = sanitized.replace(settings.API_SECRET_KEY, "[REDACTED_SECRET]")
    except Exception:
        pass
        
    return sanitized


class SecretMaskingFilter(logging.Filter):
    """Logging filter to automatically redact API keys and sensitive tokens from log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = GENERIC_API_KEY_REGEX.sub("[REDACTED_API_KEY]", record.msg)
                record.msg = BEARER_TOKEN_REGEX.sub("Bearer [REDACTED_TOKEN]", record.msg)
                from app.config import settings
                if settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 6:
                    record.msg = record.msg.replace(settings.GEMINI_API_KEY, "[REDACTED_API_KEY]")
                if settings.API_SECRET_KEY and len(settings.API_SECRET_KEY) > 4:
                    record.msg = record.msg.replace(settings.API_SECRET_KEY, "[REDACTED_SECRET]")
        except Exception:
            pass
        return True



def validate_path_containment(child_path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """Ensure that child_path resolves strictly within base_dir to prevent path traversal."""
    resolved_base = Path(base_dir).resolve()
    resolved_child = Path(child_path).resolve()
    
    try:
        resolved_child.relative_to(resolved_base)
    except ValueError:
        raise ValidationError(f"Path traversal detected: {child_path} is outside {base_dir}")
        
    return resolved_child


def compute_file_hash(file_bytes_or_path: Union[bytes, str, Path]) -> str:
    """Compute SHA-256 hash of raw bytes or file on disk."""
    hasher = hashlib.sha256()
    if isinstance(file_bytes_or_path, bytes):
        hasher.update(file_bytes_or_path)
    else:
        with open(file_bytes_or_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def validate_magic_signature(content: bytes, ext: str) -> bool:
    """Validate that file content bytes match the expected signature for its extension.
    
    Protects against file spoofing (e.g. executable renamed to .pdf).
    """
    if not content:
        raise FileCorruptedError("File is empty (0 bytes).")
        
    ext = ext.lower().strip()
    if not ext.startswith("."):
        ext = f".{ext}"
        
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file extension: {ext}")
        
    if ext == ".pdf":
        return content.startswith(MAGIC_SIGNATURES["pdf"])
    elif ext == ".docx":
        return content.startswith(MAGIC_SIGNATURES["docx"])
    elif ext == ".png":
        return content.startswith(MAGIC_SIGNATURES["png"])
    elif ext in (".jpg", ".jpeg"):
        return content.startswith(MAGIC_SIGNATURES["jpg"])
    elif ext == ".bmp":
        return content.startswith(MAGIC_SIGNATURES["bmp"])
    elif ext == ".tiff":
        return content.startswith(MAGIC_SIGNATURES["tiff_le"]) or content.startswith(MAGIC_SIGNATURES["tiff_be"])
    elif ext in (".txt", ".md"):
        # 1. UTF-16 encoded text with BOM
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                decoded = content[:1024].decode("utf-16")
                return "\x00" not in decoded
            except UnicodeDecodeError:
                return False
        # 2. UTF-8 encoded text with BOM
        if content.startswith(b"\xef\xbb\xbf"):
            try:
                decoded = content[3:1024].decode("utf-8")
                return "\x00" not in decoded
            except UnicodeDecodeError:
                return False
        # 3. Standard text without BOM: must not contain raw binary null bytes
        if b"\x00" in content[:1024]:
            return False
        try:
            content[:1024].decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                content[:1024].decode("latin-1")
                return True
            except UnicodeDecodeError:
                return False
        return False


def get_mime_type(ext_or_filename: str) -> str:
    """Get standardized MIME type from file extension or filename."""
    ext = ext_or_filename.lower().strip()
    if "." in ext:
        ext = f".{ext.rsplit('.', 1)[1]}"
    elif not ext.startswith("."):
        ext = f".{ext}"
    return MIME_TYPE_MAP.get(ext, "application/octet-stream")

