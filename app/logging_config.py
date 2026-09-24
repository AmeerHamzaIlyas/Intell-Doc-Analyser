"""Structured logging configuration for the document intelligence platform."""
import logging
import sys
from app.config import settings

def setup_logging(level: int = logging.INFO) -> None:
    """Configure system-wide structured logging format."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    
    # Root handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level if not settings.DEBUG else logging.DEBUG)
    
    # Avoid duplicate handlers
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
        try:
            from app.core.security import SecretMaskingFilter
            console_handler.addFilter(SecretMaskingFilter())
        except Exception:
            pass
        root_logger.addHandler(console_handler)
        
    # Reduce noisy logs from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
