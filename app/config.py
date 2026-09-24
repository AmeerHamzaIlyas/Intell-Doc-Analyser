"""Application configuration managed via Pydantic Settings."""
import os
from pathlib import Path
from typing import Set, List
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.constants import SUPPORTED_EXTENSIONS

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration parameters loaded from environment and defaults."""
    
    # Application identity
    APP_NAME: str = "Intelligent Document Understanding & Analysis System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Storage Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    STORAGE_DIR: Path = PROJECT_ROOT / "data" / "storage"
    DB_PATH: Path = PROJECT_ROOT / "data" / "app.db"
    VECTOR_DIR: Path = PROJECT_ROOT / "data" / "vector_index"
    
    # Ingestion & Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: Set[str] = SUPPORTED_EXTENSIONS
    
    # Chunking defaults
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 150
    
    # Gemini AI configuration
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 3072
    ENABLE_GEMINI_FALLBACK: bool = True
    
    # Hybrid Search parameters
    DEFAULT_TOP_K: int = 5
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    
    # Security & CORS
    CORS_ORIGINS: List[str] = ["*"]
    API_KEY_AUTH_ENABLED: bool = False
    API_SECRET_KEY: str = os.environ.get("APP_API_KEY", "")
    MAX_QUERY_LENGTH: int = 4000
    MAX_CONVERSATION_TURNS: int = 50

    @field_validator("MAX_UPLOAD_SIZE_MB")
    @classmethod
    def validate_max_upload_size(cls, v: int) -> int:
        if v <= 0 or v > 500:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be between 1 and 500 MB")
        return v

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("PORT must be between 1 and 65535")
        return v

    @field_validator("DEFAULT_CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 50 or v > 8000:
            raise ValueError("DEFAULT_CHUNK_SIZE must be between 50 and 8000 characters")
        return v

    @field_validator("RRF_K")
    @classmethod
    def validate_rrf_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("RRF_K constant must be at least 1")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.DEFAULT_CHUNK_OVERLAP < 0:
            raise ValueError("DEFAULT_CHUNK_OVERLAP cannot be negative")
        if self.DEFAULT_CHUNK_OVERLAP >= self.DEFAULT_CHUNK_SIZE:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be strictly less than DEFAULT_CHUNK_SIZE")
        return self
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MAX_UPLOAD_SIZE_MB to bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    def ensure_directories(self) -> None:
        """Ensure all required runtime data directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DIR.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()
