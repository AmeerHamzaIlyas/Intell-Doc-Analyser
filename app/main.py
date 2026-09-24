"""FastAPI application entry point, lifecycle management, middleware, and route registration."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api import (
    compare_router,
    documents_router,
    health_router,
    rag_router,
    search_router,
    summarize_router,
)
from app.config import settings
from app.core.exceptions import (
    DocumentIntelligenceException,
    DocumentNotFoundError,
    FileCorruptedError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.logging_config import get_logger, setup_logging
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging()
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    
    # 1. Ensure storage directories
    settings.ensure_directories()

    # 2. Initialize Database
    db.init_db()

    # 3. Load Vector Store & BM25 indices
    vector_store.load()
    bm25_index.load()
    logger.info(
        "Application ready. Loaded %d vectors and %d BM25 indexed chunks.",
        vector_store.count(), bm25_index.count()
    )

    yield

    # Shutdown
    logger.info("Persisting indices and shutting down...")
    vector_store.save()
    bm25_index.save()


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade Document Intelligence, Multimodal Extraction, Hybrid Search, and Grounded RAG Platform.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Enforce API Key authentication on /api/* endpoints when API_KEY_AUTH_ENABLED is True."""
    if settings.API_KEY_AUTH_ENABLED and request.url.path.startswith("/api/"):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:].strip()
            elif auth_header:
                api_key = auth_header.strip()

        from app.core.security import verify_api_key
        if not verify_api_key(api_key):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or missing API key."}
            )

    return await call_next(request)


# Custom Exception Handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": "Validation Error", "detail": exc.message})


@app.exception_handler(FileSizeExceededError)
async def file_size_exception_handler(request: Request, exc: FileSizeExceededError):
    return JSONResponse(status_code=413, content={"error": "File Too Large", "detail": exc.message})


@app.exception_handler(UnsupportedFileTypeError)
async def file_type_exception_handler(request: Request, exc: UnsupportedFileTypeError):
    return JSONResponse(status_code=415, content={"error": "Unsupported Media Type", "detail": exc.message})


@app.exception_handler(FileCorruptedError)
async def file_corrupted_exception_handler(request: Request, exc: FileCorruptedError):
    return JSONResponse(status_code=422, content={"error": "Corrupted File", "detail": exc.message})


@app.exception_handler(DocumentNotFoundError)
async def not_found_exception_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(status_code=404, content={"error": "Not Found", "detail": exc.message})


@app.exception_handler(DocumentIntelligenceException)
async def domain_exception_handler(request: Request, exc: DocumentIntelligenceException):
    return JSONResponse(status_code=500, content={"error": "System Error", "detail": exc.message})


# Include Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(summarize_router)
app.include_router(compare_router)

# Mount Static Files
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve single-page frontend application dashboard."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "version": settings.APP_VERSION,
        "health": "/health/ready",
    }
