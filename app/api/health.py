"""Health check endpoints for liveness and readiness probes."""
from datetime import datetime, timezone
from fastapi import APIRouter
from app.config import settings
from app.services.ai.gemini_client import gemini_client
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
def liveness():
    """Liveness probe to check if the server is running."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
def readiness():
    """Readiness probe to check if database, storage, and AI subsystems are ready."""
    checks = {
        "database": False,
        "storage_writable": False,
        "vector_store": False,
        "bm25_index": False,
        "gemini_api": False,
    }

    # 1. DB check
    try:
        db.count_documents()
        checks["database"] = True
    except Exception:
        pass

    # 2. Storage check
    try:
        test_file = settings.DATA_DIR / ".readiness_probe"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        checks["storage_writable"] = True
    except Exception:
        pass

    # 3. Vector & BM25 check
    checks["vector_store"] = vector_store.count() >= 0
    checks["bm25_index"] = bm25_index.count() >= 0

    # 4. Gemini API check
    checks["gemini_api"] = gemini_client.is_available()

    all_ready = checks["database"] and checks["storage_writable"]
    return {
        "ready": all_ready,
        "checks": checks,
        "stats": db.get_system_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
