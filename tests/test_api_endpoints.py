"""Integration tests for FastAPI REST API endpoints."""
import io
import shutil
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from app.config import settings
from app.main import app
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store
from app.services.storage.bm25_index import bm25_index


@pytest.fixture(scope="module")
def client():
    """Create test client with isolated storage environment."""
    temp_dir = Path(tempfile.mkdtemp())
    orig_data = settings.DATA_DIR
    orig_upload = settings.UPLOAD_DIR
    orig_storage = settings.STORAGE_DIR
    orig_db = settings.DB_PATH
    orig_vector = settings.VECTOR_DIR

    settings.DATA_DIR = temp_dir
    settings.UPLOAD_DIR = temp_dir / "uploads"
    settings.STORAGE_DIR = temp_dir / "storage"
    settings.DB_PATH = temp_dir / "test_api.db"
    settings.VECTOR_DIR = temp_dir / "test_vector"
    settings.ensure_directories()

    db.db_path = settings.DB_PATH
    db.init_db()
    vector_store.storage_dir = settings.VECTOR_DIR
    vector_store.clear()
    bm25_index.storage_dir = settings.VECTOR_DIR
    bm25_index.clear()

    with TestClient(app) as test_client:
        yield test_client

    shutil.rmtree(temp_dir, ignore_errors=True)
    settings.DATA_DIR = orig_data
    settings.UPLOAD_DIR = orig_upload
    settings.STORAGE_DIR = orig_storage
    settings.DB_PATH = orig_db
    settings.VECTOR_DIR = orig_vector
    settings.ensure_directories()


def test_health_endpoints(client: TestClient):
    """Test liveness and readiness endpoints."""
    live_res = client.get("/health/live")
    assert live_res.status_code == 200
    assert live_res.json()["status"] == "healthy"

    ready_res = client.get("/health/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["ready"] is True
    assert ready_res.json()["checks"]["database"] is True


def test_frontend_and_static_serving(client: TestClient):
    """Verify single-page frontend application dashboard and static assets are served."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "text/html" in root_res.headers.get("content-type", "")
    assert "DocPulse" in root_res.text

    css_res = client.get("/static/css/styles.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers.get("content-type", "")

    js_res = client.get("/static/js/app.js")
    assert js_res.status_code == 200



def test_document_lifecycle_api(client: TestClient):
    """Test full document lifecycle: upload, get, list, search, rag, summarize, compare, delete."""
    # 1. Upload Doc A
    doc_a_content = (
        b"# AI Infrastructure Report 2026\n\n"
        b"Enterprise adoption of generative AI increased by 40% in 2026.\n\n"
        b"## Key Drivers\n"
        b"Primary adoption drivers include automated code generation and document analysis.\n\n"
        b"## Recommendations\n"
        b"Organizations must invest in data hygiene and retrieval-augmented generation architectures."
    )
    upload_res_a = client.post(
        "/api/documents/upload",
        files={"file": ("ai_infra_report.md", io.BytesIO(doc_a_content), "text/markdown")},
    )
    assert upload_res_a.status_code == 200
    doc_a = upload_res_a.json()
    doc_id_a = doc_a["metadata"]["document_id"]
    assert doc_id_a is not None
    assert doc_a["chunk_count"] >= 1

    # 2. Upload Doc B (modified version for comparison)
    doc_b_content = (
        b"# AI Infrastructure Report 2026 - Revision 2\n\n"
        b"Enterprise adoption of generative AI increased by 65% in 2026.\n\n"
        b"## Key Drivers\n"
        b"Adoption drivers include code generation, multimodal analysis, and real-time streaming.\n\n"
        b"## Recommendations\n"
        b"Organizations must deploy zero-trust security and hybrid search pipelines."
    )
    upload_res_b = client.post(
        "/api/documents/upload",
        files={"file": ("ai_infra_report_v2.md", io.BytesIO(doc_b_content), "text/markdown")},
    )
    assert upload_res_b.status_code == 200
    doc_b = upload_res_b.json()
    doc_id_b = doc_b["metadata"]["document_id"]

    # 3. List Documents
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 2

    # 4. Get Single Document Detail & Chunks
    get_res = client.get(f"/api/documents/{doc_id_a}")
    assert get_res.status_code == 200
    assert get_res.json()["metadata"]["filename"] == "ai_infra_report.md"

    chunks_res = client.get(f"/api/documents/{doc_id_a}/chunks")
    assert chunks_res.status_code == 200
    assert len(chunks_res.json()) >= 1

    # 5. System Stats
    stats_res = client.get("/api/documents/stats/overview")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_documents"] >= 2

    # 6. Search API
    search_res = client.post(
        "/api/search",
        json={"query": "generative AI adoption", "top_k": 3, "mode": "hybrid"},
    )
    assert search_res.status_code == 200
    assert search_res.json()["total_results"] > 0
    assert "latency_ms" in search_res.json()

    # 7. RAG Q&A API
    rag_res = client.post(
        "/api/rag/query",
        json={"query": "What percentage did enterprise adoption increase in 2026?", "top_k": 3},
    )
    assert rag_res.status_code == 200
    rag_json = rag_res.json()
    assert len(rag_json["answer"]) > 0
    assert len(rag_json["referenced_chunks"]) > 0
    assert rag_json["grounding_status"] in ("GROUNDED", "PARTIAL")

    # 8. Summarize API
    summary_res = client.post(f"/api/summarize/{doc_id_a}")
    assert summary_res.status_code == 200
    summary_json = summary_res.json()
    assert len(summary_json["executive_summary"]) > 0

    # 9. Compare API
    compare_res = client.post(
        "/api/compare",
        json={"doc_id_a": doc_id_a, "doc_id_b": doc_id_b},
    )
    assert compare_res.status_code == 200
    comp_json = compare_res.json()
    assert comp_json["structural_diff"]["similarity_percentage"] > 0
    assert "overall_verdict" in comp_json["semantic_analysis"]

    # 10. Delete Document
    del_res = client.delete(f"/api/documents/{doc_id_a}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Verify 404 after deletion
    not_found = client.get(f"/api/documents/{doc_id_a}")
    assert not_found.status_code == 404
