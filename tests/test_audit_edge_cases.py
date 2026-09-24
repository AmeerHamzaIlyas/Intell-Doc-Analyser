"""Edge-case audit test suite covering boundary conditions, unusual data, and structural variations."""
import io
import shutil
import tempfile
from pathlib import Path
import docx
from fastapi.testclient import TestClient
import pypdf
import pytest
from app.config import settings
from app.main import app
from app.services.processing.chunker import IntelligentChunker
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store


@pytest.fixture(scope="module")
def client():
    """Create isolated test environment for edge-case audit tests."""
    temp_dir = Path(tempfile.mkdtemp())
    orig_data = settings.DATA_DIR
    orig_upload = settings.UPLOAD_DIR
    orig_storage = settings.STORAGE_DIR
    orig_db = settings.DB_PATH
    orig_vector = settings.VECTOR_DIR

    settings.DATA_DIR = temp_dir
    settings.UPLOAD_DIR = temp_dir / "uploads"
    settings.STORAGE_DIR = temp_dir / "storage"
    settings.DB_PATH = temp_dir / "edge_test.db"
    settings.VECTOR_DIR = temp_dir / "edge_vector"
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


def test_edge_empty_documents(client: TestClient):
    """Test 0-byte files, whitespace-only files, and empty pages."""
    # 1. Zero-byte upload is rejected with 400 ValidationError
    res_zero = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    )
    assert res_zero.status_code == 400
    assert "empty" in res_zero.text.lower()

    # 2. Whitespace-only text file ingests cleanly with empty document quality warning
    whitespace_bytes = b"   \n\t  \r\n   "
    res_ws = client.post(
        "/api/documents/upload",
        files={"file": ("whitespace_only.txt", io.BytesIO(whitespace_bytes), "text/plain")}
    )
    assert res_ws.status_code == 200
    meta_ws = res_ws.json()
    assert meta_ws["chunk_count"] == 0
    warnings = meta_ws["metadata"]["extracted_entities"].get("quality_warnings", [])
    assert any("EMPTY_DOCUMENT" in w for w in warnings)

    # 3. Summarizing an empty document returns a clean summary without crashing
    doc_id = meta_ws["metadata"]["document_id"]
    sum_res = client.post(f"/api/summarize/{doc_id}")
    assert sum_res.status_code == 200
    assert "no extractable text" in sum_res.json()["executive_summary"].lower()


def test_edge_corrupted_files(client: TestClient):
    """Test corrupted headers and damaged container structures return 422."""
    # 1. Truncated PDF header with broken byte payload
    corrupted_pdf = b"%PDF-1.5 broken payload without EOF marker"
    res_pdf = client.post(
        "/api/documents/upload",
        files={"file": ("corrupt.pdf", io.BytesIO(corrupted_pdf), "application/pdf")}
    )
    assert res_pdf.status_code == 422
    assert "corrupted" in res_pdf.text.lower() or "failed to read" in res_pdf.text.lower()

    # 2. Corrupted DOCX (magic bytes PK\x03\x04 but broken zip structure)
    corrupted_docx = b"PK\x03\x04\x00\x00corrupted_zip_bytes"
    res_docx = client.post(
        "/api/documents/upload",
        files={"file": ("corrupt.docx", io.BytesIO(corrupted_docx), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert res_docx.status_code == 422
    assert "corrupted" in res_docx.text.lower() or "malformed" in res_docx.text.lower()


def test_edge_unsupported_formats(client: TestClient):
    """Test unsupported extensions are rejected with 415."""
    unsupported_payloads = [
        ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00"),
        ("script.py", b"import os; os.system('echo hi')"),
        ("data.bin", b"\x00\x01\x02\x03\x04\x05"),
        ("audio.mp3", b"ID3\x03\x00\x00\x00"),
    ]
    for filename, content in unsupported_payloads:
        res = client.post(
            "/api/documents/upload",
            files={"file": (filename, io.BytesIO(content), "application/octet-stream")}
        )
        assert res.status_code == 415
        assert "not supported" in res.text.lower()


def test_edge_unicode_complex_scripts(client: TestClient):
    """Test non-Latin scripts (Japanese, Arabic, Cyrillic), emojis, and zero-width characters."""
    unicode_content = (
        "# ドキュメント分析システム\n\n"
        "AI技術を活用した次世代のインテリジェント文書解析プラットフォーム。\n\n"
        "## العربية الفصحى\n"
        "نظام ذكي متقدم لتحليل وفهم الوثائق والملفات الرسمية بدقة عالية.\n\n"
        "## Русский Текст\n"
        "Платформа интеллектуального анализа и поиска по корпоративным документам.\n\n"
        "## Emojis & Symbols 🚀🔥\n"
        "Performance boost: +50% 📈. Zero-width test: clean\u200b\u200c\u200dtext."
    )
    res = client.post(
        "/api/documents/upload",
        files={"file": ("unicode_suite.md", io.BytesIO(unicode_content.encode("utf-8")), "text/markdown")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["chunk_count"] >= 1

    # Search for Japanese keyword
    search_jp = client.post(
        "/api/search",
        json={"query": "次世代のインテリジェント", "top_k": 2}
    )
    assert search_jp.status_code == 200
    assert search_jp.json()["total_results"] >= 1

    # Search for Arabic keyword
    search_ar = client.post(
        "/api/search",
        json={"query": "الوثائق", "top_k": 2}
    )
    assert search_ar.status_code == 200
    assert search_ar.json()["total_results"] >= 1


def test_edge_tables_extraction(client: TestClient):
    """Test structured table parsing in Word DOCX."""
    doc = docx.Document()
    doc.add_heading("Financial Summary Q4", level=1)
    doc.add_paragraph("Key financial highlights across product departments:")

    table = doc.add_table(rows=3, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Quarter"
    hdr_cells[1].text = "Revenue"
    hdr_cells[2].text = "Operating Margin"

    row1 = table.rows[1].cells
    row1[0].text = "Q3 2026"
    row1[1].text = "$12.4M"
    row1[2].text = "28.5%"

    row2 = table.rows[2].cells
    row2[0].text = "Q4 2026"
    row2[1].text = "$16.8M"
    row2[2].text = "34.2%"

    buf = io.BytesIO()
    doc.save(buf)

    res = client.post(
        "/api/documents/upload",
        files={"file": ("financial_summary.docx", io.BytesIO(buf.getvalue()), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["chunk_count"] >= 1

    # Retrieve chunks and verify Markdown table representation
    doc_id = data["metadata"]["document_id"]
    chunks_res = client.get(f"/api/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    table_chunk = [c for c in chunks if "|" in c["text"] and "---" in c["text"]]
    assert len(table_chunk) >= 1
    assert "$16.8M" in table_chunk[0]["text"]


def test_edge_long_unpunctuated_paragraph_chunking():
    """Verify recursive sub-chunker splits continuous text without punctuation or spaces."""
    chunker = IntelligentChunker(chunk_size=400, chunk_overlap=50)

    # 1. 2000-word unpunctuated text (no periods, commas, or exclamation marks)
    long_unpunctuated = "word " * 1000  # 5,000 characters
    from app.services.extraction.base import ExtractedPage
    page = ExtractedPage(page_number=1, text=long_unpunctuated)

    chunks = chunker.chunk_document("test_doc_unpunctuated", [page])
    assert len(chunks) > 1
    # Check that ALL generated chunks strictly respect chunk_size
    for c in chunks:
        assert len(c.text) <= 450  # Allows minor overlap margin, strictly not 5000

    # 2. Continuous unbroken token with 0 spaces
    unbroken_token = "A" * 1500
    page_unbroken = ExtractedPage(page_number=1, text=unbroken_token)
    unbroken_chunks = chunker.chunk_document("test_doc_unbroken", [page_unbroken])
    assert len(unbroken_chunks) >= 3
    for c in unbroken_chunks:
        assert len(c.text) <= 450


def test_edge_multiple_pages_pagination(client: TestClient):
    """Test 10-page document extraction with provenance tracking across pages."""
    writer = pypdf.PdfWriter()
    for page_idx in range(1, 11):
        writer.add_blank_page(width=612, height=792)

    buf = io.BytesIO()
    writer.write(buf)

    res = client.post(
        "/api/documents/upload",
        files={"file": ("ten_pages.pdf", io.BytesIO(buf.getvalue()), "application/pdf")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["page_count"] == 10


def test_edge_duplicate_documents_repeated_uploads(client: TestClient):
    """Test that uploading the same document multiple times does not duplicate vectors or BM25 index."""
    content = (
        b"# Duplicate Resilience Contract\n\n"
        b"This contract tests that duplicate uploads do not cause index pollution or vector multiplication.\n\n"
        b"Unique Token: RESILIENCE_TOKEN_2026."
    )
    # First upload
    res1 = client.post(
        "/api/documents/upload",
        files={"file": ("resilience_contract.md", io.BytesIO(content), "text/markdown")}
    )
    assert res1.status_code == 200
    data1 = res1.json()
    doc_id1 = data1["metadata"]["document_id"]
    initial_vectors = vector_store.count()
    initial_bm25 = bm25_index.count()

    # Second upload (same content, different filename)
    res2 = client.post(
        "/api/documents/upload",
        files={"file": ("resilience_contract_dup.md", io.BytesIO(content), "text/markdown")}
    )
    assert res2.status_code == 200
    data2 = res2.json()

    # Must return the existing document ID and mark as duplicate
    assert data2["metadata"]["document_id"] == doc_id1
    assert data2["is_duplicate"] is True

    # Vector store and BM25 index must NOT have grown
    assert vector_store.count() == initial_vectors
    assert bm25_index.count() == initial_bm25


def test_edge_questions_empty_and_extreme_length(client: TestClient):
    """Verify input validation for empty and excessively long questions."""
    # 1. Empty query search -> 400
    res_empty_search = client.post("/api/search", json={"query": "   ", "top_k": 3})
    assert res_empty_search.status_code == 400

    # 2. Empty query RAG -> 400
    res_empty_rag = client.post("/api/rag/query", json={"query": "   "})
    assert res_empty_rag.status_code == 400

    # 3. Excessive query length (> 4000 characters) -> 422 Unprocessable Entity
    oversized_query = "What is " + ("very long question " * 300)  # > 5500 characters
    res_oversized = client.post("/api/search", json={"query": oversized_query})
    assert res_oversized.status_code == 422


def test_edge_invalid_api_requests(client: TestClient):
    """Test error handling on invalid IDs, self-comparison, and out-of-range parameters."""
    # 1. Non-existent document detail -> 404
    res_not_found = client.get("/api/documents/doc_does_not_exist_9999")
    assert res_not_found.status_code == 404

    # 2. Compare document with itself -> 400
    res_self_compare = client.post(
        "/api/compare",
        json={"doc_id_a": "doc_123", "doc_id_b": "doc_123"}
    )
    assert res_self_compare.status_code == 400
    assert "itself" in res_self_compare.text.lower()

    # 3. Compare non-existent document -> 404
    res_cmp_404 = client.post(
        "/api/compare",
        json={"doc_id_a": "nonexistent_a", "doc_id_b": "nonexistent_b"}
    )
    assert res_cmp_404.status_code == 404

    # 4. Search with invalid top_k (< 1) -> 422
    res_invalid_k = client.post("/api/search", json={"query": "test", "top_k": 0})
    assert res_invalid_k.status_code == 422
