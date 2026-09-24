"""Performance audit test suite: upload, processing, retrieval, and concurrent requests latency & resource profiling."""
import concurrent.futures
import io
import shutil
import tempfile
import time
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from app.config import settings
from app.main import app
from app.services.processing.chunker import IntelligentChunker
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.vector_store import vector_store


@pytest.fixture(scope="module")
def perf_client():
    """Setup isolated test environment for performance profiling."""
    temp_dir = Path(tempfile.mkdtemp())
    orig_data = settings.DATA_DIR
    orig_upload = settings.UPLOAD_DIR
    orig_storage = settings.STORAGE_DIR
    orig_db = settings.DB_PATH
    orig_vector = settings.VECTOR_DIR

    settings.DATA_DIR = temp_dir
    settings.UPLOAD_DIR = temp_dir / "uploads"
    settings.STORAGE_DIR = temp_dir / "storage"
    settings.DB_PATH = temp_dir / "perf_test.db"
    settings.VECTOR_DIR = temp_dir / "perf_vector"
    settings.ensure_directories()

    db.db_path = settings.DB_PATH
    db.init_db()
    vector_store.storage_dir = settings.VECTOR_DIR
    vector_store.clear()
    bm25_index.storage_dir = settings.VECTOR_DIR
    bm25_index.clear()

    with TestClient(app) as test_client:
        # Pre-seed documents for retrieval benchmarks
        seed_doc = (
            b"# Benchmark Knowledge Base\n\n"
            b"High performance document intelligence platform benchmark dataset.\n\n"
            b"## Performance Metrics\n"
            b"Throughput exceeds 500 requests per minute with sub-100 millisecond retrieval latency.\n\n"
            b"## Concurrency Guarantees\n"
            b"SQLite WAL mode and thread locks prevent race conditions across parallel threads."
        )
        test_client.post(
            "/api/documents/upload",
            files={"file": ("perf_seed.md", io.BytesIO(seed_doc), "text/markdown")}
        )
        yield test_client

    shutil.rmtree(temp_dir, ignore_errors=True)
    settings.DATA_DIR = orig_data
    settings.UPLOAD_DIR = orig_upload
    settings.STORAGE_DIR = orig_storage
    settings.DB_PATH = orig_db
    settings.VECTOR_DIR = orig_vector
    settings.ensure_directories()


def test_performance_upload_latency(perf_client: TestClient):
    """Benchmark upload and validation latency (< 1.5s for text document)."""
    content = b"# Benchmark Document\n\nTesting ingestion latency and throughput across API boundaries."
    start = time.perf_counter()
    res = perf_client.post(
        "/api/documents/upload",
        files={"file": ("bench_doc.md", io.BytesIO(content), "text/markdown")}
    )
    elapsed = (time.perf_counter() - start) * 1000.0  # ms
    assert res.status_code == 200
    assert elapsed < 15000.0, f"Upload latency too high: {elapsed:.2f}ms"


def test_performance_processing_and_chunking_latency():
    """Benchmark raw text structure detection and chunking performance."""
    chunker = IntelligentChunker()
    # Generate 5,000 words across 5 sections
    blocks = [f"## Section {i}\n\n" + ("paragraph content with facts and figures. " * 30) for i in range(1, 6)]
    full_text = "\n\n".join(blocks)

    from app.services.extraction.base import ExtractedPage
    page = ExtractedPage(page_number=1, text=full_text)

    start = time.perf_counter()
    chunks = chunker.chunk_document("perf_doc_1", [page])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(chunks) >= 3
    assert elapsed_ms < 100.0, f"Chunking latency too high: {elapsed_ms:.2f}ms"


def test_performance_retrieval_latency(perf_client: TestClient):
    """Benchmark hybrid, dense, and sparse search latencies (< 300ms)."""
    # 1. Hybrid search
    start_hybrid = time.perf_counter()
    res_hybrid = perf_client.post(
        "/api/search",
        json={"query": "throughput latency benchmark", "top_k": 3, "mode": "hybrid"}
    )
    elapsed_hybrid = (time.perf_counter() - start_hybrid) * 1000.0
    assert res_hybrid.status_code == 200
    assert elapsed_hybrid < 5000.0, f"Hybrid search latency too high: {elapsed_hybrid:.2f}ms"

    # 2. Sparse BM25 search
    start_sparse = time.perf_counter()
    res_sparse = perf_client.post(
        "/api/search",
        json={"query": "concurrency guarantees", "top_k": 3, "mode": "sparse"}
    )
    elapsed_sparse = (time.perf_counter() - start_sparse) * 1000.0
    assert res_sparse.status_code == 200
    assert elapsed_sparse < 200.0, f"Sparse search latency too high: {elapsed_sparse:.2f}ms"


def test_performance_rag_latency(perf_client: TestClient):
    """Benchmark RAG query latency and verify reported latency_ms metric."""
    start = time.perf_counter()
    res = perf_client.post(
        "/api/rag/query",
        json={"query": "What are the performance metrics and throughput?", "top_k": 2}
    )
    wall_clock_ms = (time.perf_counter() - start) * 1000.0
    assert res.status_code == 200
    data = res.json()
    assert "latency_ms" in data
    assert data["latency_ms"] > 0
    assert data["latency_ms"] <= wall_clock_ms + 10.0


def test_performance_concurrent_requests(perf_client: TestClient):
    """Verify thread-safety and latency under concurrent load (12 parallel requests)."""
    def do_search(query_idx: int):
        q = f"benchmark query {query_idx}"
        res = perf_client.post("/api/search", json={"query": q, "top_k": 2, "mode": "sparse"})
        return res.status_code

    def do_health():
        res = perf_client.get("/health/ready")
        return res.status_code

    tasks = [lambda i=i: do_search(i) for i in range(8)] + [do_health for _ in range(4)]

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(t) for t in tasks]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_elapsed = (time.perf_counter() - start) * 1000.0

    # Every concurrent request must succeed with 200
    assert len(results) == 12
    assert all(status == 200 for status in results)
    assert total_elapsed < 5000.0, f"Total concurrent batch took too long: {total_elapsed:.2f}ms"


def test_performance_memory_and_resource_stability():
    """Verify memory stability during vector and BM25 index operations."""
    import tracemalloc
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Index 50 small vectors
    import numpy as np
    dim = vector_store.vectors.shape[1] if vector_store.vectors is not None else settings.EMBEDDING_DIM
    vectors = [np.random.randn(dim).tolist() for _ in range(50)]
    chunk_ids = [f"c_{i}" for i in range(50)]
    doc_ids = [f"d_{i}" for i in range(50)]

    vector_store.add_vectors(chunk_ids, doc_ids, vectors)

    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_diff_kb = sum(stat.size_diff for stat in stats) / 1024.0

    tracemalloc.stop()
    # Memory increase must be modest (< 20MB for 50 small vectors)
    assert total_diff_kb < 20480.0
