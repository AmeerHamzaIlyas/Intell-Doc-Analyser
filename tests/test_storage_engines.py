"""Unit tests for storage engines: Database (SQLite), VectorStore, BM25Index, and HybridRetriever."""
import shutil
import tempfile
from pathlib import Path
import pytest
from app.models.document import DocumentMetadata, DocumentChunk
from app.services.storage.bm25_index import BM25Index
from app.services.storage.database import Database
from app.services.storage.hybrid_retriever import HybridRetriever
from app.services.storage.vector_store import VectorStore


@pytest.fixture
def temp_env():
    """Create isolated temporary directory for test databases and indices."""
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test.db"
    vector_dir = temp_dir / "vector_test"
    vector_dir.mkdir(parents=True, exist_ok=True)
    
    test_db = Database(db_path=db_path)
    test_vector_store = VectorStore(storage_dir=vector_dir)
    test_bm25 = BM25Index(storage_dir=vector_dir)
    test_retriever = HybridRetriever(
        db_instance=test_db,
        vec_store=test_vector_store,
        bm25_idx=test_bm25,
        rrf_k=60
    )
    
    yield {
        "dir": temp_dir,
        "db": test_db,
        "vector_store": test_vector_store,
        "bm25": test_bm25,
        "retriever": test_retriever,
    }
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_database_crud(temp_env):
    """Test full CRUD operations on Database."""
    db: Database = temp_env["db"]
    
    # 1. Create document
    doc = DocumentMetadata(
        document_id="doc-test-1",
        filename="report.pdf",
        original_filename="original_report.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        file_size=2048,
        file_hash="abc123hash",
        page_count=3,
        word_count=400,
        char_count=2500,
        title="Annual Financial Report",
    )
    db.save_document(doc)
    
    # 2. Retrieve by ID and by hash
    fetched = db.get_document("doc-test-1")
    assert fetched is not None
    assert fetched.filename == "report.pdf"
    assert fetched.title == "Annual Financial Report"
    
    by_hash = db.get_document_by_hash("abc123hash")
    assert by_hash is not None
    assert by_hash.document_id == "doc-test-1"
    
    # 3. Create Chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-test-1",
            chunk_index=0,
            text="Executive summary of Q4 fiscal performance.",
            page_number=1,
            section_title="Summary",
            token_count=7,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-test-1",
            chunk_index=1,
            text="Net revenue increased by 25% year-over-year.",
            page_number=2,
            section_title="Financials",
            token_count=7,
        ),
    ]
    db.save_chunks(chunks)
    
    doc_chunks = db.get_chunks_for_document("doc-test-1")
    assert len(doc_chunks) == 2
    assert doc_chunks[0].chunk_id == "chunk-1"
    assert doc_chunks[1].chunk_id == "chunk-2"
    
    batch_chunks = db.get_chunks_by_ids(["chunk-2", "chunk-1"])
    assert len(batch_chunks) == 2
    assert batch_chunks[0].chunk_id == "chunk-2"
    
    # 4. Update status and summary
    db.update_document_status("doc-test-1", "INDEXED")
    assert db.get_document("doc-test-1").status == "INDEXED"
    
    db.update_document_summary("doc-test-1", {"executive_summary": "Revenue was great."})
    assert db.get_document("doc-test-1").summary["executive_summary"] == "Revenue was great."
    
    # 5. Delete document cascades to chunks
    deleted = db.delete_document("doc-test-1")
    assert deleted is True
    assert db.get_document("doc-test-1") is None
    assert len(db.get_chunks_for_document("doc-test-1")) == 0


def test_vector_store_operations(temp_env):
    """Test vector indexing, normalization, cosine search, and persistence."""
    vs: VectorStore = temp_env["vector_store"]
    
    # Vectors:
    # chunk-1: [1.0, 0.0, 0.0]
    # chunk-2: [0.0, 1.0, 0.0]
    # chunk-3: [0.707, 0.707, 0.0]
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    doc_ids = ["doc-A", "doc-A", "doc-B"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.707, 0.707, 0.0],
    ]
    vs.add_vectors(chunk_ids, doc_ids, embeddings)
    assert vs.count() == 3
    
    # Query vector close to chunk-1: [1.0, 0.1, 0.0]
    results = vs.search(query_vector=[1.0, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "chunk-1"  # Highest cosine similarity
    
    # Query with document filter (only doc-B)
    doc_b_results = vs.search(query_vector=[1.0, 0.0, 0.0], top_k=5, document_ids=["doc-B"])
    assert len(doc_b_results) == 1
    assert doc_b_results[0][0] == "chunk-3"
    
    # Test persistence by reloading into a new store
    new_vs = VectorStore(storage_dir=temp_env["vector_store"].storage_dir)
    assert new_vs.count() == 3
    reloaded_res = new_vs.search(query_vector=[0.0, 1.0, 0.0], top_k=1)
    assert reloaded_res[0][0] == "chunk-2"
    
    # Test deletion of doc-A
    deleted_count = new_vs.delete_document_vectors("doc-A")
    assert deleted_count == 2
    assert new_vs.count() == 1
    assert new_vs.chunk_ids[0] == "chunk-3"


def test_bm25_index_operations(temp_env):
    """Test BM25 keyword search, score ranking, and persistence."""
    bm25: BM25Index = temp_env["bm25"]
    
    chunks = [
        DocumentChunk(
            chunk_id="c-1",
            document_id="doc-1",
            chunk_index=0,
            text="Antigravity deep learning artificial intelligence models in Python.",
        ),
        DocumentChunk(
            chunk_id="c-2",
            document_id="doc-1",
            chunk_index=1,
            text="Cloud databases, SQL queries, and distributed storage infrastructure.",
        ),
        DocumentChunk(
            chunk_id="c-3",
            document_id="doc-2",
            chunk_index=0,
            text="Artificial intelligence ethics and machine learning governance.",
        ),
    ]
    bm25.add_chunks(chunks)
    assert bm25.count() == 3
    
    # Search "databases" -> should match c-2
    results = bm25.search("databases storage", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "c-2"
    
    # Search "artificial intelligence" -> matches c-1 and c-3
    ai_results = bm25.search("artificial intelligence", top_k=2)
    matched_ids = [cid for cid, _ in ai_results]
    assert "c-1" in matched_ids
    assert "c-3" in matched_ids
    
    # Test document filtering
    filtered_results = bm25.search("artificial intelligence", top_k=5, document_ids=["doc-2"])
    assert len(filtered_results) == 1
    assert filtered_results[0][0] == "c-3"
    
    # Test persistence
    new_bm25 = BM25Index(storage_dir=bm25.storage_dir)
    assert new_bm25.count() == 3
    
    # Test document deletion
    deleted = new_bm25.delete_document("doc-1")
    assert deleted == 2
    assert new_bm25.count() == 1


def test_hybrid_retriever(temp_env):
    """Test end-to-end HybridRetriever combining dense and sparse search."""
    db: Database = temp_env["db"]
    vs: VectorStore = temp_env["vector_store"]
    bm25: BM25Index = temp_env["bm25"]
    retriever: HybridRetriever = temp_env["retriever"]
    
    # Save document
    doc = DocumentMetadata(
        document_id="doc-hybrid",
        filename="guide.pdf",
        original_filename="guide.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        file_size=1024,
        file_hash="hash999",
    )
    db.save_document(doc)
    
    # Chunks
    chunk_a = DocumentChunk(
        chunk_id="ch-a",
        document_id="doc-hybrid",
        chunk_index=0,
        text="FastAPI web application development with async endpoints.",
        page_number=1,
        section_title="Architecture",
    )
    chunk_b = DocumentChunk(
        chunk_id="ch-b",
        document_id="doc-hybrid",
        chunk_index=1,
        text="Deep neural networks and transformer architectures.",
        page_number=2,
        section_title="Machine Learning",
    )
    db.save_chunks([chunk_a, chunk_b])
    
    # Index in BM25
    bm25.add_chunks([chunk_a, chunk_b])
    
    # Index in Vector Store (2D vectors)
    vs.add_vectors(
        chunk_ids=["ch-a", "ch-b"],
        doc_ids=["doc-hybrid", "doc-hybrid"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    
    # 1. Search sparse
    sparse_res = retriever.search(query="FastAPI web application", mode="sparse", top_k=2)
    assert sparse_res.total_results > 0
    assert sparse_res.results[0].chunk_id == "ch-a"
    assert sparse_res.results[0].filename == "guide.pdf"
    
    # 2. Search dense
    dense_res = retriever.search(
        query="transformers",
        query_vector=[0.0, 1.0],
        mode="dense",
        top_k=2
    )
    assert dense_res.total_results > 0
    assert dense_res.results[0].chunk_id == "ch-b"
    
    # 3. Search hybrid with RRF
    hybrid_res = retriever.search(
        query="FastAPI endpoints",
        query_vector=[1.0, 0.0],
        mode="hybrid",
        top_k=2
    )
    assert hybrid_res.total_results > 0
    assert hybrid_res.results[0].chunk_id == "ch-a"
    assert hybrid_res.results[0].retrieval_method in ("hybrid_rrf", "hybrid_dense", "hybrid_sparse")
    assert hybrid_res.latency_ms >= 0.0
