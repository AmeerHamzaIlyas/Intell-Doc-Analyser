"""Unit tests for Stage 4: AI Services (Embeddings, RAG, Summarization, Comparison)."""
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from app.models.document import DocumentChunk, DocumentMetadata
from app.models.search_rag import RAGRequest
from app.services.ai.comparison_service import ComparisonService
from app.services.ai.embedding_service import EmbeddingService
from app.services.ai.rag_service import RAGService
from app.services.ai.summary_service import SummaryService
from app.services.storage.bm25_index import BM25Index
from app.services.storage.database import Database
from app.services.storage.hybrid_retriever import HybridRetriever
from app.services.storage.vector_store import VectorStore


@pytest.fixture
def ai_env():
    """Create isolated temporary environment for AI service tests."""
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "ai_test.db"
    vector_dir = temp_dir / "ai_vector"
    vector_dir.mkdir(parents=True, exist_ok=True)

    test_db = Database(db_path=db_path)
    test_vs = VectorStore(storage_dir=vector_dir)
    test_bm25 = BM25Index(storage_dir=vector_dir)
    test_retriever = HybridRetriever(db_instance=test_db, vec_store=test_vs, bm25_idx=test_bm25)
    test_embedder = EmbeddingService(vec_store=test_vs)
    test_rag = RAGService(retriever=test_retriever, embedder=test_embedder)
    test_summary = SummaryService(db_instance=test_db)
    test_comparison = ComparisonService(db_instance=test_db)

    yield {
        "dir": temp_dir,
        "db": test_db,
        "vs": test_vs,
        "bm25": test_bm25,
        "retriever": test_retriever,
        "embedder": test_embedder,
        "rag": test_rag,
        "summary": test_summary,
        "comparison": test_comparison,
    }
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_service_properties(ai_env):
    """Test embedding generation and L2 normalization."""
    embedder: EmbeddingService = ai_env["embedder"]
    vs: VectorStore = ai_env["vs"]

    texts = [
        "Cloud architecture and microservices deployment.",
        "Deep learning and natural language processing models."
    ]
    embeddings = embedder.embed_texts(texts)
    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) in (384, 3072)
        norm = np.linalg.norm(np.array(emb, dtype=np.float32))
        assert np.isclose(norm, 1.0, atol=1e-3)

    # Test indexing chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_test_1",
            document_id="doc_emb",
            chunk_index=0,
            text=texts[0],
            page_number=1,
            section_title="Architecture",
        ),
        DocumentChunk(
            chunk_id="chunk_test_2",
            document_id="doc_emb",
            chunk_index=1,
            text=texts[1],
            page_number=2,
            section_title="AI",
        ),
    ]
    embedder.index_chunks(chunks)
    assert vs.count() == 2


def test_rag_service_flow(ai_env):
    """Test full RAG question answering pipeline with citation resolution."""
    db: Database = ai_env["db"]
    vs: VectorStore = ai_env["vs"]
    bm25: BM25Index = ai_env["bm25"]
    embedder: EmbeddingService = ai_env["embedder"]
    rag: RAGService = ai_env["rag"]

    # 1. Setup document
    doc = DocumentMetadata(
        document_id="doc_policy",
        filename="company_policy.pdf",
        original_filename="company_policy.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        file_size=4096,
        file_hash="hash_pol",
        page_count=3,
    )
    db.save_document(doc)

    chunks = [
        DocumentChunk(
            chunk_id="c_pol_1",
            document_id="doc_policy",
            chunk_index=0,
            text="Standard employee annual leave is 25 days per calendar year.",
            page_number=1,
            section_title="Vacation Policy",
        ),
        DocumentChunk(
            chunk_id="c_pol_2",
            document_id="doc_policy",
            chunk_index=1,
            text="Health insurance coverage takes effect immediately upon employment start date.",
            page_number=2,
            section_title="Health Benefits",
        ),
    ]
    db.save_chunks(chunks)
    bm25.add_chunks(chunks)
    embedder.index_chunks(chunks)

    # 2. Query RAG
    request = RAGRequest(query="How many vacation days do employees receive?", top_k=2)
    response = rag.answer_query(request)

    assert response.query == request.query
    assert len(response.referenced_chunks) > 0
    assert response.referenced_chunks[0].document_id == "doc_policy"
    assert response.grounding_status in ("GROUNDED", "PARTIAL")
    assert len(response.citations) > 0
    assert response.latency_ms >= 0.0


def test_summary_service(ai_env):
    """Test multi-tier document summarization."""
    db: Database = ai_env["db"]
    summary_svc: SummaryService = ai_env["summary"]

    doc = DocumentMetadata(
        document_id="doc_summary_test",
        filename="project_charter.txt",
        original_filename="project_charter.txt",
        file_type="txt",
        mime_type="text/plain",
        file_size=2048,
        file_hash="hash_charter",
        title="Project Charter 2026",
    )
    db.save_document(doc)

    chunks = [
        DocumentChunk(
            chunk_id="ch_1",
            document_id="doc_summary_test",
            chunk_index=0,
            text="Project Antigravity focuses on automated code intelligence and multi-modal understanding.",
            page_number=1,
            section_title="Objective",
        ),
        DocumentChunk(
            chunk_id="ch_2",
            document_id="doc_summary_test",
            chunk_index=1,
            text="Key milestones include Q1 prototype launch and Q2 enterprise security verification.",
            page_number=1,
            section_title="Milestones",
        ),
    ]
    db.save_chunks(chunks)

    summary = summary_svc.summarize_document("doc_summary_test")
    assert summary.document_id == "doc_summary_test"
    assert len(summary.executive_summary) > 0
    assert len(summary.key_findings) > 0


def test_comparison_service(ai_env):
    """Test structural and semantic comparison between two documents."""
    db: Database = ai_env["db"]
    comp_svc: ComparisonService = ai_env["comparison"]

    # Doc A
    doc_a = DocumentMetadata(
        document_id="doc_v1",
        filename="Contract_v1.docx",
        original_filename="Contract_v1.docx",
        file_type="docx",
        mime_type="application/docx",
        file_size=1024,
        file_hash="hash_v1",
        word_count=50,
        page_count=1,
        title="Service Agreement v1",
    )
    db.save_document(doc_a)
    chunk_a = DocumentChunk(
        chunk_id="ca_1",
        document_id="doc_v1",
        chunk_index=0,
        text="The vendor shall deliver services within 30 days of contract signing.\nPayment shall be Net 30.",
        page_number=1,
    )
    db.save_chunks([chunk_a])

    # Doc B (modified)
    doc_b = DocumentMetadata(
        document_id="doc_v2",
        filename="Contract_v2.docx",
        original_filename="Contract_v2.docx",
        file_type="docx",
        mime_type="application/docx",
        file_size=1100,
        file_hash="hash_v2",
        word_count=55,
        page_count=1,
        title="Service Agreement v2",
    )
    db.save_document(doc_b)
    chunk_b = DocumentChunk(
        chunk_id="cb_1",
        document_id="doc_v2",
        chunk_index=0,
        text="The vendor shall deliver services within 15 days of contract signing.\nPayment shall be Net 15.\nLate fees of 2% apply.",
        page_number=1,
    )
    db.save_chunks([chunk_b])

    comparison = comp_svc.compare_documents("doc_v1", "doc_v2")
    assert comparison.doc_a.document_id == "doc_v1"
    assert comparison.doc_b.document_id == "doc_v2"
    assert "similarity_percentage" in comparison.structural_diff
    assert comparison.structural_diff["similarity_percentage"] > 0
    assert "unified_diff" in comparison.structural_diff
    assert "overall_verdict" in comparison.semantic_analysis
