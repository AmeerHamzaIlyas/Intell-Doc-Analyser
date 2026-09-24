"""AI Evaluation audit test suite: retrieval relevance, faithfulness, citations, hallucination, and grounding."""
import io
import shutil
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from app.config import settings
from app.main import app
from app.models.search_rag import RAGRequest
from app.services.ai.rag_service import rag_service
from app.services.storage.bm25_index import bm25_index
from app.services.storage.database import db
from app.services.storage.hybrid_retriever import hybrid_retriever
from app.services.storage.vector_store import vector_store


@pytest.fixture(scope="module")
def ai_eval_env():
    """Setup isolated test corpus with known ground-truth facts for AI evaluation."""
    temp_dir = Path(tempfile.mkdtemp())
    orig_data = settings.DATA_DIR
    orig_upload = settings.UPLOAD_DIR
    orig_storage = settings.STORAGE_DIR
    orig_db = settings.DB_PATH
    orig_vector = settings.VECTOR_DIR

    settings.DATA_DIR = temp_dir
    settings.UPLOAD_DIR = temp_dir / "uploads"
    settings.STORAGE_DIR = temp_dir / "storage"
    settings.DB_PATH = temp_dir / "ai_eval.db"
    settings.VECTOR_DIR = temp_dir / "ai_eval_vector"
    settings.ensure_directories()

    db.db_path = settings.DB_PATH
    db.init_db()
    vector_store.storage_dir = settings.VECTOR_DIR
    vector_store.clear()
    bm25_index.storage_dir = settings.VECTOR_DIR
    bm25_index.clear()

    with TestClient(app) as test_client:
        # Ingest Ground Truth Document 1: Quantum Computing Progress
        quantum_doc = (
            b"# Quantum Computing Breakthrough 2026\n\n"
            b"In October 2026, researchers demonstrated quantum advantage with a 1,024-qubit processor.\n\n"
            b"## Error Correction\n"
            b"Surface code error correction achieved a physical-to-logical qubit ratio of 100:1.\n\n"
            b"## Cryogenic Systems\n"
            b"Operating temperature was stabilized at 15 millikelvin using helium dilution refrigerators.\n\n"
            b"## Commercial Applications\n"
            b"Targeted applications include molecular battery modeling and high-throughput drug candidate screening."
        )
        test_client.post(
            "/api/documents/upload",
            files={"file": ("quantum_breakthrough.md", io.BytesIO(quantum_doc), "text/markdown")}
        )

        # Ingest Ground Truth Document 2: Global Energy Transition
        energy_doc = (
            b"# Global Renewable Energy Transition Report 2026\n\n"
            b"Global solar installations reached 480 gigawatts capacity in 2026.\n\n"
            b"## Grid Storage\n"
            b"Utility-scale battery storage deployments grew by 55% year-over-year.\n\n"
            b"## Wind Energy\n"
            b"Offshore wind farms contributed 95 gigawatts of generation capacity in northern waters."
        )
        test_client.post(
            "/api/documents/upload",
            files={"file": ("renewable_energy_2026.md", io.BytesIO(energy_doc), "text/markdown")}
        )

        yield test_client

    shutil.rmtree(temp_dir, ignore_errors=True)
    settings.DATA_DIR = orig_data
    settings.UPLOAD_DIR = orig_upload
    settings.STORAGE_DIR = orig_storage
    settings.DB_PATH = orig_db
    settings.VECTOR_DIR = orig_vector
    settings.ensure_directories()


def test_ai_retrieval_relevance(ai_eval_env: TestClient):
    """Evaluate retrieval precision and recall across sparse, dense, and hybrid modes."""
    # 1. Exact term query: "1,024-qubit" should retrieve quantum document at rank 1
    res_hybrid = ai_eval_env.post(
        "/api/search",
        json={"query": "1,024-qubit processor quantum advantage", "top_k": 3, "mode": "hybrid"}
    )
    assert res_hybrid.status_code == 200
    results = res_hybrid.json()["results"]
    assert len(results) > 0
    top_result = results[0]
    assert "quantum_breakthrough.md" in top_result["filename"]
    assert "1,024-qubit" in top_result["text"]

    # 2. Sparse BM25 retrieval for specific technical terminology
    res_sparse = ai_eval_env.post(
        "/api/search",
        json={"query": "millikelvin cryogenic helium dilution", "top_k": 2, "mode": "sparse"}
    )
    assert res_sparse.status_code == 200
    assert len(res_sparse.json()["results"]) > 0
    assert "quantum_breakthrough.md" in res_sparse.json()["results"][0]["filename"]

    # 3. Energy query should retrieve energy document at rank 1
    res_energy = ai_eval_env.post(
        "/api/search",
        json={"query": "solar installations gigawatts capacity", "top_k": 2, "mode": "hybrid"}
    )
    assert res_energy.status_code == 200
    assert "renewable_energy_2026.md" in res_energy.json()["results"][0]["filename"]


def test_ai_context_relevance(ai_eval_env: TestClient):
    """Verify context chunks retrieved for RAG directly match the user query intent."""
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "What is the physical-to-logical qubit ratio?", "top_k": 2}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    assert len(data["referenced_chunks"]) > 0
    top_chunk = data["referenced_chunks"][0]
    assert "100:1" in top_chunk["text"]
    assert "Error Correction" in top_chunk["text"] or "Error Correction" in (top_chunk.get("section_title") or "")


def test_ai_answer_faithfulness(ai_eval_env: TestClient):
    """Evaluate that the generated answer is faithful to the source document context."""
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "At what temperature was the cryogenic system stabilized?", "top_k": 2}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    answer = data["answer"]
    assert "15 millikelvin" in answer or "15" in answer
    assert data["grounding_status"] in ("GROUNDED", "PARTIAL")


def test_ai_citation_correctness(ai_eval_env: TestClient):
    """Evaluate that citations accurately point to the source document, page, and section."""
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "How many gigawatts of solar installations were deployed in 2026?", "top_k": 2}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    citations = data["citations"]
    assert len(citations) >= 1
    c = citations[0]
    assert "renewable_energy_2026.md" in c["filename"]
    assert c["page_number"] == 1
    assert c["confidence_score"] >= 0.70  # Valid calibrated confidence score
    assert len(c["snippet"]) > 0


def test_ai_hallucination_prevention(ai_eval_env: TestClient):
    """Verify that unstated facts or fabricated numbers are not asserted as true facts."""
    # Ask about a fictional detail not present in the quantum doc
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "What was the total budget cost in dollars for the 1024-qubit project?", "top_k": 2}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    # The system must not hallucinate a dollar figure like "$50 million"
    assert "$" not in data["answer"] or "not contain" in data["answer"].lower()


def test_ai_out_of_context_rejection(ai_eval_env: TestClient):
    """Evaluate that completely irrelevant, out-of-corpus queries are rejected with NO_EVIDENCE."""
    out_of_context_queries = [
        "What is the capital city of Mars?",
        "Who won the 1994 FIFA World Cup soccer tournament?",
        "Recipe for homemade sourdough bread with active starter yeast",
    ]
    for q in out_of_context_queries:
        rag_res = ai_eval_env.post(
            "/api/rag/query",
            json={"query": q, "top_k": 2}
        )
        assert rag_res.status_code == 200
        data = rag_res.json()
        assert data["grounding_status"] == "NO_EVIDENCE"
        assert "not contain sufficient information" in data["answer"].lower() or "insufficient" in data["answer"].lower()
        assert len(data["citations"]) == 0


def test_ai_answer_completeness(ai_eval_env: TestClient):
    """Evaluate multi-part question addressing multiple facets."""
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "What are the commercial applications of the quantum processor?", "top_k": 3}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    answer = data["answer"].lower()
    assert "battery" in answer or "drug" in answer or "molecular" in answer


def test_ai_answer_consistency(ai_eval_env: TestClient):
    """Evaluate that repeated queries yield consistent factual metrics."""
    query = "How much did utility-scale battery storage deployments grow?"
    res1 = ai_eval_env.post("/api/rag/query", json={"query": query, "top_k": 2})
    res2 = ai_eval_env.post("/api/rag/query", json={"query": query, "top_k": 2})

    assert res1.status_code == 200
    assert res2.status_code == 200

    ans1 = res1.json()["answer"]
    ans2 = res2.json()["answer"]

    # Both answers must agree on the factual figure (55%)
    assert "55%" in ans1
    assert "55%" in ans2


def test_regression_referenced_chunks_schema_and_null_safety(ai_eval_env: TestClient):
    """Regression test ensuring referenced_chunks are properly extracted, populated with complete schema,
    and null-safe when evaluating metadata like section_title."""
    # 1. Valid context query returning chunks
    rag_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "What was the operating temperature stabilized at?", "top_k": 2}
    )
    assert rag_res.status_code == 200
    data = rag_res.json()
    assert "referenced_chunks" in data
    assert isinstance(data["referenced_chunks"], list)
    assert len(data["referenced_chunks"]) > 0

    # Ensure top_chunk can be safely extracted and contains required fields
    top_chunk = data["referenced_chunks"][0]
    assert "chunk_id" in top_chunk
    assert "document_id" in top_chunk
    assert "filename" in top_chunk
    assert "page_number" in top_chunk
    assert "text" in top_chunk
    assert "score" in top_chunk
    # section_title can be None or string, must be safe against string operations
    sec_title = top_chunk.get("section_title")
    assert sec_title is None or isinstance(sec_title, str)
    assert "Cryogenic" in top_chunk["text"] or "Cryogenic" in (sec_title or "")

    # 2. Out-of-context query returns valid list without schema failure
    out_res = ai_eval_env.post(
        "/api/rag/query",
        json={"query": "Xylophone zebra quantum unicorn nonsense query 999", "top_k": 2}
    )
    assert out_res.status_code == 200
    out_data = out_res.json()
    assert "referenced_chunks" in out_data
    assert isinstance(out_data["referenced_chunks"], list)

