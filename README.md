# Intelligent Document Understanding & Analysis System

An enterprise-grade, multimodal Document Intelligence, Hybrid Search, and Grounded Retrieval-Augmented Generation (RAG) platform. Built from the ground up for high-precision document extraction, cross-modal optical character recognition (OCR), defense-in-depth AI security, and verifiable provenance citations.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Preprocessing"]
        A[Upload: PDF, DOCX, TXT, MD, Images] --> B{Magic Signature & MIME Validator}
        B -->|Valid| C[Multimodal Extractor Pipeline]
        B -->|Invalid| B1[Reject 400 / 415 / 422]
        C --> D[PDF / DOCX / Text / Vision OCR]
        D --> E[Text Cleaning & Unicode Normalization]
        E --> F[Intelligent Structure-Aware Chunker]
    end

    subgraph STORAGE ["2. Dual-Engine Storage & Indexing"]
        F --> G[(SQLite WAL Engine)]
        F --> H[Dense Vector Index (NumPy Cosine)]
        F --> I[Sparse BM25 Inverted Index]
        G --- G1[Metadata & Provenance Offsets]
    end

    subgraph RETRIEVAL ["3. Hybrid Search & RAG Synthesis"]
        Q[User Query / Question] --> J{Query Classifier & Sanitizer}
        J -->|Dense Path| H
        J -->|Sparse Path| I
        H & I --> K[Reciprocal Rank Fusion (RRF)]
        K --> L[Provenance Filter & Context Builder]
        L --> M[<untrusted_document_context> XML Encapsulation]
        M --> N[Gemini 3.6 Flash / Fallback Synthesizer]
        N --> O[AI Output Sanitizer & Secret Redactor]
        O --> P[Grounded Answer + Verifiable Citations]
    end
```

---

## Key Engineering Features

### 1. Multimodal Document Extraction & OCR
- **PDF Extraction**: Multi-page parsing using `pypdf` with outline/bookmark extraction and per-page boundary tracking.
- **Vision OCR Fallback**: Automatic image detection for scanned documents and image formats (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`) leveraging Gemini Multimodal Vision with graceful metadata fallback.
- **DOCX Parsing**: Native heading hierarchy preservation and Markdown table extraction.
- **Unicode & Layout Cleaning**: Automatic hyphenation repair, unicode NFKC normalization, and whitespace cleanup.

### 2. Intelligent Structure-Aware Chunking
- Preserves natural semantic boundaries (headers, paragraphs, and tables).
- Computes character start/end offsets, token counts, and page numbers for strict audit trails and citation rendering.
- Configurable window size (default 800 tokens) with configurable sliding overlap (default 150 tokens).

### 3. Sub-Second Hybrid Search Engine
- **Dense Semantic Search**: Cosine similarity over normalized embeddings (`gemini-embedding-001` or fallback dense hashing).
- **Sparse Lexical Search**: BM25 inverted index with tokenization, document frequency weighting, and saturation scaling.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse result rankings using parameter $k=60$ to achieve superior precision and recall across keyword and semantic queries.
- **Sub-50ms Local Retrieval**: In-memory optimized NumPy vector operations and inverted keyword index.

### 4. Grounded RAG Studio with Verifiable Citations
- Strict factual grounding prompts instructing the model to rely solely on provided context.
- Every response provides explicit citation markers: `[Doc: <file>, Page: <page>]` or `[Citation: <id>]`.
- Confidence scoring and Grounding Status tags: `GROUNDED`, `PARTIAL`, or `NO_EVIDENCE`.
- Hallucination control: Out-of-domain or unanswerable queries safely trigger polite refusals without fabricating facts.

### 5. Multi-Tier Summarization & Side-by-Side Comparison
- **Hierarchical Summarization**: Executive summary, key findings, and section-by-section breakdown with SQLite-backed caching.
- **Document Comparison Engine**: Dual-level document comparison combining structural diffs (token overlap, Jaccard similarity, word/page metrics) with LLM semantic delta analysis (agreements, contradictions, unique items).

### 6. Defense-in-Depth AI Security & Hardening
- **Prompt Injection Defense**: Multi-pattern regex filter detecting direct and indirect injection vectors, instruction overrides, boundary breakouts, and secret exfiltration attempts.
- **Context Isolation**: Retrieved chunks are wrapped inside `<untrusted_document_context>` XML tags, instructing LLM reasoning engines to treat document text strictly as passive data.
- **Extractive Injection Filtering**: Fallback extractors actively discard chunks containing injection signatures.
- **Output Leakage Redaction**: Centralized `sanitize_ai_output` utility scrubs Google API keys (`AIza...`), Bearer tokens, and internal prompt templates before responses leave the backend.
- **Logging Privacy**: `SecretMaskingFilter` masks credentials and sensitive patterns across all standard console and file log outputs.
- **Path Traversal Prevention**: Storage writes and deletions are strictly validated against target root directories with canonical path resolution.
- **API Key Authentication**: Configurable middleware protecting `/api/*` routes via `X-API-Key` or `Authorization: Bearer` headers.

---

## User Workflow Experience

The single-page glassmorphic frontend (`app/static/index.html`) provides a responsive, desktop-grade dashboard:

1. **Upload**: Drag-and-drop or browse files with real-time XHR progress indicators.
2. **Process**: Instant background extraction, chunking, and dual-engine indexing.
3. **Inspect**: Interactive document modal detailing MIME type, SHA-256 hash, word count, and tokenized chunks.
4. **Search**: Search across ingested documents using Hybrid, Dense, or Sparse BM25 retrieval.
5. **Ask (RAG Studio)**: Chat with documents in a conversational interface with grounding status tags.
6. **Provenance Drawer**: Click any citation button to slide open the exact grounded source snippet, confidence score, and page number.
7. **Summarize**: Generate executive summaries and key findings on demand with persistent caching.
8. **Compare**: Select any two documents to view structural overlap and semantic comparative verdicts.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health/live` | Application liveness probe |
| `GET` | `/health/ready` | Readiness check (database, vector store, BM25 indices) |
| `GET` | `/api/documents/stats/overview` | High-level metrics (total docs, chunks, indexed vectors) |
| `POST` | `/api/documents/upload` | Multipart upload and ingestion of documents |
| `GET` | `/api/documents` | Paginated list of ingested documents |
| `GET` | `/api/documents/{doc_id}` | Detailed document metadata and chunk metrics |
| `GET` | `/api/documents/{doc_id}/chunks` | Full list of extracted semantic chunks for a document |
| `DELETE`| `/api/documents/{doc_id}` | Delete document and cascade delete all vector and BM25 indices |
| `POST` | `/api/search` | Execute dense, sparse, or hybrid reciprocal rank fusion search |
| `POST` | `/api/rag/query` | Grounded question answering with citations and conversation history |
| `POST` | `/api/summarize/{doc_id}` | Generate or retrieve cached multi-tier summary |
| `POST` | `/api/compare` | Compare two documents structurally and semantically |

---

## Project Structure

```
Intell-Doc-Analyser/
├── app/
│   ├── main.py                  # FastAPI app lifecycle, middleware, exception handlers
│   ├── config.py                # Environment configuration and directory provisioning
│   ├── logging_config.py        # Centralized logging with SecretMaskingFilter
│   ├── api/                     # REST API route handlers
│   │   ├── health.py            # Liveness & readiness probes
│   │   ├── documents.py         # Ingestion, listing, inspection, deletion
│   │   ├── search.py            # Hybrid search endpoint
│   │   ├── rag.py               # Grounded RAG Q&A endpoint
│   │   ├── summarize.py         # Multi-tier summarization
│   │   └── compare.py           # Document comparison & diffing
│   ├── core/                    # Core models, constants, and security utilities
│   │   ├── constants.py         # File types, magic bytes, status constants
│   │   ├── exceptions.py        # Custom domain exception taxonomy
│   │   └── security.py          # Injection detection, XML isolation, output redaction
│   ├── models/                  # Pydantic schemas (Document, Chunk, Search, RAG)
│   ├── services/
│   │   ├── document_service.py  # Central ingestion and processing orchestrator
│   │   ├── extraction/          # PDF, DOCX, Text, and Gemini Vision OCR extractors
│   │   ├── processing/          # Text cleaning, intelligent chunking, structure detection
│   │   ├── storage/             # SQLite WAL database, NumPy VectorStore, BM25 Index, Hybrid Retriever
│   │   └── ai/                  # Gemini client, embedding service, RAG engine, summarizer, comparator
│   └── static/                  # Single-page frontend dashboard (HTML, CSS, ES Modules)
│       ├── index.html           # Glassmorphic UI layout
│       ├── css/styles.css       # Design tokens, typography, animations
│       └── js/                  # api.js, app.js, rag_chat.js, viewer.js
├── tests/                       # Complete automated audit and regression test suites
│   ├── test_foundation.py              # Baseline system tests
│   ├── test_storage_engines.py         # SQLite, VectorStore, and BM25 engine tests
│   ├── test_pipeline_stage3.py         # Extraction, chunking, and validation tests
│   ├── test_api_endpoints.py           # REST API lifecycle tests
│   ├── test_ai_services.py             # RAG, summarization, and comparison unit tests
│   ├── test_audit_ai_evaluation.py     # Grounding, citations, and hallucination tests
│   ├── test_audit_edge_cases.py        # Boundary conditions and corrupt file handling
│   ├── test_audit_performance.py       # Concurrency and latency benchmark tests
│   ├── test_audit_security.py          # Security, path traversal, and injection tests
│   ├── test_adversarial_rag_security.py# Indirect prompt injection & redaction tests
│   └── test_e2e_user_workflow.py       # Full 11-step end-to-end user perspective audit
├── Dockerfile                   # Containerized production deployment
├── pyproject.toml               # Python project configuration
├── requirements.txt             # Locked production dependencies
├── .env.example                 # Example configuration environment variables
└── README.md                    # Project documentation
```

---

## Quickstart & Installation

### Prerequisites
- Python 3.10+
- (Optional) Google Gemini API Key (set via `GEMINI_API_KEY`)

### Local Setup
1. Clone the repository or navigate to the project directory:
   ```bash
   cd Intell-Doc-Analyser
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your GEMINI_API_KEY if desired
   ```

5. Launch the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. Open your browser:
   - **Dashboard**: [http://localhost:8000](http://localhost:8000)
   - **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Readiness**: [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

### Docker Deployment
```bash
docker build -t intell-doc-analyser .
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key_here" intell-doc-analyser
```

---

## Verification & Test Suite Execution

The repository includes a comprehensive automated test suite spanning 12 test modules covering foundational unit tests, storage engine validation, multimodal extraction, hybrid search, AI grounding/citations, edge cases, security audits, adversarial penetration tests, and end-to-end user journeys.

To run the complete test suite:
```bash
pytest -v
```
