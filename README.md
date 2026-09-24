Intelligent Document Understanding & Analysis System

An enterprise-grade, multimodal Document Intelligence, Hybrid Search, and Grounded Retrieval-Augmented Generation (RAG) platform. Built from the ground up for high-precision document extraction, cross-modal optical character recognition (OCR), defense-in-depth AI security, and verifiable provenance citations.


 Key Engineering Features

1. Multimodal Document Extraction & OCR
PDF Extraction: Multi-page parsing using `pypdf` with outline/bookmark extraction and per-page boundary tracking.
Vision OCR Fallback: Automatic image detection for scanned documents and image formats (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`) leveraging Gemini Multimodal Vision with graceful metadata fallback.
DOCX Parsing: Native heading hierarchy preservation and Markdown table extraction.
Unicode & Layout Cleaning: Automatic hyphenation repair, unicode NFKC normalization, and whitespace cleanup.

 2. Intelligent Structure-Aware Chunking
Preserves natural semantic boundaries (headers, paragraphs, and tables).
Computes character start/end offsets, token counts, and page numbers for strict audit trails and citation rendering.
Configurable window size (default 800 tokens) with configurable sliding overlap (default 150 tokens).

 3. Sub-Second Hybrid Search Engine
Dense Semantic Search: Cosine similarity over normalized embeddings (`gemini-embedding-001` or fallback dense hashing).
Reciprocal Rank Fusion (RRF): Merges dense and sparse result rankings using parameter $k=60$ to achieve superior precision and recall across keyword and semantic queries.
Sub-50ms Local Retrieval: In-memory optimized NumPy vector operations and inverted keyword index.

 4. Grounded RAG Studio with Verifiable Citations
Strict factual grounding prompts instructing the model to rely solely on provided context.
Every response provides explicit citation markers: `[Doc: <file>, Page: <page>]` or `[Citation: <id>]`.
Confidence scoring and Grounding Status tags: `GROUNDED`, `PARTIAL`, or `NO_EVIDENCE`.
Hallucination control: Out-of-domain or unanswerable queries safely trigger polite refusals without fabricating facts.

 5. Multi-Tier Summarization & Side-by-Side Comparison
Hierarchical Summarization: Executive summary, key findings, and section-by-section breakdown with SQLite-backed caching.
Document Comparison Engine: Dual-level document comparison combining structural diffs (token overlap, Jaccard similarity, word/page metrics) with LLM semantic delta analysis (agreements, contradictions, unique items).

 6. Defense-in-Depth AI Security & Hardening
Prompt Injection Defense: Multi-pattern regex filter detecting direct and indirect injection vectors, instruction overrides, boundary breakouts, and secret exfiltration attempts.
Context Isolation: Retrieved chunks are wrapped inside `<untrusted_document_context>` XML tags, instructing LLM reasoning engines to treat document text strictly as passive data.
Extractive Injection Filtering: Fallback extractors actively discard chunks containing injection signatures.
Output Leakage Redaction: Centralized `sanitize_ai_output` utility scrubs Google API keys (`AIza...`), Bearer tokens, and internal prompt templates before responses leave the backend.
Logging Privacy: `SecretMaskingFilter` masks credentials and sensitive patterns across all standard console and file log outputs.
Path Traversal Prevention: Storage writes and deletions are strictly validated against target root directories with canonical path resolution.


 User Workflow Experience

The single-page glassmorphic frontend (`app/static/index.html`) provides a responsive, desktop-grade dashboard:

1. Upload: Drag-and-drop or browse files with real-time XHR progress indicators.
2. Proces: Instant background extraction, chunking, and dual-engine indexing.
3. Inspect: Interactive document modal detailing MIME type, SHA-256 hash, word count, and tokenized chunks.
4. Search: Search across ingested documents using Hybrid, Dense, or Sparse BM25 retrieval.
5. Ask (RAG Studio): Chat with documents in a conversational interface with grounding status tags.
6. Provenance Drawer: Click any citation button to slide open the exact grounded source snippet, confidence score, and page number.
7. Summarize: Generate executive summaries and key findings on demand with persistent caching.
8. Compare: Select any two documents to view structural overlap and semantic comparative verdicts.


 Project Structure

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


