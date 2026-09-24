/**
 * Main Application Orchestrator for Intelligent Document Understanding & Analysis System
 */

import { Api } from './api.js';
import { Viewer } from './viewer.js';
import { RagChat } from './rag_chat.js';

class App {
  constructor() {
    this.documents = [];
    this.currentTab = 'repo';
  }

  async init() {
    this.setupNavigation();
    this.setupUploadHandlers();
    this.setupSearchHandlers();
    this.setupSummaryHandlers();
    this.setupComparisonHandlers();
    this.setupModalHandlers();
    
    // Initialize RAG chat
    RagChat.init();

    // Initial data load
    await this.refreshSystemStats();
    await this.loadDocuments();

    // Periodic light health polling (every 30s)
    setInterval(() => this.refreshSystemStats(), 30000);
  }

  /* -------------------------------------------------------------------------- */
  /* Navigation & Tabs                                                          */
  /* -------------------------------------------------------------------------- */
  setupNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        this.switchTab(tabId);
      });
    });
  }

  switchTab(tabId) {
    this.currentTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.classList.toggle('active', p.id === `tab-${tabId}`);
    });

    // Populate dropdowns if switching to compare or summary
    if (tabId === 'summary') {
      this.populateSummaryDropdown();
    } else if (tabId === 'compare') {
      this.populateCompareDropdowns();
    } else if (tabId === 'health') {
      this.loadHealthDiagnostics();
    }
  }

  /* -------------------------------------------------------------------------- */
  /* System Metrics & Overview                                                  */
  /* -------------------------------------------------------------------------- */
  async refreshSystemStats() {
    try {
      const stats = await Api.getStats();
      const docCountEl = document.getElementById('metricDocs');
      const vectorCountEl = document.getElementById('metricVectors');
      const chunkCountEl = document.getElementById('metricChunks');

      if (docCountEl) docCountEl.textContent = stats.total_documents || 0;
      if (vectorCountEl) vectorCountEl.textContent = stats.vector_count || 0;
      if (chunkCountEl) chunkCountEl.textContent = stats.total_chunks || 0;

      const dot = document.querySelector('.status-dot');
      const label = document.getElementById('statusLabel');
      if (dot && label) {
        dot.style.backgroundColor = 'var(--accent-emerald)';
        label.textContent = 'Engine Online';
      }
    } catch (err) {
      console.warn("Failed to load system stats", err);
      const dot = document.querySelector('.status-dot');
      const label = document.getElementById('statusLabel');
      if (dot && label) {
        dot.style.backgroundColor = 'var(--accent-amber)';
        label.textContent = 'Degraded';
      }
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Document Repository & Ingestion                                            */
  /* -------------------------------------------------------------------------- */
  async loadDocuments() {
    const container = document.getElementById('documentsGrid');
    if (!container) return;

    try {
      const res = await Api.listDocuments(100, 0);
      this.documents = res.documents || [];
      this.renderDocumentCards();
      this.updateDocFilters();
    } catch (err) {
      this.showToast(err.message, 'error');
    }
  }

  renderDocumentCards() {
    const container = document.getElementById('documentsGrid');
    if (!container) return;

    if (this.documents.length === 0) {
      container.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
          <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📂</div>
          <h3 style="color: #fff; font-size: 1.1rem; margin-bottom: 0.35rem;">No documents ingested yet</h3>
          <p style="font-size: 0.85rem;">Upload PDF, DOCX, TXT, or scanned image documents above to begin indexing.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = this.documents.map(item => {
      const m = item.metadata;
      let typeClass = 'doc-type-pdf';
      if (m.file_type === 'docx') typeClass = 'doc-type-docx';
      else if (m.file_type === 'txt' || m.file_type === 'md') typeClass = 'doc-type-txt';
      else if (['png', 'jpg', 'jpeg', 'tiff'].includes(m.file_type)) typeClass = 'doc-type-img';

      const formattedSize = (m.file_size / 1024).toFixed(1) + ' KB';
      const createdDate = new Date(m.created_at).toLocaleDateString();

      return `
        <div class="doc-card" data-doc-id="${m.document_id}">
          <div>
            <div class="doc-card-top">
              <span class="doc-type-badge ${typeClass}">${m.file_type}</span>
              <div style="flex: 1; overflow: hidden;">
                <h4 class="doc-card-title" title="${escapeHtml(m.filename)}">${escapeHtml(m.filename)}</h4>
                <div class="doc-card-meta">${formattedSize} • ${createdDate}</div>
              </div>
            </div>

            <div class="doc-card-stats">
              <div>
                <div class="doc-stat-val">${m.page_count}</div>
                <div class="doc-stat-label">Pages</div>
              </div>
              <div>
                <div class="doc-stat-val">${m.word_count}</div>
                <div class="doc-stat-label">Words</div>
              </div>
              <div>
                <div class="doc-stat-val">${item.chunk_count}</div>
                <div class="doc-stat-label">Chunks</div>
              </div>
            </div>
          </div>

          <div class="doc-card-actions">
            <button class="btn-icon btn-inspect" data-doc-id="${m.document_id}" title="Inspect Chunks & Metadata">
              🔍
            </button>
            <button class="btn-icon btn-summarize" data-doc-id="${m.document_id}" title="Summarize Document">
              📑
            </button>
            <button class="btn-icon btn-danger btn-delete" data-doc-id="${m.document_id}" title="Delete Document & Cascade Indices">
              🗑️
            </button>
          </div>
        </div>
      `;
    }).join('');

    // Wire Card Buttons
    container.querySelectorAll('.btn-inspect').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.inspectDocument(btn.dataset.docId);
      });
    });

    container.querySelectorAll('.btn-summarize').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.quickSummarize(btn.dataset.docId);
      });
    });

    container.querySelectorAll('.btn-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.confirmDelete(btn.dataset.docId);
      });
    });
  }

  setupUploadHandlers() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const btnSelect = document.getElementById('btnSelectFile');

    if (!dropzone || !fileInput) return;

    btnSelect.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
      if (e.target !== btnSelect) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) this.handleFileUpload(files[0]);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        this.handleFileUpload(fileInput.files[0]);
        fileInput.value = '';
      }
    });
  }

  async handleFileUpload(file) {
    const progressCont = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    progressCont.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = `Uploading ${file.name}... (0%)`;

    try {
      const detail = await Api.uploadDocument(file, (percent) => {
        progressBar.style.width = `${percent}%`;
        progressText.textContent = `Processing and chunking ${file.name}... (${percent}%)`;
      });

      progressBar.style.width = '100%';
      progressText.textContent = `Successfully ingested "${detail.metadata.filename}"!`;
      this.showToast(`Document "${detail.metadata.filename}" indexed (${detail.chunk_count} chunks)`, 'success');

      setTimeout(() => {
        progressCont.style.display = 'none';
      }, 2000);

      await this.loadDocuments();
      await this.refreshSystemStats();

    } catch (err) {
      progressBar.style.width = '0%';
      progressCont.style.display = 'none';
      this.showToast(`Upload failed: ${err.message}`, 'error');
    }
  }

  async inspectDocument(docId) {
    try {
      const [detail, chunks] = await Promise.all([
        Api.getDocument(docId),
        Api.getChunks(docId)
      ]);
      Viewer.openDocumentModal(detail, chunks);
    } catch (err) {
      this.showToast(`Failed to inspect document: ${err.message}`, 'error');
    }
  }

  async confirmDelete(docId) {
    const doc = this.documents.find(d => d.metadata.document_id === docId);
    const filename = doc ? doc.metadata.filename : docId;

    if (!confirm(`Are you sure you want to delete "${filename}" and purge all its vector and BM25 indices?`)) {
      return;
    }

    try {
      await Api.deleteDocument(docId);
      this.showToast(`Document "${filename}" and its indices deleted.`, 'info');
      await this.loadDocuments();
      await this.refreshSystemStats();
    } catch (err) {
      this.showToast(`Failed to delete: ${err.message}`, 'error');
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Hybrid Search Studio                                                       */
  /* -------------------------------------------------------------------------- */
  setupSearchHandlers() {
    const searchBtn = document.getElementById('btnExecuteSearch');
    const searchInput = document.getElementById('searchInput');

    if (searchBtn && searchInput) {
      searchBtn.addEventListener('click', () => this.executeSearch());
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.executeSearch();
      });
    }
  }

  async executeSearch() {
    const searchInput = document.getElementById('searchInput');
    const modeSelect = document.getElementById('searchMode');
    const topKSelect = document.getElementById('searchTopK');
    const docFilter = document.getElementById('searchDocFilter');
    const resultsCont = document.getElementById('searchResultsList');
    const latencyBadge = document.getElementById('searchLatencyBadge');

    const query = searchInput.value.trim();
    if (!query) {
      this.showToast('Please enter a search query', 'info');
      return;
    }

    resultsCont.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <div class="status-dot" style="margin: 0 auto 0.5rem;"></div>
        Searching dense embeddings and sparse BM25 indices...
      </div>
    `;

    try {
      const selectedDoc = docFilter.value ? [docFilter.value] : null;
      const res = await Api.search({
        query,
        mode: modeSelect.value,
        topK: topKSelect.value,
        documentIds: selectedDoc
      });

      if (latencyBadge) {
        latencyBadge.textContent = `${res.latency_ms.toFixed(1)}ms (${res.mode.toUpperCase()})`;
        latencyBadge.style.display = 'inline-block';
      }

      if (res.results.length === 0) {
        resultsCont.innerHTML = `
          <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
            No matching chunks found above score threshold.
          </div>
        `;
        return;
      }

      resultsCont.innerHTML = res.results.map((r, idx) => `
        <div class="search-result-item">
          <div class="result-header">
            <div class="result-doc-info">
              <strong style="color: #fff; font-size: 0.95rem;">${escapeHtml(r.filename)}</strong>
              <span class="badge-page">Page ${r.page_number}</span>
              <span class="badge-page">${escapeHtml(r.section_title || 'General')}</span>
            </div>
            <div>
              <span class="badge-score">Score: ${r.score.toFixed(4)}</span>
              <span class="badge-page" style="margin-left: 0.4rem; text-transform: uppercase;">${r.retrieval_method}</span>
            </div>
          </div>
          <div class="result-snippet">${escapeHtml(r.text)}</div>
        </div>
      `).join('');

    } catch (err) {
      resultsCont.innerHTML = `
        <div style="color: var(--accent-rose); padding: 1.5rem; text-align: center;">
          Error executing search: ${err.message}
        </div>
      `;
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Summarization Studio                                                       */
  /* -------------------------------------------------------------------------- */
  setupSummaryHandlers() {
    const btnSummarize = document.getElementById('btnGenerateSummary');
    if (btnSummarize) {
      btnSummarize.addEventListener('click', () => {
        const select = document.getElementById('summaryDocSelect');
        if (select && select.value) {
          this.generateSummary(select.value);
        } else {
          this.showToast('Select a document to summarize', 'info');
        }
      });
    }
  }

  populateSummaryDropdown() {
    const select = document.getElementById('summaryDocSelect');
    if (!select) return;

    select.innerHTML = '<option value="">-- Choose Document --</option>' +
      this.documents.map(d => `
        <option value="${d.metadata.document_id}">${escapeHtml(d.metadata.filename)}</option>
      `).join('');
  }

  quickSummarize(docId) {
    this.switchTab('summary');
    const select = document.getElementById('summaryDocSelect');
    if (select) {
      select.value = docId;
      this.generateSummary(docId);
    }
  }

  async generateSummary(docId) {
    const output = document.getElementById('summaryOutput');
    if (!output) return;

    output.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <div class="status-dot" style="margin: 0 auto 0.5rem;"></div>
        Synthesizing hierarchical multi-tier summary...
      </div>
    `;

    try {
      const summary = await Api.summarize(docId);
      const findingsList = (summary.key_findings || []).map(f => `<li>${escapeHtml(f)}</li>`).join('');
      const actionsList = (summary.action_items || []).map(a => `<li>${escapeHtml(a)}</li>`).join('');

      let sectionsHtml = '';
      if (summary.section_summaries && Object.keys(summary.section_summaries).length > 0) {
        sectionsHtml = Object.entries(summary.section_summaries).map(([sec, text]) => `
          <div style="margin-bottom: 0.75rem;">
            <strong style="color: var(--accent-cyan); font-size: 0.88rem;">${escapeHtml(sec)}:</strong>
            <p style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0.2rem;">${escapeHtml(text)}</p>
          </div>
        `).join('');
      }

      output.innerHTML = `
        <div class="summary-tier">
          <h3>📌 Executive Summary</h3>
          <p style="font-size: 0.92rem; color: #f1f5f9; line-height: 1.6;">${escapeHtml(summary.executive_summary)}</p>
        </div>

        ${findingsList ? `
          <div class="summary-tier">
            <h3>🔍 Key Insights & Findings</h3>
            <ul class="findings-list">${findingsList}</ul>
          </div>
        ` : ''}

        ${sectionsHtml ? `
          <div class="summary-tier">
            <h3>📑 Section-by-Section Breakdown</h3>
            <div style="margin-top: 0.65rem;">${sectionsHtml}</div>
          </div>
        ` : ''}

        ${actionsList ? `
          <div class="summary-tier">
            <h3>⚡ Recommended Action Items</h3>
            <ul class="action-list">${actionsList}</ul>
          </div>
        ` : ''}
      `;

    } catch (err) {
      output.innerHTML = `
        <div style="color: var(--accent-rose); padding: 1.5rem; text-align: center;">
          Summarization failed: ${err.message}
        </div>
      `;
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Document Comparison Studio                                                 */
  /* -------------------------------------------------------------------------- */
  setupComparisonHandlers() {
    const btnCompare = document.getElementById('btnExecuteCompare');
    if (btnCompare) {
      btnCompare.addEventListener('click', () => this.executeComparison());
    }
  }

  populateCompareDropdowns() {
    const selectA = document.getElementById('compareDocA');
    const selectB = document.getElementById('compareDocB');
    if (!selectA || !selectB) return;

    const options = '<option value="">-- Choose Document --</option>' +
      this.documents.map(d => `
        <option value="${d.metadata.document_id}">${escapeHtml(d.metadata.filename)}</option>
      `).join('');

    selectA.innerHTML = options;
    selectB.innerHTML = options;

    if (this.documents.length >= 2) {
      selectA.value = this.documents[0].metadata.document_id;
      selectB.value = this.documents[1].metadata.document_id;
    }
  }

  async executeComparison() {
    const selectA = document.getElementById('compareDocA');
    const selectB = document.getElementById('compareDocB');
    const output = document.getElementById('compareOutput');
    if (!selectA || !selectB || !output) return;

    if (!selectA.value || !selectB.value) {
      this.showToast('Please select both documents for comparison', 'info');
      return;
    }

    if (selectA.value === selectB.value) {
      this.showToast('Cannot compare a document with itself', 'info');
      return;
    }

    output.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <div class="status-dot" style="margin: 0 auto 0.5rem;"></div>
        Computing structural diffs and analyzing semantic variance...
      </div>
    `;

    try {
      const res = await Api.compare(selectA.value, selectB.value);
      const diff = res.structural_diff || {};
      const semantic = res.semantic_analysis || {};

      const simPercent = Math.round(diff.similarity_percentage || 0);

      // Render unified diff lines
      const diffLines = (diff.unified_diff || []).map(line => {
        if (line.startsWith('+')) return `<span class="diff-add">${escapeHtml(line)}</span>`;
        if (line.startsWith('-')) return `<span class="diff-del">${escapeHtml(line)}</span>`;
        return `<span>${escapeHtml(line)}</span>`;
      }).join('\n');

      output.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
          <div class="summary-tier" style="margin-bottom: 0; text-align: center;">
            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Structural Similarity</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono);">${simPercent}%</div>
          </div>
          <div class="summary-tier" style="margin-bottom: 0; text-align: center;">
            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Added Content</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: var(--accent-emerald); font-family: var(--font-mono);">+${diff.added_count || 0}</div>
          </div>
          <div class="summary-tier" style="margin-bottom: 0; text-align: center;">
            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Removed Content</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: var(--accent-rose); font-family: var(--font-mono);">-${diff.removed_count || 0}</div>
          </div>
        </div>

        <div class="summary-tier">
          <h3>⚖️ Semantic Variance Verdict</h3>
          <p style="font-size: 0.95rem; color: #fff; margin-bottom: 0.5rem; font-weight: 600;">
            ${escapeHtml(semantic.overall_verdict || 'Comparison Complete')}
          </p>
          <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6;">
            ${escapeHtml(semantic.synthesis || '')}
          </p>
        </div>

        <div class="summary-tier">
          <h3>📝 Unified Structural Text Diff</h3>
          <div class="diff-box">${diffLines || 'No textual differences detected.'}</div>
        </div>
      `;

    } catch (err) {
      output.innerHTML = `
        <div style="color: var(--accent-rose); padding: 1.5rem; text-align: center;">
          Comparison failed: ${err.message}
        </div>
      `;
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Health & Diagnostics Studio                                                */
  /* -------------------------------------------------------------------------- */
  async loadHealthDiagnostics() {
    const container = document.getElementById('healthDiagnosticsContent');
    if (!container) return;

    container.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <div class="status-dot" style="margin: 0 auto 0.5rem;"></div>
        Querying backend subsystem probes...
      </div>
    `;

    try {
      const [health, stats] = await Promise.all([
        Api.getHealth(),
        Api.getStats()
      ]);

      container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem;">
          <div class="summary-tier">
            <h3>Database & SQLite Subsystem</h3>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #cbd5e1;">
              <div><strong>Status:</strong> <span style="color: var(--accent-emerald);">READY</span></div>
              <div><strong>Total Documents:</strong> ${stats.total_documents}</div>
              <div><strong>Total Chunks:</strong> ${stats.total_chunks}</div>
              <div><strong>Total Word Count:</strong> ${stats.total_words}</div>
            </div>
          </div>

          <div class="summary-tier">
            <h3>Vector Storage Engine</h3>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #cbd5e1;">
              <div><strong>Status:</strong> <span style="color: var(--accent-emerald);">ONLINE</span></div>
              <div><strong>Active Embeddings:</strong> ${stats.vector_count}</div>
              <div><strong>Similarity Metric:</strong> Cosine (Unit Normalized Dot Product)</div>
            </div>
          </div>

          <div class="summary-tier">
            <h3>Sparse BM25 Index Engine</h3>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #cbd5e1;">
              <div><strong>Status:</strong> <span style="color: var(--accent-emerald);">ONLINE</span></div>
              <div><strong>Distinct Terms:</strong> ${stats.bm25_terms}</div>
              <div><strong>Algorithm:</strong> Okapi BM25 (k1=1.5, b=0.75)</div>
            </div>
          </div>

          <div class="summary-tier">
            <h3>AI Inference & Extraction</h3>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #cbd5e1;">
              <div><strong>Extraction Formats:</strong> PDF, DOCX, TXT, MD, Images (OCR)</div>
              <div><strong>Fallback Heuristics:</strong> Enabled</div>
              <div><strong>Readiness Checks:</strong> ${JSON.stringify(health.checks)}</div>
            </div>
          </div>
        </div>
      `;
    } catch (err) {
      container.innerHTML = `
        <div style="color: var(--accent-rose); padding: 1.5rem; text-align: center;">
          Failed to query diagnostics: ${err.message}
        </div>
      `;
    }
  }

  /* -------------------------------------------------------------------------- */
  /* Filters, Modals, & Toasts                                                  */
  /* -------------------------------------------------------------------------- */
  updateDocFilters() {
    const searchFilter = document.getElementById('searchDocFilter');
    if (searchFilter) {
      searchFilter.innerHTML = '<option value="">All Documents</option>' +
        this.documents.map(d => `
          <option value="${d.metadata.document_id}">${escapeHtml(d.metadata.filename)}</option>
        `).join('');
    }
  }

  setupModalHandlers() {
    const closeDocModal = document.getElementById('btnCloseDocModal');
    const docModal = document.getElementById('docDetailModal');
    if (closeDocModal && docModal) {
      closeDocModal.addEventListener('click', () => Viewer.closeDocumentModal());
      docModal.addEventListener('click', (e) => {
        if (e.target === docModal) Viewer.closeDocumentModal();
      });
    }

    const closeDrawer = document.getElementById('btnCloseDrawer');
    if (closeDrawer) {
      closeDrawer.addEventListener('click', () => Viewer.closeCitation());
    }
  }

  showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

// Instantiate and start app on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
});
