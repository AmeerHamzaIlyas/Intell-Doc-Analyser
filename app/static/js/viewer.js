/**
 * Viewer & Citation Inspector Module
 * Handles modal displays, provenance drawers, and chunk inspections.
 */

export const Viewer = {
  activeCitations: new Map(),

  /**
   * Register citations from current RAG or Search response
   */
  registerCitations(citations) {
    this.activeCitations.clear();
    if (Array.isArray(citations)) {
      citations.forEach(cit => {
        this.activeCitations.set(cit.citation_id, cit);
      });
    }
  },

  /**
   * Open the side-drawer to inspect source citation provenance
   */
  openCitation(citationId) {
    const citation = this.activeCitations.get(parseInt(citationId, 10));
    const drawer = document.getElementById('citationDrawer');
    const content = document.getElementById('drawerContent');

    if (!drawer || !content) return;

    if (!citation) {
      content.innerHTML = `
        <div style="color: var(--text-muted); padding: 1rem 0;">
          <p>Citation #${citationId} details are unavailable or unreferenced in this context window.</p>
        </div>
      `;
    } else {
      content.innerHTML = `
        <div style="margin-bottom: 1.25rem;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <span class="status-pill" style="font-size: 0.75rem;">Citation #${citation.citation_id}</span>
            <span class="badge-score">Confidence: ${Math.round(citation.confidence_score * 100)}%</span>
          </div>
          <h3 style="font-size: 1.1rem; color: #fff; word-break: break-all;">${citation.filename}</h3>
          <p style="font-size: 0.8rem; color: var(--text-dim); margin-top: 0.25rem;">
            Document ID: <code>${citation.document_id}</code>
          </p>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1.25rem;">
          <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm);">
            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Page Number</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono);">
              Page ${citation.page_number}
            </div>
          </div>
          <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm);">
            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Section</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
              ${citation.section_title || 'General'}
            </div>
          </div>
        </div>

        <div>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
            Exact Grounded Snippet
          </div>
          <div class="result-snippet" style="font-size: 0.88rem; max-height: 380px; overflow-y: auto;">
            ${escapeHtml(citation.snippet)}
          </div>
        </div>
      `;
    }

    drawer.classList.add('open');
  },

  /**
   * Open citation matching document name and page number
   */
  openCitationByDocPage(docName, pageNum) {
    let matchedId = null;
    for (const [id, cit] of this.activeCitations.entries()) {
      if (cit.filename.toLowerCase() === docName.toLowerCase() && cit.page_number === pageNum) {
        matchedId = id;
        break;
      }
    }
    if (matchedId !== null) {
      this.openCitation(matchedId);
    } else if (this.activeCitations.size > 0) {
      // Fallback to first citation
      const firstId = this.activeCitations.keys().next().value;
      this.openCitation(firstId);
    }
  },

  /**
   * Close the citation drawer
   */
  closeCitation() {
    const drawer = document.getElementById('citationDrawer');
    if (drawer) drawer.classList.remove('open');
  },

  /**
   * Show document modal with full metadata and chunks list
   */
  openDocumentModal(docDetail, chunks = []) {
    const modal = document.getElementById('docDetailModal');
    const body = document.getElementById('docModalBody');
    const title = document.getElementById('docModalTitle');
    if (!modal || !body || !title) return;

    const m = docDetail.metadata;
    title.textContent = m.filename;

    let chunksHtml = '';
    if (chunks.length === 0) {
      chunksHtml = '<p style="color: var(--text-muted);">No chunks found for this document.</p>';
    } else {
      chunksHtml = chunks.map((c, idx) => `
        <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-glass); border-radius: var(--radius-sm); padding: 0.85rem; margin-bottom: 0.75rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; font-size: 0.75rem; color: var(--text-dim);">
            <span><strong>Chunk #${c.chunk_index + 1}</strong> | Page ${c.page_number} (${c.section_title || 'General'})</span>
            <span>${c.token_count || 0} tokens</span>
          </div>
          <div style="font-size: 0.82rem; color: #cbd5e1; font-family: var(--font-mono); white-space: pre-wrap; line-height: 1.5;">${escapeHtml(c.text)}</div>
        </div>
      `).join('');
    }

    body.innerHTML = `
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.5rem;">
        <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-size: 0.68rem; color: var(--text-muted);">STATUS</div>
          <div style="font-size: 0.88rem; font-weight: 700; color: var(--accent-emerald);">${m.status}</div>
        </div>
        <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-size: 0.68rem; color: var(--text-muted);">PAGES</div>
          <div style="font-size: 0.88rem; font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono);">${m.page_count}</div>
        </div>
        <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-size: 0.68rem; color: var(--text-muted);">WORDS</div>
          <div style="font-size: 0.88rem; font-weight: 700; color: #fff; font-family: var(--font-mono);">${m.word_count}</div>
        </div>
        <div style="background: rgba(255,255,255,0.04); padding: 0.65rem; border-radius: var(--radius-sm); text-align: center;">
          <div style="font-size: 0.68rem; color: var(--text-muted);">CHUNKS</div>
          <div style="font-size: 0.88rem; font-weight: 700; color: var(--accent-secondary); font-family: var(--font-mono);">${docDetail.chunk_count}</div>
        </div>
      </div>

      <div style="margin-bottom: 1.5rem;">
        <h4 style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Extraction & Document Metadata</h4>
        <div style="font-size: 0.82rem; line-height: 1.8; color: #94a3b8;">
          <div><strong>MIME Type:</strong> ${m.mime_type}</div>
          <div><strong>SHA-256 Hash:</strong> <code style="font-size: 0.75rem;">${m.file_hash}</code></div>
          <div><strong>Detected Language:</strong> ${m.detected_language}</div>
          <div><strong>Estimated Reading Time:</strong> ${m.reading_time_minutes.toFixed(1)} mins</div>
          <div><strong>Uploaded At:</strong> ${m.created_at}</div>
        </div>
      </div>

      <div>
        <h4 style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.75rem; text-transform: uppercase;">Indexed Semantic Chunks (${chunks.length})</h4>
        <div style="max-height: 320px; overflow-y: auto; padding-right: 0.5rem;">
          ${chunksHtml}
        </div>
      </div>
    `;

    modal.style.display = 'flex';
  },

  closeDocumentModal() {
    const modal = document.getElementById('docDetailModal');
    if (modal) modal.style.display = 'none';
  }
};

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
