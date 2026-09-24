/**
 * API Client Module for Intelligent Document Understanding & Analysis System
 * Centralized fetch wrappers with robust error handling and progress hooks.
 */

const API_BASE = window.location.origin;

export const Api = {
  /**
   * Check system health and readiness
   */
  async getHealth() {
    const res = await fetch(`${API_BASE}/health/ready`);
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return await res.json();
  },

  /**
   * Get system operational statistics
   */
  async getStats() {
    const res = await fetch(`${API_BASE}/api/documents/stats/overview`);
    if (!res.ok) throw new Error(`Failed to load system stats (${res.status})`);
    return await res.json();
  },

  /**
   * List ingested documents
   */
  async listDocuments(limit = 50, offset = 0) {
    const res = await fetch(`${API_BASE}/api/documents?limit=${limit}&offset=${offset}`);
    if (!res.ok) throw new Error(`Failed to list documents (${res.status})`);
    return await res.json();
  },

  /**
   * Get detailed metadata for a single document
   */
  async getDocument(docId) {
    const res = await fetch(`${API_BASE}/api/documents/${docId}`);
    if (!res.ok) throw new Error(`Document not found (${res.status})`);
    return await res.json();
  },

  /**
   * Get all chunks for a document
   */
  async getChunks(docId) {
    const res = await fetch(`${API_BASE}/api/documents/${docId}/chunks`);
    if (!res.ok) throw new Error(`Failed to load chunks (${res.status})`);
    return await res.json();
  },

  /**
   * Upload document with XMLHttpRequest progress tracking
   */
  uploadDocument(file, onProgress = null) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.open('POST', `${API_BASE}/api/documents/upload`);

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            onProgress(percent);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const json = JSON.parse(xhr.responseText);
            resolve(json);
          } catch (e) {
            reject(new Error("Malformed server response"));
          }
        } else {
          let errMsg = `Upload failed (${xhr.status})`;
          try {
            const errObj = JSON.parse(xhr.responseText);
            errMsg = errObj.detail || errObj.message || errMsg;
          } catch {}
          reject(new Error(errMsg));
        }
      };

      xhr.onerror = () => reject(new Error("Network connection error during upload."));
      xhr.send(formData);
    });
  },

  /**
   * Delete document and cascade indices
   */
  async deleteDocument(docId) {
    const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error(`Failed to delete document (${res.status})`);
    return await res.json();
  },

  /**
   * Execute hybrid, dense, or sparse search
   */
  async search({ query, mode = "hybrid", topK = 5, documentIds = null, minScore = 0.0 }) {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        mode,
        top_k: parseInt(topK, 10),
        document_ids: documentIds && documentIds.length ? documentIds : null,
        min_score: parseFloat(minScore)
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Search failed (${res.status})`);
    }
    return await res.json();
  },

  /**
   * Grounded RAG Query with Citation Extraction
   */
  async queryRAG({ query, documentIds = null, topK = 5, conversationHistory = [] }) {
    const res = await fetch(`${API_BASE}/api/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        document_ids: documentIds && documentIds.length ? documentIds : null,
        top_k: parseInt(topK, 10),
        conversation_history: conversationHistory
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `RAG Query failed (${res.status})`);
    }
    return await res.json();
  },

  /**
   * Generate multi-tier document summary
   */
  async summarize(docId, forceRefresh = false) {
    const res = await fetch(`${API_BASE}/api/summarize/${docId}?force_refresh=${forceRefresh}`, {
      method: 'POST'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Summarization failed (${res.status})`);
    }
    return await res.json();
  },

  /**
   * Compare two documents structurally and semantically
   */
  async compare(docIdA, docIdB) {
    const res = await fetch(`${API_BASE}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id_a: docIdA, doc_id_b: docIdB })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Comparison failed (${res.status})`);
    }
    return await res.json();
  }
};
