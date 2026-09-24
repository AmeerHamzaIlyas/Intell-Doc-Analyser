/**
 * RAG Chat Studio Controller
 * Handles conversation state, grounding tag rendering, and citation link synthesis.
 */

import { Api } from './api.js';
import { Viewer } from './viewer.js';

export const RagChat = {
  conversationHistory: [],
  selectedDocIds: [],

  init() {
    const sendBtn = document.getElementById('btnSendChat');
    const input = document.getElementById('chatInput');
    const clearBtn = document.getElementById('btnClearChat');

    if (sendBtn && input) {
      sendBtn.addEventListener('click', () => this.sendMessage());
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearHistory());
    }

    // Expose citation click globally for synthesized HTML buttons
    window.viewCitation = (id) => {
      Viewer.openCitation(id);
    };
    window.viewCitationByDocPage = (docName, pageNum) => {
      Viewer.openCitationByDocPage(docName, pageNum);
    };
  },

  async sendMessage() {
    const input = document.getElementById('chatInput');
    const historyContainer = document.getElementById('chatHistory');
    if (!input || !historyContainer) return;

    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    input.disabled = true;

    // Render User Bubble
    this.appendMessage('user', query);

    // Render Loading Bubble
    const loadingId = 'loading-' + Date.now();
    this.appendLoading(loadingId);

    try {
      const topKInput = document.getElementById('ragTopK');
      const topK = topKInput ? parseInt(topKInput.value, 10) : 5;

      const response = await Api.queryRAG({
        query,
        documentIds: this.selectedDocIds,
        topK,
        conversationHistory: this.conversationHistory
      });

      this.removeLoading(loadingId);

      // Register citations for drawer
      Viewer.registerCitations(response.citations);

      // Append assistant answer
      this.appendAssistantResponse(response);

      // Record in local conversation history
      this.conversationHistory.push({ role: 'user', content: query });
      this.conversationHistory.push({ role: 'assistant', content: response.answer });

    } catch (err) {
      this.removeLoading(loadingId);
      this.appendMessage('assistant', `⚠️ **Error:** ${err.message || 'Failed to generate answer'}`);
    } finally {
      input.disabled = false;
      input.focus();
    }
  },

  appendMessage(role, text) {
    const container = document.getElementById('chatHistory');
    if (!container) return;

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble chat-bubble-${role}`;
    bubble.innerHTML = this.formatMarkdown(text);
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
  },

  appendLoading(id) {
    const container = document.getElementById('chatHistory');
    if (!container) return;

    const bubble = document.createElement('div');
    bubble.id = id;
    bubble.className = 'chat-bubble chat-bubble-assistant';
    bubble.innerHTML = `
      <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-muted);">
        <div class="status-dot"></div>
        <span>Retrieving hybrid context and synthesizing grounded answer...</span>
      </div>
    `;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
  },

  removeLoading(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  },

  appendAssistantResponse(ragRes) {
    const container = document.getElementById('chatHistory');
    if (!container) return;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble chat-bubble-assistant';

    // Status Tag
    let statusClass = 'tag-grounded';
    let statusLabel = 'GROUNDED EVIDENCE';
    if (ragRes.grounding_status === 'PARTIAL') {
      statusClass = 'tag-partial';
      statusLabel = 'PARTIAL EVIDENCE';
    } else if (ragRes.grounding_status === 'NO_EVIDENCE') {
      statusClass = 'tag-no-evidence';
      statusLabel = 'NO DIRECT EVIDENCE';
    }

    const tagHtml = `<div class="grounding-status-tag ${statusClass}">
      ● ${statusLabel} (${ragRes.latency_ms.toFixed(0)}ms)
    </div>`;

    // Process citations in answer
    const formattedAnswer = this.formatAnswerWithCitations(ragRes.answer);

    // Citations Footer Pill Bar
    let citationFooter = '';
    if (ragRes.citations && ragRes.citations.length > 0) {
      const pills = ragRes.citations.map(c => `
        <button class="citation-link" onclick="window.viewCitation(${c.citation_id})">
          📄 [Citation: ${c.citation_id}] Page ${c.page_number}
        </button>
      `).join(' ');

      citationFooter = `
        <div style="margin-top: 0.85rem; padding-top: 0.65rem; border-top: 1px solid var(--border-glass); font-size: 0.75rem; color: var(--text-dim);">
          <div style="margin-bottom: 0.35rem; text-transform: uppercase; font-size: 0.68rem; color: var(--text-muted);">Verified Citations:</div>
          <div style="display: flex; flex-wrap: wrap; gap: 0.35rem;">
            ${pills}
          </div>
        </div>
      `;
    }

    bubble.innerHTML = tagHtml + `<div>${formattedAnswer}</div>` + citationFooter;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
  },

  formatAnswerWithCitations(rawText) {
    if (!rawText) return '';
    let text = this.formatMarkdown(rawText);

    // Replace [Citation: X] with interactive buttons
    text = text.replace(/\[Citation:\s*(\d+)\]/gi, (match, id) => {
      return `<button class="citation-link" onclick="window.viewCitation(${id})">📄 Citation ${id}</button>`;
    });

    // Replace [Doc: <filename>, Page: <page>] with interactive buttons
    text = text.replace(/\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)(?:,\s*Section:\s*([^\]]+))?\]/gi, (match, docName, pageNum) => {
      const cleanDoc = docName.trim();
      return `<button class="citation-link" onclick="window.viewCitationByDocPage('${cleanDoc}', ${parseInt(pageNum, 10)})">📄 ${cleanDoc} (p. ${pageNum})</button>`;
    });

    return text;
  },

  formatMarkdown(text) {
    if (!text) return '';
    let escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Bold
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Code blocks
    escaped = escaped.replace(/```([\s\S]*?)```/g, '<pre class="result-snippet">$1</pre>');
    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 0.1rem 0.3rem; border-radius: 4px; font-size: 0.85em;">$1</code>');
    // Line breaks
    escaped = escaped.replace(/\n\n/g, '<br/><br/>').replace(/\n/g, '<br/>');

    return escaped;
  },

  clearHistory() {
    this.conversationHistory = [];
    const container = document.getElementById('chatHistory');
    if (container) {
      container.innerHTML = `
        <div class="chat-bubble chat-bubble-assistant">
          👋 Welcome to the <strong>Grounded RAG Studio</strong>. Ask questions across your ingested documents.
          Every answer is strictly verifiable with page numbers, exact source snippets, and grounding confidence scores.
        </div>
      `;
    }
  }
};
