"""Intelligent structure-aware and recursive document chunker."""
import re
from typing import List, Optional
from app.config import settings
from app.models.document import DocumentChunk, DocumentSection
from app.services.extraction.base import ExtractedPage
from app.services.processing.structure import StructureDetector


class IntelligentChunker:
    """Splits documents into semantic chunks while maintaining structure, overlap, and provenance."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

    def chunk_document(
        self,
        document_id: str,
        pages: List[ExtractedPage],
        sections: Optional[List[DocumentSection]] = None
    ) -> List[DocumentChunk]:
        """Chunk all pages of a document, tracking active sections and page provenance."""
        if not pages:
            return []

        chunks: List[DocumentChunk] = []
        chunk_index = 0
        current_section = "Introduction"
        global_char_offset = 0

        # Build section lookup if provided
        section_positions = sections or StructureDetector.detect_sections(pages)
        section_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split page into structural blocks (paragraphs / tables)
            blocks = self._split_into_blocks(page_text)
            
            current_chunk_text = ""
            current_start_offset = global_char_offset

            for block in blocks:
                # Check if block is a heading
                _, heading_title = StructureDetector._classify_heading(block.strip())
                if heading_title:
                    current_section = heading_title

                # If block alone is larger than chunk_size, recursively split into subchunks
                if len(block) > self.chunk_size:
                    # Flush current buffer first
                    if current_chunk_text.strip():
                        chunks.append(
                            self._create_chunk(
                                document_id=document_id,
                                chunk_index=chunk_index,
                                text=current_chunk_text.strip(),
                                page_number=page.page_number,
                                section_title=current_section,
                                char_start=current_start_offset,
                                char_end=current_start_offset + len(current_chunk_text),
                            )
                        )
                        chunk_index += 1
                        current_chunk_text = ""

                    # Split large block recursively (by sentences, words, or character windows)
                    subchunks = self._split_into_subchunks(block)
                    for sub in subchunks:
                        if not sub.strip():
                            continue
                        chunks.append(
                            self._create_chunk(
                                document_id=document_id,
                                chunk_index=chunk_index,
                                text=sub.strip(),
                                page_number=page.page_number,
                                section_title=current_section,
                                char_start=current_start_offset,
                                char_end=current_start_offset + len(sub),
                            )
                        )
                        chunk_index += 1
                        current_start_offset += len(sub)
                    current_chunk_text = ""
                else:
                    # Fits or extends current buffer
                    if len(current_chunk_text) + len(block) + 2 > self.chunk_size and current_chunk_text:
                        chunks.append(
                            self._create_chunk(
                                document_id=document_id,
                                chunk_index=chunk_index,
                                text=current_chunk_text.strip(),
                                page_number=page.page_number,
                                section_title=current_section,
                                char_start=current_start_offset,
                                char_end=current_start_offset + len(current_chunk_text),
                            )
                        )
                        chunk_index += 1
                        # Retain overlap from end of current chunk
                        overlap = self._get_overlap(current_chunk_text)
                        current_chunk_text = (overlap + "\n\n" + block).strip()
                        current_start_offset += len(current_chunk_text)
                    else:
                        if current_chunk_text:
                            current_chunk_text += "\n\n" + block
                        else:
                            current_chunk_text = block

            # Flush remaining buffer for this page
            if current_chunk_text.strip():
                chunks.append(
                    self._create_chunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=current_chunk_text.strip(),
                        page_number=page.page_number,
                        section_title=current_section,
                        char_start=current_start_offset,
                        char_end=current_start_offset + len(current_chunk_text),
                    )
                )
                chunk_index += 1

            global_char_offset += len(page_text) + 2

        return chunks

    def _split_into_subchunks(self, text: str) -> List[str]:
        """Recursively split large text into chunks <= chunk_size, breaking on sentences, words, or chars."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        # 1. Try splitting by sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) > 1:
            chunks: List[str] = []
            curr = ""
            for s in sentences:
                if len(s) > self.chunk_size:
                    if curr.strip():
                        chunks.append(curr.strip())
                        curr = ""
                    chunks.extend(self._split_into_subchunks(s))
                elif len(curr) + len(s) + 1 > self.chunk_size and curr:
                    chunks.append(curr.strip())
                    curr = (self._get_overlap(curr) + " " + s).strip()
                else:
                    curr = f"{curr} {s}".strip() if curr else s
            if curr.strip():
                chunks.append(curr.strip())
            return chunks

        # 2. Try splitting by words
        words = text.split()
        if len(words) > 1:
            chunks: List[str] = []
            curr = ""
            for w in words:
                if len(w) > self.chunk_size:
                    if curr.strip():
                        chunks.append(curr.strip())
                        curr = ""
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for i in range(0, len(w), step):
                        chunks.append(w[i:i + self.chunk_size])
                elif len(curr) + len(w) + 1 > self.chunk_size and curr:
                    chunks.append(curr.strip())
                    curr = (self._get_overlap(curr) + " " + w).strip()
                else:
                    curr = f"{curr} {w}".strip() if curr else w
            if curr.strip():
                chunks.append(curr.strip())
            return chunks

        # 3. Fallback: hard character slice for continuous unbroken tokens
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

    def _split_into_blocks(self, text: str) -> List[str]:
        """Split text into paragraph and table blocks."""
        # Split on 2 or more newlines
        raw_blocks = re.split(r"\n\s*\n", text)
        return [b.strip() for b in raw_blocks if b.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split long paragraph into sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, text: str) -> str:
        """Extract trailing characters for sliding window overlap."""
        if len(text) <= self.chunk_overlap:
            return text
        # Try to break at a sentence or word boundary within the overlap window
        tail = text[-self.chunk_overlap:]
        # Find first whitespace or newline in the tail to avoid half-words
        space_pos = tail.find(" ")
        if space_pos != -1 and space_pos < len(tail) - 10:
            return tail[space_pos + 1:]
        return tail

    @staticmethod
    def _create_chunk(
        document_id: str,
        chunk_index: int,
        text: str,
        page_number: int,
        section_title: str,
        char_start: int,
        char_end: int,
    ) -> DocumentChunk:
        """Helper to construct a fully qualified DocumentChunk."""
        words = text.split()
        approx_tokens = max(1, int(len(text) / 4))
        return DocumentChunk(
            chunk_id=f"{document_id}_{chunk_index}",
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            page_number=page_number,
            section_title=section_title or "General",
            token_count=approx_tokens,
            char_start=char_start,
            char_end=char_end,
            metadata={
                "word_count": len(words),
                "has_table": "|" in text and "---" in text,
            }
        )


# Global chunker instance
chunker = IntelligentChunker()
