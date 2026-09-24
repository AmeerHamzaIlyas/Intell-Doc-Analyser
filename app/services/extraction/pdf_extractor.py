"""PDF extractor using pypdf with page tracking and metadata extraction."""
from pathlib import Path
from typing import List, Optional
import pypdf
from app.core.exceptions import ExtractionError, FileCorruptedError
from app.logging_config import get_logger
from app.services.extraction.base import BaseExtractor, ExtractedDocument, ExtractedPage
from app.services.extraction.ocr_extractor import OCRExtractor

logger = get_logger(__name__)


class PDFExtractor(BaseExtractor):
    """Extracts text page-by-page from PDF documents, with fallback OCR for scanned pages."""

    def __init__(self, ocr_extractor: Optional[OCRExtractor] = None):
        self.ocr_extractor = ocr_extractor or OCRExtractor()

    def extract(self, file_path: Path) -> ExtractedDocument:
        try:
            reader = pypdf.PdfReader(str(file_path))
            if reader.is_encrypted:
                try:
                    decrypt_status = reader.decrypt("")
                    if decrypt_status == 0:
                        raise ExtractionError("PDF is password-protected and cannot be read without a password.")
                except Exception as dec_err:
                    raise ExtractionError(f"PDF is password-protected: {dec_err}")
        except ExtractionError:
            raise
        except (pypdf.errors.PdfReadError, pypdf.errors.EmptyFileError) as e:
            raise FileCorruptedError(f"Corrupted or malformed PDF document: {e}")
        except Exception as e:
            raise FileCorruptedError(f"Failed to read PDF file: {e}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise FileCorruptedError("PDF contains zero pages")

        pages: List[ExtractedPage] = []
        full_text_blocks: List[str] = []
        section_hints: List[str] = []

        # Try to read outline / bookmarks
        try:
            outline = reader.outline
            if outline:
                for item in outline:
                    if hasattr(item, "title") and item.title:
                        section_hints.append(item.title)
        except Exception:
            pass

        # Extract page by page
        for page_num, page in enumerate(reader.pages, start=1):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Error extracting text from page %d of %s: %s", page_num, file_path.name, e)

            # Check if page is empty or likely scanned
            cleaned_text = text.strip()
            has_ocr = False
            
            # If the page has almost no text and reader extracted images, note it
            if len(cleaned_text) < 30 and len(page.images) > 0:
                logger.debug("Page %d of %s has minimal text and %d images; flagging for OCR", page_num, file_path.name, len(page.images))
                # If images exist on scanned page, extract first image to temp for OCR
                try:
                    for img_obj in page.images:
                        # Convert pypdf image to bytes
                        import io
                        from PIL import Image
                        img = Image.open(io.BytesIO(img_obj.data))
                        # If OCR is available, run OCR on the main image
                        if self.ocr_extractor.api_key:
                            temp_img_path = file_path.parent / f"temp_{file_path.stem}_p{page_num}.png"
                            img.save(temp_img_path)
                            try:
                                ocr_res = self.ocr_extractor.extract(temp_img_path)
                                cleaned_text = ocr_res.full_text
                                has_ocr = True
                            finally:
                                temp_img_path.unlink(missing_ok=True)
                        break
                except Exception as img_err:
                    logger.warning("Could not extract image for OCR on page %d: %s", page_num, img_err)

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=cleaned_text,
                    tables=[],
                    section_hints=[],
                    has_ocr=has_ocr
                )
            )
            if cleaned_text:
                full_text_blocks.append(f"--- Page {page_num} ---\n{cleaned_text}")

        # Metadata from doc info
        metadata = {}
        title = None
        author = None
        try:
            if reader.metadata:
                if reader.metadata.title:
                    title = reader.metadata.title
                    metadata["title"] = title
                if reader.metadata.author:
                    author = reader.metadata.author
                    metadata["author"] = author
        except Exception:
            pass

        if not title and section_hints:
            title = section_hints[0]
        elif not title:
            title = file_path.stem

        return ExtractedDocument(
            pages=pages,
            metadata=metadata,
            total_pages=total_pages,
            full_text="\n\n".join(full_text_blocks),
            title=title,
            author=author
        )
