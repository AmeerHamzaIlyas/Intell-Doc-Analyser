"""OCR Extractor using Gemini Multimodal Vision with graceful fallback."""
from pathlib import Path
from typing import Optional
from PIL import Image
from app.config import settings
from app.core.exceptions import OCRError
from app.logging_config import get_logger
from app.services.extraction.base import BaseExtractor, ExtractedDocument, ExtractedPage

logger = get_logger(__name__)


class OCRExtractor(BaseExtractor):
    """Performs high-fidelity OCR and layout extraction using Gemini Multimodal Vision."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text from an image or scanned document."""
        # 1. Inspect image properties with Pillow
        try:
            with Image.open(file_path) as img:
                img_format = img.format
                width, height = img.size
        except Exception as e:
            raise OCRError(f"Failed to open image for OCR: {e}")

        extracted_text = ""
        has_ocr = False

        # 2. Use Gemini Vision if API key is configured
        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                
                with open(file_path, "rb") as f:
                    image_bytes = f.read()

                # Infer mime type
                mime = f"image/{img_format.lower()}" if img_format else "image/jpeg"
                if mime == "image/jpg":
                    mime = "image/jpeg"

                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
                prompt = (
                    "Perform high-accuracy optical character recognition (OCR) on this document image. "
                    "Extract all readable text, preserve headings, bullet points, and structure. "
                    "If there are any tables, format them neatly as Markdown tables. "
                    "Output only the extracted text without conversational introductions."
                )

                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[image_part, prompt]
                )
                extracted_text = response.text.strip() if response.text else ""
                has_ocr = True
                logger.info("Successfully performed Gemini Vision OCR on %s (%d chars)", file_path.name, len(extracted_text))
            except Exception as e:
                logger.warning("Gemini Vision OCR failed for %s: %s. Falling back to basic metadata.", file_path.name, e)
                extracted_text = f"[Scanned Image: {file_path.name}, Dimensions: {width}x{height}, Format: {img_format}. OCR processing error: {e}]"
        else:
            extracted_text = (
                f"[Image Document: {file_path.name}\n"
                f"Dimensions: {width}x{height} pixels, Format: {img_format}.\n"
                f"Notice: GEMINI_API_KEY is required for multimodal OCR text extraction.]"
            )

        page = ExtractedPage(
            page_number=1,
            text=extracted_text,
            tables=[],
            section_hints=[],
            has_ocr=has_ocr
        )

        return ExtractedDocument(
            pages=[page],
            metadata={
                "image_width": width,
                "image_height": height,
                "image_format": img_format,
                "ocr_performed": has_ocr
            },
            total_pages=1,
            full_text=extracted_text,
            title=file_path.stem
        )
