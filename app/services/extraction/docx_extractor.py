"""DOCX document extractor preserving headings, tables, and lists."""
from pathlib import Path
from typing import List
import zipfile
import docx
from app.core.exceptions import ExtractionError, FileCorruptedError
from app.services.extraction.base import BaseExtractor, ExtractedDocument, ExtractedPage


class DocxExtractor(BaseExtractor):
    """Extracts structured text, headings, and tables from Word (.docx) files."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        try:
            doc = docx.Document(str(file_path))
        except (zipfile.BadZipFile, docx.opc.exceptions.PackageNotFoundError) as e:
            raise FileCorruptedError(f"Malformed or corrupted DOCX archive: {e}")
        except Exception as e:
            raise FileCorruptedError(f"Failed to parse DOCX document: {e}")

        extracted_text_blocks: List[str] = []
        section_hints: List[str] = []
        tables_data: List[List[List[str]]] = []

        # Extract paragraphs & headings
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            style_name = para.style.name if para.style else ""
            if "Heading" in style_name:
                level = 1
                try:
                    level = int(style_name.replace("Heading", "").strip())
                except ValueError:
                    pass
                prefix = "#" * level
                heading_line = f"{prefix} {text}"
                extracted_text_blocks.append(heading_line)
                section_hints.append(text)
            else:
                extracted_text_blocks.append(text)

        # Extract tables and convert to Markdown table format
        for table in doc.tables:
            table_rows: List[List[str]] = []
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                table_rows.append(row_cells)
            
            if table_rows:
                tables_data.append(table_rows)
                # Format as Markdown table
                md_table_lines: List[str] = []
                header = table_rows[0]
                md_table_lines.append("| " + " | ".join(header) + " |")
                md_table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for row in table_rows[1:]:
                    # Ensure same column count
                    padded = row + [""] * (len(header) - len(row))
                    md_table_lines.append("| " + " | ".join(padded[:len(header)]) + " |")
                
                extracted_text_blocks.append("\n".join(md_table_lines))

        full_text = "\n\n".join(extracted_text_blocks)

        # Metadata from core properties
        metadata = {}
        title = None
        author = None
        try:
            props = doc.core_properties
            if props.title:
                title = props.title
                metadata["title"] = props.title
            if props.author:
                author = props.author
                metadata["author"] = props.author
        except Exception:
            pass

        if not title and section_hints:
            title = section_hints[0]
        elif not title:
            title = file_path.stem

        page = ExtractedPage(
            page_number=1,
            text=full_text,
            tables=tables_data,
            section_hints=section_hints
        )

        return ExtractedDocument(
            pages=[page],
            metadata=metadata,
            total_pages=1,
            full_text=full_text,
            title=title,
            author=author
        )
