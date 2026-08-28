from __future__ import annotations

import fitz  # PyMuPDF


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a text-based PDF.

    Raises:
        ValueError: if the PDF cannot be opened or contains no extractable text.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Could not open the PDF.") from exc

    pages: list[str] = []
    try:
        for page in doc:
            pages.append(page.get_text("text"))
    finally:
        doc.close()

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. The resume may be a scanned/image-only PDF."
        )
    return text
