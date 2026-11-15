
from pathlib import Path

import pymupdf


def extract_pdf(pdf_path: str | Path) -> dict:
    """
    Extract text and metadata from a PDF document.
    """

    pdf_path = Path(pdf_path)

    # Validate file existence
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Validate file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Open PDF
    with pymupdf.open(pdf_path) as document:

        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        pages = []

        # Extract text page by page
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            pages.append({
                "page_number": page_number,
                "text": text
            })

        # Combine all pages
        full_text = "\n\n".join(
            page["text"] for page in pages
        )

        return {
            "source_file": pdf_path.name,
            "total_pages": len(pages),
            "metadata": document.metadata,
            "pages": pages,
            "full_text": full_text
        }
