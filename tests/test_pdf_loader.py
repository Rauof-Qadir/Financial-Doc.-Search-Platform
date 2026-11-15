

import pymupdf
import pytest

from src.ingestion.pdf_loader import extract_pdf


def create_test_pdf(path):
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (50, 50),
        "Annual Financial Report 2026"
    )

    document.save(path)
    document.close()


def test_extract_pdf_successfully(tmp_path):

    pdf_path = tmp_path / "financial_report.pdf"

    create_test_pdf(pdf_path)

    result = extract_pdf(pdf_path)

    assert result["source_file"] == "financial_report.pdf"
    assert result["total_pages"] == 1
    assert "Annual Financial Report 2026" in result["full_text"]


def test_missing_pdf():

    with pytest.raises(FileNotFoundError):
        extract_pdf("missing_report.pdf")


def test_invalid_file_extension(tmp_path):

    file_path = tmp_path / "report.txt"
    file_path.write_text("This is not a PDF")

    with pytest.raises(ValueError):
        extract_pdf(file_path)
