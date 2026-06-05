import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also extract table text
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n".join(paragraphs)


def extract_text_from_xlsx(file_bytes: bytes) -> str:
    import pandas as pd
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    parts = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, dtype=str)
        df = df.fillna("")
        parts.append(f"Sheet: {sheet_name}")
        # Column headers
        parts.append(" | ".join(str(c) for c in df.columns))
        # Row data
        for _, row in df.iterrows():
            row_str = " | ".join(str(v) for v in row.values if str(v).strip())
            if row_str:
                parts.append(row_str)
    return "\n".join(parts)


def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError("pytesseract and Pillow are required for image processing.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed. Install it with:\n"
            "  macOS: brew install tesseract\n"
            "  Ubuntu: sudo apt-get install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    elif ext in (".xlsx", ".xls"):
        return extract_text_from_xlsx(file_bytes)
    else:
        return extract_text_from_image(file_bytes)
